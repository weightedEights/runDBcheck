#include "integrator.h"
#include "radac.h"
#include "errorcodes.h"
#include "params.h"

// Dll storage space
static localvars lv;

// test functions
void testSamplebuffer(dworddoublebuffer *sbuf)
{
	DWORD *ib;
	int s;

	ib = sbuf->inbuf;
	ib[0] = 0x00000100;
	ib[1] = 32;
	ib[2] = 0;
	ib[3] = 0;
	ib[4] = 0x00008001;
	ib[10] = 1;
	ib[11] = 0;
	ib[12] = sbuf->size-42;

	for (s=32;s<sbuf->size;s++)
		ib[s] = 0x0002fffe;
};


// misc. helper functions

double getTime()
{
	int cnt;
	DWORD msw,lsw,chk;
	double secs;

	for (cnt=0;cnt<5;cnt++)
	{
		readRegisters(MEMREG,rcTimeMsw,1,&msw);
		readRegisters(MEMREG,rcTimeLsw,1,&lsw);
		readRegisters(MEMREG,rcTimeMsw,1,&chk);
		if (msw == chk)
		{
			secs = (double)(0x0fffffff & msw) * 4096.0 + (double)((0xfff00000 & lsw) >> 20);
			return secs;
		}
	}
	return 0.0;
}



DWORD translateError(DWORD code,char *ret,int bufsize)
//
// Translate a win32 error message to a string
// Look for custom error code indicator to deal
// with internal errors
//
{
	DWORD fac;
	char msg[MSGSIZE];

	if (code == 0)
	{
		sprintf_s(msg,MSGSIZE,"%s","ok");
		strcpy_s(ret,bufsize,msg);
		return strlen(msg);
	};

	if ((code & 0x20000000) == 0x20000000)
	{
		fac = code & 0x0fff0000;
		switch(fac)
		{
			case modeprocess:
				switch(code)
				{
					case ecBeamcodeNotInList:
						sprintf_s(msg,MSGSIZE,"%s","Mode process error: beamcode not in list");
						strcpy_s(ret,bufsize,msg);
						return strlen(msg);
					case ecIndexOutOfBounds:
						sprintf_s(msg,MSGSIZE,"%s","Mode process error: index out of bounds");
						strcpy_s(ret,bufsize,msg);
						return strlen(msg);
					case ecSamplesMisalignment:
						sprintf_s(msg,MSGSIZE,"%s","Mode process error: samples misaligned");
						strcpy_s(ret,bufsize,msg);
						return strlen(msg);						
					default:
						sprintf_s(msg,MSGSIZE,"%s","Mode process error: undefined");
						strcpy_s(ret,bufsize,msg);
						return strlen(msg);
				};
			case integrator:
				switch(code)
				{
					case ecSampleBufferCorrupt:
						sprintf_s(msg,MSGSIZE,"%s","Integrator error: samples buffer corrupted");
						strcpy_s(ret,bufsize,msg);
						return strlen(msg);						
					default:
						sprintf_s(msg,MSGSIZE,"Integrator error: undefined");
						strcpy_s(ret,bufsize,msg);
						return strlen(msg);
				}
			default:
				sprintf_s(msg,MSGSIZE,"Unknown error: undefined");
				strcpy_s(ret,bufsize,msg);
				return strlen(msg);
		};
	}
	else
		return FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM,NULL,code,0,ret,bufsize,NULL);
};

void error(DWORD code)
{
	lv.errorCode = code;
	SetEvent(lv.endIntegration);
};

int busy(void)
{
	return (WaitForSingleObject(lv.runIntegrator,0) == WAIT_OBJECT_0);
};

int fireEvent(char *eventname)
{
	PyGILState_STATE gstate;

	gstate = PyGILState_Ensure();
	PyObject_CallMethod(lv.self,"integratorEvent","s",eventname);
	PyGILState_Release(gstate);
    return 1;
};





// Threads used for integration and processing
void processThread(void *var)
{
	PyGILState_STATE gstate;
	int m;
	tmode *mode;
	DWORD fault,tid;
	char msg[MSGSIZE];

	tid = GetCurrentThreadId();
	gstate = PyGILState_Ensure();
	sprintf_s(msg,MSGSIZE,"processThread started, id: %i",tid);
	PyObject_CallMethod(lv.log,"info","s",msg);
	PyGILState_Release(gstate);

	while(WaitForSingleObject(lv.doneTrigger,0) == WAIT_TIMEOUT)
	{
		//// Signal process done
		//woSetReady(lv.processState);
		//woWaitReady(lv.processState,INFINITE);

		//// Wait until integrationThread signals
		//woWaitBusy(lv.processState,INFINITE);
		SignalObjectAndWait(lv.processReady,lv.doProcess,INFINITE,FALSE);

		// If done is signalled skip to exit without processing
		if (WaitForSingleObject(lv.doneTrigger,0) == WAIT_OBJECT_0)
			goto done;

		// Call mode process routines
		for (m=0;m<lv.modelist->count;m++)
		{
			mode = lv.modelist->items[m];
			fault = mode->process(mode->index,lv.pulseToProcess,lv.samplesToProcess);
			if (fault)
			{
				error(fault);
				break;
			};
		};

	};
done:
	SetEvent(lv.processThreadDone);
	_endthread();
};

void integrationThread(void *var)
{
	PyGILState_STATE gstate;
	int npulse,wait,sync,err,synchronizing,skip,process;
	DWORD rc,fault,tid,*hdr,nspulse;
	char msg[MSGSIZE];
	double secs;

	tid = GetCurrentThreadId();
	gstate = PyGILState_Ensure();
	sprintf_s(msg,MSGSIZE,"integrationThread started, id: %i",tid);
	PyObject_CallMethod(lv.log,"info","s",msg);
	PyGILState_Release(gstate);

	secs = 0.0;
	while(WaitForSingleObject(lv.doneTrigger,0) == WAIT_TIMEOUT)
	{
		if (WaitForSingleObject(lv.runIntegrator,INFINITE) == WAIT_OBJECT_0)
		{
			
			lv.missedPulses = 0;
			skip = 0;
			synchronizing = (lv.synctime > 0.1);
			for (npulse=0;npulse<lv.npulsesint;npulse++)
			{
				process = 1; //defaults to process the samples
				if (WaitForSingleObject(lv.doneTrigger,0) == WAIT_OBJECT_0)
					goto done;

				// End integration early. Error in process thread
				if (WaitForSingleObject(lv.endIntegration,0) == WAIT_OBJECT_0)
				{
					ResetEvent(lv.runIntegrator);
					// stop sampling. 
					readRegisters(MEMREG,rcControl,1,&rc);
					rc &= ~crSampleEnable;
					writeRegisters(MEMREG,rcControl,1,&rc);
					fireEvent("-integrating");
					fault = lv.errorCode;
					skip = 1;
					goto forceEndIntegration;
				};

				// Check to see if integrator has been stopped
				if (WaitForSingleObject(lv.stopIntegrator,0) == WAIT_OBJECT_0)
				{
					process = 0; // stop processing samples just collect them
					if (secs >= lv.synctime)
					{
						lv.synctime = 0.0;
						ResetEvent(lv.runIntegrator);
						// stop sampling. 
						readRegisters(MEMREG,rcControl,1,&rc);
						rc &= ~crSampleEnable;
						writeRegisters(MEMREG,rcControl,1,&rc);
						fireEvent("-integrating");
						fault = 0;
						skip = 1;
						goto forceEndIntegration;
					}
				}
					

				//Collectsamples here
				while (TRUE)
				// dtc synchronization loop
				{
					fault = collectSamples(lv.samplebuffer->size,lv.samplebuffer->inbuf);
					if (fault)
					{
						error(fault);
						goto skipprocessing;
					};

					hdr = (DWORD *)lv.samplebuffer->inbuf;
					if (hdr[0] != 0x00000100)
					{
						error(ecSampleBufferCorrupt);
						goto skipprocessing;
					}

					nspulse = (hdr[11]<<16)+hdr[12];
					secs = (double)(0x0fffffff & hdr[7]) * 4096.0 + (double)((0xfff00000 & hdr[8]) >> 20);

					// Break here if we are not trying to synchronize
					if (!synchronizing)
						break;

					// remaining loop code only runs when synchronizing

					// if we are past the synctime
					sync = (secs >= lv.synctime);

					// only allow synctime to be max 30 secs into the future,
					// so we dont get stuck here forever
					err = (lv.synctime > (secs+30));

					if (sync || err)  
					{
						lv.intcount = 0;
						synchronizing = 0;
						lv.synctime = 0.0;
						fireEvent("+integrating");
						break;
					}
				}
				
				if (process)
				{

					// do processing
					wait = ((npulse == 0) || (npulse == (lv.npulsesint-1)));
					if (wait)
					{
						if (npulse == 0)
						{
							lv.startmsw = hdr[7];
							lv.startlsw = hdr[8];
							lv.starttime = getTime();
						}
						WaitForSingleObject(lv.processReady,INFINITE);
					}

//					if (nspulse == 0) // skip processing if there are no samples
//						continue;

					if (WaitForSingleObject(lv.processReady,0) == WAIT_OBJECT_0)
					{
						sbSwap(lv.samplebuffer);
						lv.pulseToProcess = npulse;
						lv.samplesToProcess = lv.samplebuffer->outbuf;
						//woSetBusy(lv.processState);
						ResetEvent(lv.processReady);
						PulseEvent(lv.doProcess);
					}
					else
						lv.missedPulses ++;
				}
skipprocessing:;

			}; //end for loop

forceEndIntegration:
			// make sure endtime is correctly set to last pulse integrated
			lv.endmsw = hdr[7];
			lv.endlsw = hdr[8];
			lv.endtime = getTime();

			// increment integration count
			lv.intcount++;

			// Wait for last pulse to be processed
			WaitForSingleObject(lv.processReady,INFINITE);

			// Call shell integrationDone function
			translateError(fault,msg,MSGSIZE);
			gstate = PyGILState_Ensure();
			PyObject_CallMethod(lv.shell,"integrationDone","ksiikkkkk",fault,msg,skip,lv.missedPulses,
								lv.intcount,lv.startmsw,lv.startlsw,lv.endmsw,lv.endlsw);
			PyGILState_Release(gstate);		
		}
	}

done:

	// stop sampling
	readRegisters(MEMREG,rcControl,1,&rc);
	rc &= ~crSampleEnable;
	writeRegisters(MEMREG,rcControl,1,&rc);

	// signal thread is done
	SetEvent(lv.integrationThreadDone);
	_endthread();
};

// Python exported routines

static PyObject *int_initialize(PyObject *self, PyObject *args)
{
	HANDLE process,mainthread;
	DWORD priorityclass,tid;
	int mainPriority,integrationPriority,processPriority;
	char msg[MSGSIZE];

	PyObject *conf,*main,*modes,*log;

	if (!PyArg_ParseTuple(args, "OOO",&lv.self,&lv.shell,&lv.log))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: initialize({shell,log})");
		goto error;
	};
	Py_XINCREF(lv.shell);

	Py_XINCREF(lv.log); //Make sure log is not garbage collected upon exit;

	// Create modelist
	lv.modelist = (tmodelist *) malloc(sizeof(tmodelist));
	lv.modelist->count = 0;

	InitializeCriticalSection(&lv.lock);
	lv.processReady = CreateEvent(NULL,TRUE,FALSE,NULL);
	lv.doProcess = CreateEvent(NULL,TRUE,FALSE,NULL);
	lv.stopIntegrator = CreateEvent(NULL,TRUE,FALSE,NULL);
	lv.runIntegrator = CreateEvent(NULL,TRUE,FALSE,NULL);
	lv.doneTrigger = CreateEvent(NULL,TRUE,FALSE,NULL);

	lv.integrationThreadDone = CreateEvent(NULL,TRUE,FALSE,NULL);
	lv.processThreadDone = CreateEvent(NULL,TRUE,FALSE,NULL);
	lv.endIntegration = CreateEvent(NULL,TRUE,FALSE,NULL);

	// Get handle to current process
	process = GetCurrentProcess();
	SetProcessAffinityMask(process,1);
	mainthread = GetCurrentThread();
	// Boost to realtime priority class
	SetPriorityClass(process,REALTIME_PRIORITY_CLASS);

	SetThreadPriority(mainthread,THREAD_PRIORITY_IDLE);
	mainPriority = GetThreadPriority(mainthread);

	PyObject_CallMethod(lv.log,"info","s","going to start integrationThread");
	lv.integrationThread = (HANDLE)_beginthread(integrationThread,0,NULL);
	SetThreadPriority(lv.integrationThread,THREAD_PRIORITY_ABOVE_NORMAL);
//	SetThreadPriority(lv.integrationThread,THREAD_PRIORITY_IDLE);
	integrationPriority = GetThreadPriority(lv.integrationThread);

	PyObject_CallMethod(lv.log,"info","s","going to start processThread");
	lv.processThread = (HANDLE)_beginthread(processThread,0,NULL);
	SetThreadPriority(lv.processThread,THREAD_PRIORITY_TIME_CRITICAL);
	processPriority = GetThreadPriority(lv.processThread);


	sprintf_s(msg,MSGSIZE,"Process thread priority: %i",processPriority);
	PyObject_CallMethod(lv.log,"info","s",msg);

	sprintf_s(msg,MSGSIZE,"Integration thread priority: %i",integrationPriority);
	PyObject_CallMethod(lv.log,"info","s",msg);

	sprintf_s(msg,MSGSIZE,"Main thread priority: %i",mainPriority);
	PyObject_CallMethod(lv.log,"info","s",msg);
	
cleanup:
	Py_RETURN_NONE;
error:
	return NULL;
};

static PyObject *int_configure(PyObject *self, PyObject *args)
{
	PyObject *conf,*main,*log,*modes,*par;
	PyObject *dllpath,*index;
	int item,len,sbsize;
	char msg[MSGSIZE];
	tmode *mode;
	DWORD nradacheaderwords;
	int ires;

	if (!PyArg_ParseTuple(args, "O",&conf))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure({config dict})");
		goto error;
	};

	if (!PyDict_Check(conf))
	{
		PyErr_SetString(PyExc_TypeError,"configure needs a python dict holding setup info as argument 1");
		goto error;
	};

	lv.conf = conf;
	Py_XINCREF(lv.conf); //Make sure conf is not garbage collected upon exit;

	// Reset endIntegration event
	ResetEvent(lv.endIntegration);

	// Remove possible 'old' modes
	listClear(lv.modelist);

	// Load new modes
	modes = PyDict_GetItemString(lv.conf,"modes");
	len = PyList_Size(modes);
	sprintf_s(msg,MSGSIZE,"modes list length: %i",len);
	PyObject_CallMethod(lv.log,"info","s",msg);
	for (item=0;item<len;item++)
	{
		// Load modes
		mode = listNewItem(lv.modelist);
		mode->mode = PyList_GetItem(modes,item);
		dllpath = PyObject_GetAttrString(mode->mode,"dllpath");
		mode->lib = LoadLibrary(PyString_AsString(dllpath));
		mode->process = (moprocessfunc)GetProcAddress(mode->lib,"process");
		if (mode->process == NULL)
		{
			sprintf_s(msg,MSGSIZE,"Could not load library: %s",PyString_AsString(dllpath));
			PyErr_SetString(PyExc_ImportError,msg);
			goto error;
		};

		// Parse mode attributes
		index = PyObject_GetAttrString(mode->mode,"index");
		mode->index = PyInt_AsLong(index);

		sprintf_s(msg,MSGSIZE,"Loaded mode: %s. Index: %i",PyString_AsString(dllpath),mode->index);
		PyObject_CallMethod(lv.log,"info","s",msg);
	};

	//npulsesint
	if (!getInt(lv.conf,"npulsesint",&lv.npulsesint))
	{
		PyErr_SetString(PyExc_KeyError,"npulsesint not in parameters dict");
		goto error;
	};
	sprintf_s(msg,MSGSIZE,"npulsesint set to: %i",lv.npulsesint);
	PyObject_CallMethod(lv.log,"info","s",msg);

	//maxsamples
	par = PyDict_GetItemString(lv.conf,"maxsamples");
	if (!getInt(lv.conf,"maxsamples",&ires))
	{
		PyErr_SetString(PyExc_KeyError,"maxsamples not in parameters dict");
		goto error;
	};
	readRegisters(MEMREG,rcNHeaderWords,1,&nradacheaderwords);
	sbsize = ires+nradacheaderwords+10; //needs to be bigger than maxsamples+nradacheaderwords
	lv.samplebuffer = (dworddoublebuffer *)malloc(sizeof(dworddoublebuffer));
	sbReset(lv.samplebuffer);
	sbSetSize(lv.samplebuffer,sbsize);
	sprintf_s(msg,MSGSIZE,"samplebuffer size set to: %i",lv.samplebuffer->size);
	PyObject_CallMethod(lv.log,"info","s",msg);

	// reset integration count
	lv.intcount = 0;

	PyObject_CallMethod(lv.log,"debug","s","Integrator configured");
cleanup:
	Py_RETURN_NONE;
error:
	return NULL;
};


static PyObject *int_shutdown(PyObject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ""))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: shutdown()");
		goto error;
	};

	// Kill threads
	PyObject_CallMethod(lv.log,"info","s","Setting doneTrigger to stop threads");
	SetEvent(lv.doneTrigger);

	Py_BEGIN_ALLOW_THREADS;
	// Wait for threads to complete
	while (WaitForSingleObject(lv.processThreadDone,100) == WAIT_TIMEOUT)
		SetEvent(lv.doProcess);
	while (WaitForSingleObject(lv.integrationThreadDone,100) == WAIT_TIMEOUT)
		SetEvent(lv.runIntegrator);
	Py_END_ALLOW_THREADS;
	PyObject_CallMethod(lv.log,"info","s","Threads terminated");

	// Close all open handles
	CloseHandle(lv.doneTrigger);
	CloseHandle(lv.processThreadDone);
	CloseHandle(lv.integrationThreadDone);
	CloseHandle(lv.processReady);
	CloseHandle(lv.doProcess);
	CloseHandle(lv.stopIntegrator);
	CloseHandle(lv.runIntegrator);
	//CloseHandle(lv.integrationState.busy);
	//CloseHandle(lv.integrationState.ready);
	//CloseHandle(lv.processState.busy);
	//CloseHandle(lv.processState.ready);

	//Clear modelist
	listClear(lv.modelist);


	// Release various python objects
	Py_XDECREF(lv.shell);
	Py_XDECREF(lv.log);
	Py_XDECREF(lv.conf);

cleanup:
	Py_RETURN_NONE;
error:
	return NULL;
};


static PyObject *int_start(PyObject *self, PyObject *args)
{
	char errMsg[MSGSIZE];
	DWORD rc;

	double timetag;

	if (!PyArg_ParseTuple(args, "d",&timetag))
		timetag = 0;
	lv.synctime = timetag;

	if (WaitForSingleObject(lv.endIntegration,0) == WAIT_OBJECT_0)
	{
		// Error condition throw an exception instead of
		// starting new integration
		translateError(lv.errorCode,errMsg,MSGSIZE);
		PyErr_SetString(PyExc_RuntimeError,errMsg);
		return NULL;
	};

	// start sampling
	readRegisters(MEMREG,rcControl,1,&rc);
	rc |= crSampleEnable;
	writeRegisters(MEMREG,rcControl,1,&rc);

	ResetEvent(lv.stopIntegrator);
	SetEvent(lv.runIntegrator);
	Py_RETURN_NONE;
};

static PyObject *int_stop(PyObject *self, PyObject *args)
{
	char errMsg[MSGSIZE];
	DWORD rc;
	double timetag;

	if (!PyArg_ParseTuple(args, "d",&timetag))
		timetag = 0;
	lv.synctime = timetag;

	SetEvent(lv.stopIntegrator);

	Py_RETURN_NONE;
};

static PyObject *int_busy(PyObject *self, PyObject *args)
{
	return Py_BuildValue("i",busy());
};

static PyObject *int_stringError(PyObject *self, PyObject *args)
{
	DWORD code;
	char errmsg[MSGSIZE];

	if (!PyArg_ParseTuple(args, "I",&code))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: string = stringError(errorCode)");
		goto error;
	};

	translateError(code,errmsg,MSGSIZE);

cleanup:
	return Py_BuildValue("s",errmsg);
error:
	return NULL;
};


// Python Initialization code

static PyMethodDef intMethods[] = 
{
	{"initialize", int_initialize, METH_VARARGS, "initialize(log,done callback)"},
	{"configure", int_configure, METH_VARARGS, "configure({config dict}). Configures the integrator"},
	{"shutdown", int_shutdown, METH_VARARGS, "shutdown(). Shuts the integrator/processor down"},
	{"start", int_start, METH_VARARGS, "start(). Starts the integration"},
	{"stop", int_stop, METH_VARARGS, "stop(). Stops the integration"},
	{"busy", int_busy, METH_VARARGS, "boolean = busy(). Query integrator state"},
	{"stringError", int_stringError, METH_VARARGS, "string = stringError(errorCode). Translate error code to message"},
	{NULL, NULL, 0, NULL} /* Sentinel */
};


PyMODINIT_FUNC
initextIntegrator(void)
{
	PyObject *m;

	m = Py_InitModule("extIntegrator", intMethods);
	if (m == NULL)
		return;
	import_array();
};

