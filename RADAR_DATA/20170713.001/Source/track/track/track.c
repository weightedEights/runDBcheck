// Global definitions and includes
#include "common.h"

// Mode specific includes
#include "track.h"

// Common mode helper includes
#include "modeCommon.h"
#include "radac.h"


void logMessage(PyObject *log, char *msg)
{
	PyGILState_STATE gstate;

	gstate = PyGILState_Ensure();
	PyObject_CallMethod(log,"info","s",msg);
	PyGILState_Release(gstate);
}



// dll exported process routine
DWORD process(int inst,int npulse, void *inbuf)
{
	localvars *vars;
	DWORD *hdr,*sd;
	DWORD dw;
	int bp,bcRet;
	char path[MAXPATH],msg[MSGSIZE];
	PyGILState_STATE gstate;
	DWORD usec,lsec,msec,sec;
	int c;
    DWORD *dwp;
    float *fp;
	double *dp;
	double rt;
	TRACK *track;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		vars->pulse = 0;
		//bcReset(vars->beamcodes);
	};

	hdr = (DWORD *)inbuf;
	if (hdr[0] != 0x00000100) // header sanity check
		return ecSamplesMisalignment;
	sd = hdr+vars->nradacheaderwords+vars->indexsample;

	// extract radactime
	usec = hdr[8] & 0x000fffff;
	lsec = ((hdr[8] >> 20) & 0x00000fff);
	msec = ((hdr[7] << 12) & 0xfffff000);
	sec = msec | lsec;
	rt = (float)sec + (float)usec/1e6;

	if (hdr[10] == vars->modegroup) // modegroup test
	{
		if (vars->pulse >= vars->ntotalpulses)
		{
			return ecIndexOutOfBounds;
		}

		// fill header arrays
		dwp = (DWORD *)PyArray_DATA(vars->framecount);
		dwp[vars->pulse] = hdr[2];
		dwp = (DWORD *)PyArray_DATA(vars->pulsecount);
		dwp[vars->pulse] = hdr[3];
		dwp = (DWORD *)PyArray_DATA(vars->beamcode);
		dwp[vars->pulse] = hdr[4];
		dwp = (DWORD *)PyArray_DATA(vars->timecount);
		dwp[vars->pulse] = hdr[5];
		dwp = (DWORD *)PyArray_DATA(vars->timestatus);
		dwp[vars->pulse] = hdr[6];
		dp = (double *)PyArray_DATA(vars->radactime);
		dp[vars->pulse] = rt;
		dwp = (DWORD *)PyArray_DATA(vars->group);
		dwp[vars->pulse] = hdr[10];
		dwp = (DWORD *)PyArray_DATA(vars->nsamplespulse);
		dwp[vars->pulse] = (hdr[11]<<16)+hdr[12];
		dwp = (DWORD *)PyArray_DATA(vars->code);
		for (c=0;c<NCODES;c++)
			dwp[vars->pulse*NCODES+c] = hdr[13+c];

		// make samples floats
		fp = (float *)PyArray_DATA(vars->samples)+vars->pulse*vars->ngates*COMPLEX;
		cvfloats(sd,fp,vars->ngates);

		vars->pulse++;
	}
	// Check for current track info
	if (vars->instance == 0) //only 1 instance needs to modify hardware
	{
		track = tbGetTrack(vars->tracks,rt);
		if (track != NULL)
		{
			switch(track->state)
			{
				case tInit:
					// program beamcodes and ids
					writeRegisters(MEMBC2,0,track->nobjects,track->data);
					dw = 0x80000003;
					writeRegisters(MEMREG,rcIppMask,1,&dw);		// turn on tracking ipps
					track->state = tRunning;
					break;
				case tDone:
					// turn off tracking pulses
					dw = 0x80000001;
					writeRegisters(MEMREG,rcIppMask,1,&dw); //set ippmask to 1 to turn off tracking pulses
					track->state = tFinished;
					break;
				default:
					break;
			}
		}

	}

	if (npulse == vars->npulsesint-1)
	{
		// save arrays
		// Beamcodes and pulsesintegrated
		//beamcodes = createArray(PyArray_INT32,1,vars->beamcodes->npos);
		//pbc = PyArray_DATA(beamcodes);
		//pulsesintegrated = createArray(PyArray_INT32,1,vars->beamcodes->npos);
		//ppi = PyArray_DATA(pulsesintegrated);
		//for (bp=0;bp<vars->beamcodes->npos;bp++)
		//{
		//	pbc[bp] = vars->beamcodes->codes[bp];
		//	ppi[bp] = vars->beamcodes->count[bp];
		//}

		gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects

		//h5Dynamic(vars->self,vars->root,"Beamcodes",beamcodes);
		//h5Dynamic(vars->self,vars->root,"PulsesIntegrated",pulsesintegrated);

		//header
		h5Dynamic(vars->self,vars->root,"RadacHeader/FrameCount",vars->framecount);
		h5Dynamic(vars->self,vars->root,"RadacHeader/PulseCount",vars->pulsecount);
		h5Dynamic(vars->self,vars->root,"RadacHeader/BeamCode",vars->beamcode);
		h5Dynamic(vars->self,vars->root,"RadacHeader/TimeCount",vars->timecount);
		h5Dynamic(vars->self,vars->root,"RadacHeader/TimeStatus",vars->timestatus);
		h5Dynamic(vars->self,vars->root,"RadacHeader/RadacTime",vars->radactime);
		h5Dynamic(vars->self,vars->root,"RadacHeader/ModeGroup",vars->group);
		h5Dynamic(vars->self,vars->root,"RadacHeader/NSamplesPulse",vars->nsamplespulse);
		h5Dynamic(vars->self,vars->root,"RadacHeader/Code",vars->code);

		// samples
		h5Dynamic(vars->self,vars->root,"Samples/Data",vars->samples);
		PyGILState_Release(gstate);

		// release arrays
		//Py_XDECREF(beamcodes);
		//Py_XDECREF(pulsesintegrated);
		//header
		Py_XDECREF(vars->framecount);
		Py_XDECREF(vars->pulsecount);
		Py_XDECREF(vars->beamcode);
		Py_XDECREF(vars->timecount);
		Py_XDECREF(vars->timestatus);
		Py_XDECREF(vars->radactime);
		Py_XDECREF(vars->group);
		Py_XDECREF(vars->nsamplespulse);
		Py_XDECREF(vars->code);

		// samples
		Py_XDECREF(vars->samples);


		// Now recreate arrays so they are ready for next set of pulses
		vars->framecount = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
		vars->pulsecount = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
		vars->beamcode = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
		vars->timecount = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
		vars->timestatus = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
		vars->radactime = createArrayZeros(PyArray_DOUBLE,1,vars->ntotalpulses);
		vars->group = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
		vars->nsamplespulse = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
		vars->code = createArrayZeros(PyArray_ULONG,2,vars->ntotalpulses,NCODES);

//		vars->power = createArrayZeros(PyArray_FLOAT,2,vars->beamcodes->npos,vars->ngates);
		vars->samples = createArrayZeros(PyArray_FLOAT,3,vars->ntotalpulses,vars->ngates,COMPLEX);
		
	};

	return ecOk;
};



// Python exported routines

static PyObject *ext_configure(PyObject *self, PyObject *args)
{
	PyObject *parent;
	PyObject *rangearray;
	int i;
	char msg[MSGSIZE];
	char path[MAXPATH];
	double fr,dr;
	float *range;
	localvars *vars;
	int npos;

	if (!PyArg_ParseTuple(args, "O",&parent))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure(self), self=mode class");
		goto error;
	};

	vars = modeInit(parent);

	if (vars == NULL)
		goto error;

	if (!InitializeCriticalSectionAndSpinCount(&vars->lock,2) ) //lock to protect vars
		goto error;

	npos = vars->beamcodes->npos;

	// initialize tracking variables
	vars->tracks = tbCreate();
	vars->currenttrack = -2;

    //calculate and save range array
	fr = vars->firstrange-
	     vars->filterrangedelay+
		 vars->rangecorrection;
	sprintf_s(msg,MSGSIZE,"Firstrange: %f",fr);
	PyObject_CallMethod(vars->log,"info","s",msg);

	rangearray = createArray(PyArray_FLOAT,2,1,vars->ngates);
	range = (float *) PyArray_DATA(rangearray);
	for (i=0;i<vars->ngates;i++)
		range[i] = fr+(float)i*vars->samplespacing;
	h5Static(vars->self,vars->root,"Samples/Range",rangearray);
	h5Attribute(vars->self,vars->root,"Samples/Range/Unit",Py_BuildValue("s","m"));

	// Various attributes
	h5Attribute(vars->self,vars->root,"Samples/Data/Unit",Py_BuildValue("s","Samples"));

	// Create arrays so they are ready for first integration
	vars->framecount = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
	vars->pulsecount = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
	vars->beamcode = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
	vars->timecount = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
	vars->timestatus = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
	vars->radactime = createArrayZeros(PyArray_DOUBLE,1,vars->ntotalpulses);
	vars->group = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
	vars->nsamplespulse = createArrayZeros(PyArray_ULONG,1,vars->ntotalpulses);
	vars->code = createArrayZeros(PyArray_ULONG,2,vars->ntotalpulses,NCODES);

	vars->samples = createArrayZeros(PyArray_FLOAT,3,vars->ntotalpulses,vars->ngates,COMPLEX);

	PyObject_CallMethod(vars->log,"info","s","Configuration done. Full access, modified ref count");


cleanup:
	sprintf_s(msg,MSGSIZE,"Instance: %d",vars->instance);
	PyObject_CallMethod(vars->log,"info","s",msg);
	Py_RETURN_NONE;
error:
	return NULL;
};


static PyObject *ext_shutdown(PyObject *self, PyObject *args)
{
	PyObject *inst;
	int instance;
	int allNull,i;
	localvars *vars;
	char msg[MSGSIZE];

	if (!PyArg_ParseTuple(args, "i",&instance))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: shutdown(instance)");
		goto error;
	};
	vars = lv[instance];
	DeleteCriticalSection(&vars->lock);

	bcFree(vars->beamcodes);

	sprintf_s(msg,MSGSIZE,"Mode instance %i has been shut down",instance);
	PyObject_CallMethod(vars->log,"info","s",msg);

	// Release arrays
	Py_XDECREF(vars->framecount);
	Py_XDECREF(vars->pulsecount);
	Py_XDECREF(vars->beamcode);
	Py_XDECREF(vars->timecount);
	Py_XDECREF(vars->timestatus);
	Py_XDECREF(vars->radactime);
	Py_XDECREF(vars->group);
	Py_XDECREF(vars->nsamplespulse);
	Py_XDECREF(vars->code);
	Py_XDECREF(vars->samples);

	tbFree(vars->tracks);

	free(vars);
	lv[instance] = NULL;
	allNull = 1;
	for (i=0;i<lvCount;i++)
	{
		if (lv[i] != NULL)
		{
			allNull = 0;
			break;
		};
	};
	if (allNull)  //Remove list when last instance is shutdown
	{
//		free(lv);
		lv = NULL;
		lvCount = 0;
	};
			
cleanup:
	Py_RETURN_NONE;
error:
	return NULL;
};

static PyObject *ext_setTrackInfo(PyObject *self, PyObject *args)
{
	char msg[MSGSIZE];
	int instance,res;
	localvars *vars;
	PyObject *track;
	PyGILState_STATE gstate;

	if (!PyArg_ParseTuple(args, "iO",&instance,&track))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: setTrackInfo(instance,info)");
		return NULL;
	};
	vars = lv[instance];

	if (instance == 0)
	{
		gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects
		res = tbAppend(vars->tracks,track);

		if (res)
			sprintf_s(msg,MSGSIZE,"mode(%i) track: tracking information updated",instance);
		else
			sprintf_s(msg,MSGSIZE,"mode(%i) track: not all tracks were successfully added",instance);
		PyObject_CallMethod(vars->log,"info","s",msg);
		PyGILState_Release(gstate);
	}

	Py_RETURN_NONE;
};

    

// Python Initialization code

static PyMethodDef extMethods[] = 
{
	{"configure", ext_configure, METH_VARARGS, "configure({config dict}). Configures the mode"},
	{"shutdown", ext_shutdown, METH_VARARGS, "shutdown(instance). Shuts down the mode instance"},
	{"setTrackInfo", ext_setTrackInfo, METH_VARARGS, "setTrackInfo(info). Load track info"},
	{NULL, NULL, 0, NULL} /* Sentinel */
};


PyMODINIT_FUNC
inittrack(void)
{
	PyObject *m;

	m = Py_InitModule("track", extMethods);
	if (m == NULL)
		return;
	import_array();
};

