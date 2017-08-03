// Global definitions and includes
#include "common.h"

// Mode specific includes
#include "timepwr.h"

// Common mode helper includes
#include "modeCommon.h"

void logmessage(localvars *vars,char *msg)
{
	PyGILState_STATE gstate;

	gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects
	PyObject_CallMethod(vars->log,"info","s",msg);
	PyGILState_Release(gstate);
}


PyObject *getIndices(PyObject *indices,int pulse,int freq)
{
	PyObject *lpulse,*lfreq;

	lpulse = PyList_GetItem(indices,pulse);
	lfreq = PyList_GetItem(lpulse,freq);
	return lfreq;
}

int getIndex(PyObject *indices,int index,int offset)
{
	PyObject *tuple,*val;
	int v;

	tuple = PyList_GetItem(indices,index);
	val = PyTuple_GetItem(tuple,offset);
	v = PyInt_AsLong(val);
	return v;
}

float sampleAverage(float *inbuf,int si, int ei)
{
	int i,l;
	float sum;

	l = ei-si+1;
	sum = 0.0;
	for (i=si;i<=ei;i++)
		sum += inbuf[i];
	sum /= l;
	return sum;
}


// dll exported process routine
DWORD process(int inst,int npulse, void *inbuf)
{
	localvars *vars;
	DWORD *hdr,*sd;
	int bp,bcRet;
	char path[MAXPATH],msg[MSGSIZE];
	PyGILState_STATE gstate;
	int i,p,c,np,nsamples;
    DWORD *dwp;
    float *fp,*ppwr,*prat;
	int *ppulsesint;
	double *dp;
	float sum,avg;
	int cnt,si,ei;
	PyObject *indices,*pulsesintegrated,*beamcodes;
	int *pbc,*ppi;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		vars->pulse = 0;
		bcReset(vars->beamcodes);
	};

	hdr = (DWORD *)inbuf;
	if (hdr[0] != 0x00000100) // header sanity check
		return ecSamplesMisalignment;
	sd = hdr+vars->nradacheaderwords+vars->indexsample;

	if (hdr[10] == vars->modegroup) // modegroup test
	{
		// Check to see if beamcode was configured
		bcRet = bcIndex(vars->beamcodes,hdr[4],&bp);
		if (bcRet < 0) // beamcode not in list
			return ecBeamcodeNotInList;
		if (vars->pulse >= vars->ntotalpulses)
			return ecIndexOutOfBounds;

		// number of samples in ipp
		nsamples = (hdr[11]<<16)+hdr[12];

		// calculate and integrate pwr on various frequencies
		fp = (float *)PyArray_DATA(vars->samplebuffer);
		ppwr = (float *)PyArray_DATA(vars->power);
		ppulsesint = (int *)PyArray_DATA(vars->pulsesintegrated);
		cvrealmags(sd,fp,1,nsamples);
		np = hdr[3];
		for (p=1;p<=3;p++)
		{
			indices = getIndices(vars->indices,np,p);
			sum = 0.0;
			cnt = 0;
			for (i=0;i<PyList_Size(indices);i++)
			{
				si = getIndex(indices,i,0)+vars->indexadjust;
				ei = getIndex(indices,i,1)-vars->indexadjust;
				avg = sampleAverage(fp,si,ei);
				sum += avg;
				cnt ++;
			}
			if (cnt > 0)
			{
				sum /= cnt;
				ppwr[p] += sum;
				ppulsesint[p] += cnt;
			}
		}

		vars->pulse++;
	};

	if (npulse == vars->npulsesint-1)
	{
		// Scale power with number of pulses integrated
		for (p=0;p<=3;p++)
			if (ppulsesint[p] > 0)
				ppwr[p] /= ppulsesint[p];

		// Calculate ratios
		prat = PyArray_DATA(vars->ratios);
		sum = 0.0;
		for (p=0;p<=3;p++)
			sum += ppwr[p];
		if (sum == 0.0)
			sum = 1.0;

		prat[0] = ppwr[0]/sum;
		prat[1] = ppwr[1]/sum;
		prat[2] = ppwr[2]/sum;
		prat[3] = ppwr[3]/sum;
		

		// save arrays
		// Beamcodes and pulsesintegrated
		beamcodes = createArray(PyArray_INT32,1,vars->beamcodes->npos);
		pbc = PyArray_DATA(beamcodes);
		pulsesintegrated = createArray(PyArray_INT32,1,vars->beamcodes->npos);
		ppi = PyArray_DATA(pulsesintegrated);
		for (bp=0;bp<vars->beamcodes->npos;bp++)
		{
			pbc[bp] = vars->beamcodes->codes[bp];
			ppi[bp] = vars->beamcodes->count[bp];
		}

		gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects

		h5Dynamic(vars->self,vars->root,"Beamcodes",beamcodes);
		h5Dynamic(vars->self,vars->root,"PulsesIntegrated",pulsesintegrated);

		// power
		h5Dynamic(vars->self,vars->root,"Power/Data",vars->power);

		// samples
		h5Dynamic(vars->self,vars->root,"Ratios/Data",vars->ratios);

		// pulses integrated / frequency
		h5Dynamic(vars->self,vars->root,"PulsesIntegratedPerFrequency",vars->pulsesintegrated);

		PyGILState_Release(gstate);

		// release arrays
		Py_XDECREF(beamcodes);
		Py_XDECREF(pulsesintegrated);

		// dec ref count
		Py_XDECREF(vars->power);
		Py_XDECREF(vars->ratios);
		Py_XDECREF(vars->pulsesintegrated);

		// Now recreate arrays so they are ready for next set of pulses
		vars->power = createArrayZeros(PyArray_FLOAT,1,4);
		vars->ratios = createArrayZeros(PyArray_FLOAT,1,4);
		vars->pulsesintegrated = createArrayZeros(PyArray_INT32,1,4);


	};

	return ecOk;
};

// Python exported routines

static PyObject *ext_configure(PyObject *self, PyObject *args)
{
	PyObject *parent;
	int i,ires;
	char msg[MSGSIZE];
	char path[MAXPATH];
	double dres;
	localvars *vars;

	if (!PyArg_ParseTuple(args, "O",&parent))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure(self), self=mode class");
		goto error;
	};

	vars = modeInit(parent);
	if (vars == NULL)
		goto error;

	// rf indecies
	vars->indices = PyDict_GetItemString(vars->pars,"rfindices");
	Py_XINCREF(vars->indices);


	//filter time delay
	if (!getDouble(vars->pars,"filtertimedelay",&vars->filtertimedelay))
	{
		vars->filtertimedelay = 0.0;
		sprintf_s(msg,MSGSIZE,"filtertimedelay not in parameters dict, defaults to 0.0");
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	sprintf_s(msg,MSGSIZE,"filtertimedelay set to: %g",vars->filtertimedelay);
	PyObject_CallMethod(vars->log,"info","s",msg);

	//offsetsamples
	if (!getInt(vars->pars,"offsetsamples",&vars->offsetsamples))
	{
		vars->offsetsamples = 0;
		sprintf_s(msg,MSGSIZE,"offsetsamples not in parameters, defaults to: %i",vars->offsetsamples);
		PyObject_CallMethod(vars->log,"info","s",msg);
	};
	sprintf_s(msg,MSGSIZE,"offsetsamples set to: %i",vars->offsetsamples);
	PyObject_CallMethod(vars->log,"info","s",msg);

	//maxsamples
	if (!getInt(vars->pars,"maxsamples",&vars->maxsamples))
	{
		PyErr_SetString(PyExc_KeyError,"maxsamples not in parameters dict");
		goto error;
	};

	vars->indexadjust = (int) ceil(vars->filtertimedelay/vars->sampletime)+vars->offsetsamples;
	sprintf_s(msg,MSGSIZE,"indexadjust set to: %i",vars->indexadjust);
	PyObject_CallMethod(vars->log,"info","s",msg);

	vars->power = createArrayZeros(PyArray_FLOAT,1,4);
	vars->ratios = createArrayZeros(PyArray_FLOAT,1,4);
	vars->pulsesintegrated = createArrayZeros(PyArray_INT32,1,4);
	vars->samplebuffer = createArrayZeros(PyArray_FLOAT,1,vars->maxsamples); //Make sure samplebuffer can hold max number of samples


	PyObject_CallMethod(vars->log,"info","s","Configuration done");


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

	bcFree(vars->beamcodes);
	Py_XDECREF(vars->indices);
	Py_XDECREF(vars->power);
	Py_XDECREF(vars->ratios);
	Py_XDECREF(vars->pulsesintegrated);
	Py_XDECREF(vars->samplebuffer);

	sprintf_s(msg,MSGSIZE,"Mode instance %i has been shut down",instance);
	PyObject_CallMethod(vars->log,"info","s",msg);

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

    

// Python Initialization code

static PyMethodDef extMethods[] = 
{
	{"configure", ext_configure, METH_VARARGS, "configure({config dict}). Configures the mode"},
	{"shutdown", ext_shutdown, METH_VARARGS, "shutdown(instance). Shuts down the mode instance"},
	{NULL, NULL, 0, NULL} /* Sentinel */
};


PyMODINIT_FUNC
inittimepwr(void)
{
	PyObject *m;

	m = Py_InitModule("timepwr", extMethods);
	if (m == NULL)
		return;
	import_array();
};

