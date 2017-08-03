// Global definitions and includes
#include "common.h"

// Mode specific includes
#include "spectralpwr.h"
#include "four1.h"

// Common mode helper includes
#include "modeCommon.h"


// dll exported process routine
DWORD process(int inst,int npulse, void *inbuf)
{
	localvars *vars;
	DWORD *hdr,*sd;
	int bp,bcRet,s0,s1;
	char path[MAXPATH],msg[MSGSIZE];
	PyGILState_STATE gstate;
	DWORD usec,lsec,msec,sec;
	int ilagno,igate;
    float *pgate,*pgates,*pspectra,*ppower;
	Complex s;
	PyObject *pulsesintegrated,*beamcodes;
	int *pbc,*ppi;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		bcReset(vars->beamcodes);
		floatZeroArray(vars->gates);
		vars->power = createArrayZeros(PyArray_FLOAT,2,vars->beamcodes->npos,vars->npowergates);
		vars->spectra = createArrayZeros(PyArray_FLOAT,3,vars->beamcodes->npos,vars->nlags,vars->ngates);
	};

	hdr = (DWORD *)inbuf;
	if (hdr[0] != 0x00000100) // header sanity check
		return ecSamplesMisalignment;
	sd = hdr+vars->nradacheaderwords+vars->indexsample;

	// setup strides and pointers
	s0 = vars->nlags*vars->ngates;
	s1 = vars->ngates;
	ppower = (float *)PyArray_DATA(vars->power);
	pgate = (float *)PyArray_DATA(vars->gate);
	pgates = (float *)PyArray_DATA(vars->gates);

	if (hdr[10] == vars->modegroup) // modegroup test
	{
		// Check to see if beamcode was configured
		bcRet = bcIndex(vars->beamcodes,hdr[4],&bp);
		if (bcRet < 0) // beamcode not in list
			return ecBeamcodeNotInList;

		// calculate and integrate power
		cvrealmagsadd(sd,ppower+bp*vars->npowergates,1,vars->npowergates);

		// Calculate spectral array
		for (igate=0;igate<vars->ngates;igate++) 
		{
			for (ilagno=0;ilagno<vars->nlags;ilagno++)
			{
				s.c = *(sd+igate*vars->gatestep+ilagno);
				pgate[2*ilagno] = (float) s.s.r;
				pgate[2*ilagno+1] = (float) s.s.i;
			}
			four1(pgate-1,vars->nlags,vars->fftsign);
			for (ilagno=0;ilagno<vars->nlags;ilagno++) 
				pgate[2*ilagno] = pgate[2*ilagno]*pgate[2*ilagno]+
								  pgate[2*ilagno+1]*pgate[2*ilagno+1]; //square all the real values
			
			for (ilagno=0;ilagno<vars->nlags;ilagno++)
				pgates[bp*s0+ilagno*s1+igate] += pgate[2*ilagno]; //integrate all real values into gates
		}
	};

	if (npulse == vars->npulsesint-1)
	{
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

		pspectra = (float *)PyArray_DATA(vars->spectra);
		for (bp=0;bp<vars->beamcodes->npos;bp++)
		{			
			// Rearrange data from FFT
			for (igate=0;igate<vars->ngates;igate++) 
			{
				for (ilagno=0;ilagno<vars->nlags/2;ilagno++) 
				{
					pspectra[bp*s0+ilagno*s1+igate] = pgates[bp*s0+(ilagno+vars->nlags/2)*s1+igate];
					pspectra[bp*s0+(ilagno+vars->nlags/2)*s1+igate] = pgates[bp*s0+ilagno*s1+igate];
				}
			}
		}

		// save arrays
		gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects
		h5Dynamic(vars->self,vars->root,"Beamcodes",beamcodes);
		h5Dynamic(vars->self,vars->root,"PulsesIntegrated",pulsesintegrated);

		h5Dynamic(vars->self,vars->root,"Power/Data",vars->power);
		h5Dynamic(vars->self,vars->root,"Spectra/Data",vars->spectra);

		PyGILState_Release(gstate);

		// release arrays
		Py_XDECREF(beamcodes);
		Py_XDECREF(pulsesintegrated);
		Py_XDECREF(vars->power);
		Py_XDECREF(vars->spectra);
	};

	return ecOk;
};



// Python exported routines

static PyObject *ext_configure(PyObject *self, PyObject *args)
{
	PyObject *parent;
	PyObject *powerrangearray,*acfrangearray,*freqarray;
	int A,i;
	char msg[MSGSIZE];
	char path[MAXPATH];
	double frPwr,frAcf;
	float *range,*freq;
	double fs,fb,f,drangeacf;
	localvars *vars;

	if (!PyArg_ParseTuple(args, "O",&parent))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure(self), self=mode class");
		goto error;
	};

	vars = modeInit(parent);
	if (vars == NULL)
		goto error;

	//Read mode specific parameters

	//nlags
	if (!getInt(vars->pars,"nlags",&vars->nlags))
	{
		PyErr_SetString(PyExc_KeyError,"nlags not in parameters dict");
		goto error;
	};
	sprintf_s(msg,MSGSIZE,"nlags set to: %i",vars->nlags);
	PyObject_CallMethod(vars->log,"info","s",msg);

	//npowergates
	if (!getInt(vars->pars,"npowergates",&vars->npowergates))
	{
		vars->npowergates = vars->ngates;
		sprintf_s(msg,MSGSIZE,"npowergates not in parameters dict, defaults to ngates");
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	sprintf_s(msg,MSGSIZE,"npowergates set to: %i",vars->npowergates);
	PyObject_CallMethod(vars->log,"info","s",msg);

	//gatestep
	if (!getInt(vars->pars,"gatestep",&vars->gatestep))
	{
		vars->gatestep = vars->nlags;
		sprintf_s(msg,MSGSIZE,"gatestep not in parameters dict, defaults to nlags");
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	sprintf_s(msg,MSGSIZE,"gatestep set to: %i",vars->gatestep);
	PyObject_CallMethod(vars->log,"info","s",msg);

	//fftsign
	if (!getInt(vars->pars,"fftsign",&vars->fftsign))
	{
		vars->fftsign = 1;
		sprintf_s(msg,MSGSIZE,"fftsign not in parameters dict, defaults to 1");
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	sprintf_s(msg,MSGSIZE,"fftsign set to: %i",vars->fftsign);
	PyObject_CallMethod(vars->log,"info","s",msg);


    //calculate and save range arrays
	A = (vars->nlags-1)/2;
	frPwr = vars->firstrange-
			vars->filterrangedelay+
			vars->rangecorrection+
			vars->samplespacing*A;
	sprintf_s(msg,MSGSIZE,"Firstrange power: %f",frPwr);
	PyObject_CallMethod(vars->log,"info","s",msg);

	frAcf = vars->firstrange-
			vars->filterrangedelay+
			vars->rangecorrection+
			vars->samplespacing*A;
	sprintf_s(msg,MSGSIZE,"Firstrange acf: %f",frAcf);
	PyObject_CallMethod(vars->log,"info","s",msg);

	powerrangearray = createArray(PyArray_FLOAT,2,1,vars->npowergates);
	range = (float *) PyArray_DATA(powerrangearray);
	for (i=0;i<vars->npowergates;i++)
		range[i] = frPwr+(float)i*vars->samplespacing;
	h5Static(vars->self,vars->root,"Power/Range",powerrangearray);
	h5Attribute(vars->self,vars->root,"Power/Range/Unit",Py_BuildValue("s","m"));

	acfrangearray = createArray(PyArray_FLOAT,2,1,vars->ngates);
	drangeacf = vars->gatestep*vars->samplespacing;
	range = (float *) PyArray_DATA(acfrangearray);
	for (i=0;i<vars->ngates;i++)
		range[i] = frAcf+(float)i*drangeacf;
	h5Static(vars->self,vars->root,"Spectra/Range",acfrangearray);
	h5Attribute(vars->self,vars->root,"Spectra/Range/Unit",Py_BuildValue("s","m"));
	
	freqarray = createArray(PyArray_FLOAT,2,1,vars->nlags);
	freq = (float *) PyArray_DATA(freqarray);
	fs = 1.0/vars->sampletime;
	fb = fs/vars->nlags;
	f = -fs/2.0;
	for (i=0;i<vars->nlags;i++)
	{	
		freq[i] = f;
		f += fb;
	};				
	h5Static(vars->self,vars->root,"Spectra/Frequency",freqarray);
	h5Attribute(vars->self,vars->root,"Spectra/Frequency/Unit",Py_BuildValue("s","Hz"));


	// Various attributes
	h5Attribute(vars->self,vars->root,"Power/Data/Unit",Py_BuildValue("s","Samples^2"));
	h5Attribute(vars->self,vars->root,"Spectra/Data/Unit",Py_BuildValue("s","Samples^2"));

	// Temp storage
	vars->gate = createArray(PyArray_FLOAT,1,vars->nlags*COMPLEX);
	vars->gates = createArrayZeros(PyArray_FLOAT,3,vars->beamcodes->npos,vars->nlags,vars->ngates);

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
	Py_XDECREF(vars->gate);
	Py_XDECREF(vars->gates);

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
initspectralpwr(void)
{
	PyObject *m;

	m = Py_InitModule("spectralpwr", extMethods);
	if (m == NULL)
		return;
	import_array();
};

