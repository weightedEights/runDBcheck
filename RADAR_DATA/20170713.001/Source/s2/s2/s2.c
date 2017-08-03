// Global definitions and includes
#include "common.h"

// Mode specific includes
#include "s2.h"

// Common mode helper includes
#include "modeCommon.h"


// dll exported process routine
DWORD process(int inst,int npulse, void *samples)
{
	localvars *vars;
	DWORD *hdr,*sd;
	int ilagno,istrt_indx,icalc_indx,bp,bcRet;
	PyGILState_STATE gstate;
    PyObject *pwr,*acfs,*pulsesintegrated,*beamcodes;
	int *ppi,*pbc;
	int nsamples;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		bcReset(vars->beamcodes);
		floatZeroArray(vars->nsiData);
	}

	hdr = (DWORD *)samples;
	if (hdr[0] != 0x00000100) // header sanity check
		return ecSamplesMisalignment;

	sd = (DWORD *) hdr+vars->nradacheaderwords+vars->indexsample;


	if (hdr[10] == vars->modegroup)
	{
		// Check to see if beamcode was configured
		bcRet = bcIndex(vars->beamcodes,hdr[4],&bp);
		if (bcRet < 0) // beamcode not in list
			return ecBeamcodeNotInList;


		nsamples = (hdr[11]<<16)+hdr[12];

		// convert samples to floats and fold if applicable
		if (vars->fold)
			cvCfold(sd,vars->samples,nsamples);
		else
			cvCfloats(sd,vars->samples,nsamples);


    	// Calculate Uniprog array

		// calculate ngates of acfs
		istrt_indx = (long)((vars->nlags-1)/2);
		cvCmags(vars->samples+istrt_indx,PyArray_DATA(vars->acfs),vars->calcstep,vars->ngates);
		for (ilagno=1;ilagno<vars->nlags;ilagno++) 
		{
			icalc_indx = istrt_indx - (long) (ilagno/2);
			cvCmul(vars->samples+(icalc_indx),
				  vars->samples+(icalc_indx+ilagno),
				  (float *)PyArray_DATA(vars->acfs)+ilagno*vars->ngates*COMPLEX,
				  vars->calcstep,vars->ngates);
		}
		cvadd(PyArray_DATA(vars->acfs),
		      (float *)PyArray_DATA(vars->nsiData)+bp*vars->nlags*vars->ngates*COMPLEX,
		      (float *)PyArray_DATA(vars->nsiData)+bp*vars->nlags*vars->ngates*COMPLEX,
			  vars->nlags*vars->ngates);
	};

	if (npulse == vars->npulsesint-1) // Last pulse
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

		// Create arrays to be saved
		pwr = createArray(PyArray_FLOAT,2,vars->beamcodes->npos,vars->ngates);
		GetPower(PyArray_DATA(vars->nsiData),PyArray_DATA(pwr),vars->beamcodes->npos,vars->nlags,vars->ngates);
		if (!vars->poweronly)
		{
			acfs = createArray(PyArray_FLOAT,4,vars->beamcodes->npos,vars->nlags,vars->sgates,COMPLEX);
			SubIntegrate(PyArray_DATA(vars->nsiData),PyArray_DATA(acfs),vars->beamcodes->npos,vars->ngates,vars->sgates,vars->nlags,vars->subint,vars->substep);
		}

		//Save data
		gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects
		h5Dynamic(vars->self,vars->root,"Beamcodes",beamcodes);
		h5Dynamic(vars->self,vars->root,"PulsesIntegrated",pulsesintegrated);

		h5Dynamic(vars->self,vars->root,"Power/Data",pwr);
		if (!vars->poweronly)
			h5Dynamic(vars->self,vars->root,"Acf/Data",acfs);
		PyGILState_Release(gstate);
		Py_XDECREF(beamcodes);
		Py_XDECREF(pulsesintegrated);
		Py_XDECREF(pwr);
		if (!vars->poweronly)
			Py_XDECREF(acfs);
	};
	return ecOk;
};



// Python exported routines

static PyObject *ext_configure(PyObject *self, PyObject *args)
{
	PyObject *parent;
	PyObject *powerrangearray,*acfrangearray,*lagsarray;
	int i,ires;
	char msg[MSGSIZE];
	int A,B;
	float frPwr,frAcf,drangeacf;
	float *range,*lags;
	localvars *vars;

	if (!PyArg_ParseTuple(args, "O",&parent))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure(self), self=mode class");
		goto error;
	};

	vars = modeInit(parent);

	//Read mode specific parameters

	//nlags
	if (!getInt(vars->pars,"nlags",&vars->nlags))
	{
		PyErr_SetString(PyExc_KeyError,"nlags not in parameters dict");
		goto error;
	};
	sprintf_s(msg,MSGSIZE,"nlags set to: %i",vars->nlags);
	PyObject_CallMethod(vars->log,"info","s",msg);

	//subint
	if (!getInt(vars->pars,"subint",&ires))
	{
		vars->subint = 1;
		sprintf_s(msg,MSGSIZE,"subint not in parameters dict, defaults to 1");
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	else
	{
		vars->subint = ires;
		sprintf_s(msg,MSGSIZE,"subint set to: %i",vars->subint);
		PyObject_CallMethod(vars->log,"info","s",msg);
	};

	//substep
	if (!getInt(vars->pars,"substep",&ires))
	{
		vars->substep = vars->subint;
		sprintf_s(msg,MSGSIZE,"substep not in parameters dict, defaults to subint");
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	else
	{
		vars->substep = ires;
		sprintf_s(msg,MSGSIZE,"substep set to: %i",vars->subint);
		PyObject_CallMethod(vars->log,"info","s",msg);
	};

	//fold
	if (!getInt(vars->pars,"fold",&ires))
	{
		vars->fold = 0;
		sprintf_s(msg,MSGSIZE,"fold not in parameters dict, defaults to 0");
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	else
	{
		vars->fold = ires;
		sprintf_s(msg,MSGSIZE,"fold set to: %i",vars->fold);
		PyObject_CallMethod(vars->log,"info","s",msg);
	};

	// number of gates after subint
	vars->sgates = vars->ngates/vars->subint;

	//calcstep
	if (!getInt(vars->pars,"calcstep",&ires))
	{
		vars->calcstep = 1;
		sprintf_s(msg,MSGSIZE,"calcstep not in parameters dict, defaults to 1");
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	else
	{
		vars->calcstep = ires;
		sprintf_s(msg,MSGSIZE,"calcstep set to: %i",vars->calcstep);
		PyObject_CallMethod(vars->log,"info","s",msg);
	};

	if (getInt(vars->pars,"poweronly",&ires))
	{
		vars->poweronly = ires;
		sprintf_s(msg,MSGSIZE,"poweronly set to: %i",vars->poweronly);
		PyObject_CallMethod(vars->log,"info","s",msg);
	}
	else
		vars->poweronly = 0;

    //calculate and save range arrays
	A = (vars->nlags-1)/2;
	B = (vars->subint-1)/2;
	frPwr = vars->firstrange-
			vars->filterrangedelay+
			vars->rangecorrection+
			vars->samplespacing*A;
	sprintf_s(msg,MSGSIZE,"Firstrange power: %f",frPwr);
	PyObject_CallMethod(vars->log,"info","s",msg);
	frAcf = vars->firstrange-
			vars->filterrangedelay+
			vars->rangecorrection+
			vars->samplespacing*(A+B);
	sprintf_s(msg,MSGSIZE,"Firstrange acf: %f",frAcf);
	PyObject_CallMethod(vars->log,"info","s",msg);

	powerrangearray = createArray(PyArray_FLOAT,2,1,vars->ngates);
	range = (float *) PyArray_DATA(powerrangearray);
	for (i=0;i<vars->ngates;i++)
		range[i] = frPwr+(float)i*vars->samplespacing;
	h5Static(vars->self,vars->root,"Power/Range",powerrangearray);
	h5Attribute(vars->self,vars->root,"Power/Range/Unit",Py_BuildValue("s","m"));

	if (!vars->poweronly)
	{
		acfrangearray = createArray(PyArray_FLOAT,2,1,vars->sgates);
		drangeacf = vars->substep*vars->samplespacing;
		range = (float *) PyArray_DATA(acfrangearray);
		for (i=0;i<vars->sgates;i++)
			range[i] = frAcf+(float)i*drangeacf;
		h5Static(vars->self,vars->root,"Acf/Range",acfrangearray);
		h5Attribute(vars->self,vars->root,"Acf/Range/Unit",Py_BuildValue("s","m"));
		
		lagsarray = createArray(PyArray_FLOAT,2,1,vars->nlags);
		lags = (float *) PyArray_DATA(lagsarray);
		for (i=0;i<vars->nlags;i++)
			lags[i] = (float)i*vars->sampletime;
		h5Static(vars->self,vars->root,"Acf/Lags",lagsarray);
		h5Attribute(vars->self,vars->root,"Acf/Lags/Unit",Py_BuildValue("s","s"));
	}
	// float sample buffer
	vars->samples = fftwf_malloc(vars->maxsamples*sizeof(fftwf_complex));

	// Setup arrays to hold non subintegrated data while integrating
	vars->nsiData = createArray(PyArray_FLOAT,4,vars->beamcodes->npos,vars->nlags,vars->ngates,COMPLEX);
	vars->acfs = createArray(PyArray_FLOAT,3,vars->nlags,vars->ngates,COMPLEX);

	// Various attributes
	h5Attribute(vars->self,vars->root,"Power/Data/Unit",Py_BuildValue("s","Samples^2"));
	if (!vars->poweronly)
		h5Attribute(vars->self,vars->root,"Acf/Data/Unit",Py_BuildValue("s","Samples^2"));
			
	PyObject_CallMethod(vars->log,"info","s","Configuration done");


	Py_RETURN_NONE;
error:
	return NULL;
};


static PyObject *ext_shutdown(PyObject *self, PyObject *args)
{
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

	// free beamcodes
	bcFree(vars->beamcodes);

	// free complex float sample buffer
	fftwf_free(vars->samples);

	//Free integration arrays
	Py_XDECREF(vars->acfs); 
	Py_XDECREF(vars->nsiData);

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
		free(lv);
		lv = NULL;
		lvCount = 0;
	};
			
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
inits2(void)
{
	PyObject *m;

	m = Py_InitModule("s2", extMethods);
	if (m == NULL)
		return;
	import_array();
};

