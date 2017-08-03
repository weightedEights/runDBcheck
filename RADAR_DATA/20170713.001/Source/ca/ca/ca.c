// Global definitions and includes
#include "common.h"

// Mode specific includes
#include "ca.h"

// Common mode helper includes
#include "modeCommon.h"


// dll exported process routine
DWORD process(int inst,int npulse, void *inbuf)
{
	localvars *vars;
	DWORD *hdr,*sd;
	int bp,bcRet;
	char path[MAXPATH],msg[MSGSIZE];
	PyGILState_STATE gstate;
    float *fp,*samples,*nsidata,*ppower,*paverage;
	PyObject *power,*average;
	PyObject *pulsesintegrated,*beamcodes;
	int *pbc,*ppi;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		bcReset(vars->beamcodes);
		floatZeroArray(vars->nsiData);
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
		samples = (float *)PyArray_DATA(vars->samples);
		nsidata = (float *)PyArray_DATA(vars->nsiData)+bp*vars->ngates*COMPLEX;
		cvfloat(sd,samples,vars->calcstep,vars->ngates);
		cvadd(samples,nsidata,nsidata,vars->ngates);
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

		nsidata = (float *)PyArray_DATA(vars->nsiData);
		power = createArray(PyArray_FLOAT,2,vars->beamcodes->npos,vars->ngates);
		ppower = (float *)PyArray_DATA(power);
		CalcPower(nsidata,ppower,vars->beamcodes->npos,1,vars->ngates);

		average = createArray(PyArray_FLOAT,3,vars->beamcodes->npos,vars->sgates,COMPLEX);
		paverage = (float *)PyArray_DATA(average);
		SubIntegrate(nsidata,paverage,vars->beamcodes->npos,vars->ngates,vars->sgates,1,vars->subint,vars->substep);

		// save arrays
		gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects
		h5Dynamic(vars->self,vars->root,"Beamcodes",beamcodes);
		h5Dynamic(vars->self,vars->root,"PulsesIntegrated",pulsesintegrated);

		h5Dynamic(vars->self,vars->root,"Power/Data",power);
		h5Dynamic(vars->self,vars->root,"Average/Data",average);

		PyGILState_Release(gstate);

		// release arrays
		Py_XDECREF(beamcodes);
		Py_XDECREF(pulsesintegrated);
		Py_XDECREF(power);
		Py_XDECREF(average);
	};

	return ecOk;
};



// Python exported routines

static PyObject *ext_configure(PyObject *self, PyObject *args)
{
	PyObject *parent;
	PyObject *powerrange,*unirange;
	int i,ires;
	char msg[MSGSIZE];
	char path[MAXPATH];
	double fr,frPwr,frAcf,drangeuni;
	float *range;
	localvars *vars;

	if (!PyArg_ParseTuple(args, "O",&parent))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure(self), self=mode class");
		goto error;
	};

	vars = modeInit(parent);
	if (vars == NULL)
		goto error;

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

    //calculate and save range array
	fr = vars->firstrange-
	     vars->filterrangedelay+
		 vars->rangecorrection;
	sprintf_s(msg,MSGSIZE,"Firstrange: %f",fr);
	PyObject_CallMethod(vars->log,"info","s",msg);

	frPwr = fr;
	frAcf = fr+vars->samplespacing*((vars->substep-1)/2);
	powerrange = createArray(PyArray_FLOAT,2,1,vars->ngates);
	range = (float *) PyArray_DATA(powerrange);
	for (i=0;i<vars->ngates;i++)
		range[i] = frPwr+(float)i*vars->samplespacing;
	unirange = createArray(PyArray_FLOAT,2,1,vars->sgates);
	range = (float *)PyArray_DATA(unirange);
	drangeuni = vars->substep*vars->samplespacing;
	for (i=0;i<vars->sgates;i++)
		range[i] = frAcf+(float)i*drangeuni;
	
	h5Static(vars->self,vars->root,"Power/Range",powerrange);
	h5Attribute(vars->self,vars->root,"Power/Range/Unit",Py_BuildValue("s","m"));

	h5Static(vars->self,vars->root,"Average/Range",unirange);
	h5Attribute(vars->self,vars->root,"Average/Range/Unit",Py_BuildValue("s","m"));

	vars->nsiData = createArray(PyArray_FLOAT,3,vars->beamcodes->npos,vars->ngates,COMPLEX);
	vars->samples = createArray(PyArray_FLOAT,1,vars->ngates*COMPLEX);

	// Various attributes
	//h5Attribute(vars->self,vars->root,"Power/Data/Unit",Py_BuildValue("s","Samples^2"));
	//h5Attribute(vars->self,vars->root,"Samples/Data/Unit",Py_BuildValue("s","Samples"));


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
	Py_XDECREF(vars->nsiData);
	Py_XDECREF(vars->samples);

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
initca(void)
{
	PyObject *m;

	m = Py_InitModule("ca", extMethods);
	if (m == NULL)
		return;
	import_array();
};

