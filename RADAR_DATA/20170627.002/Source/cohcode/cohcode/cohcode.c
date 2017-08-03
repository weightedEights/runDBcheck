// Global definitions and includes
#include "common.h"

// Mode specific includes
#include "cohcode.h"
#include "decode.h"
#include "codeutils.h"

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
    float *ppower,*pdbuf,*pbuf;
	char *pcode;
	PyObject *pulsesintegrated,*beamcodes;
	int *pbc,*ppi;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		bcReset(vars->beamcodes);
		vars->power = createArrayZeros(PyArray_FLOAT,2,vars->beamcodes->npos,vars->ngates);
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

		pcode = (char *)PyArray_DATA(vars->code);
		strcohcode(pcode,vars->codelength,hdr+13);

		pbuf = (float *)PyArray_DATA(vars->buf);
		pdbuf = (float *)PyArray_DATA(vars->dbuf);
		cvfloats(sd,pbuf,vars->ngates+vars->codelength);
		decode(vars->codelength,pcode,pbuf,pdbuf,vars->ngates);

		ppower = PyArray_DATA(vars->power);
		cvfrealmuladd(pdbuf,pdbuf,ppower+bp*vars->ngates,1,vars->ngates);
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

		// save arrays
		gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects
		h5Dynamic(vars->self,vars->root,"Beamcodes",beamcodes);
		h5Dynamic(vars->self,vars->root,"PulsesIntegrated",pulsesintegrated);

		h5Dynamic(vars->self,vars->root,"Power/Data",vars->power);

		PyGILState_Release(gstate);

		// release arrays
		Py_XDECREF(beamcodes);
		Py_XDECREF(pulsesintegrated);
		Py_XDECREF(vars->power);
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

	if (!PyArg_ParseTuple(args, "O",&parent))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure(self), self=mode class");
		goto error;
	};

	vars = modeInit(parent);
	if (vars == NULL)
		goto error;

	// Mode specific parameters

	//codelength
	if (!getInt(vars->pars,"codelength",&vars->codelength))
	{
		PyErr_SetString(PyExc_KeyError,"codelength not in parameters dict");
		goto error;
	};
	sprintf_s(msg,MSGSIZE,"codelength set to: %i",vars->codelength);
	PyObject_CallMethod(vars->log,"info","s",msg);

	//baud
	if (!getInt(vars->pars,"baud",&vars->baud))
	{
		sprintf_s(msg,MSGSIZE,"Baud not specified. Using CodeLength instead");
		PyObject_CallMethod(vars->log,"info","s",msg);
		vars->baud = vars->codelength;
	};
	sprintf_s(msg,MSGSIZE,"baud set to: %i",vars->baud);
	PyObject_CallMethod(vars->log,"info","s",msg);

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
	h5Static(vars->self,vars->root,"Power/Range",rangearray);
	h5Attribute(vars->self,vars->root,"Power/Range/Unit",Py_BuildValue("s","m"));

	// Various attributes
	h5Attribute(vars->self,vars->root,"Power/Data/Unit",Py_BuildValue("s","Samples^2"));

	// Temp storage
	vars->code = createArray(PyArray_CHAR,1,vars->codelength+1); //+1 leaves room for term #0
	vars->buf = createArray(PyArray_FLOAT,1,(vars->ngates+vars->codelength)*COMPLEX);
	vars->dbuf = createArray(PyArray_FLOAT,1,(vars->ngates+vars->codelength)*COMPLEX);

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

	Py_XDECREF(vars->buf);
	Py_XDECREF(vars->dbuf);
	bcFree(vars->beamcodes);

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
initcohcode(void)
{
	PyObject *m;

	m = Py_InitModule("cohcode", extMethods);
	if (m == NULL)
		return;
	import_array();
};

