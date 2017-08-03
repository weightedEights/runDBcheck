// Global definitions and includes
#include "common.h"

// Mode specific includes
#include "incohcodefl1.h"
#include "codeutils.h"

// Common mode helper includes
#include "modeCommon.h"

#include "calc.h"



// dll exported process routine
DWORD process(int inst,int npulse, void *inbuf)
{
	localvars *vars;
	DWORD *hdr,*sd;
	int bp,bcRet;
	long istrt_indx;
	char *code,path[MAXPATH],msg[MSGSIZE];
	float *pacfs,*pbuf,*ppower;
	int ilagno,offset,s0,s1;
	PyGILState_STATE gstate;
	PyObject *pulsesintegrated,*beamcodes,*acfs;
	int *pbc,*ppi,*pind,*pmat;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		bcReset(vars->beamcodes);

		vars->acfs = createArrayZeros(PyArray_FLOAT,4,vars->beamcodes->npos,vars->nlags,vars->ngates,COMPLEX);
		vars->power = createArrayZeros(PyArray_FLOAT,2,vars->beamcodes->npos,vars->ngates);
	}

	hdr = (DWORD *)inbuf;
	if (hdr[0] != 0x00000100) // header sanity check
		return ecSamplesMisalignment;
	sd = hdr+vars->nradacheaderwords+vars->indexsample;

	if (hdr[10] == vars->modegroup) // modegroup test
	{
		// Check to see if beamcode was configured
		bcRet = bcIndex(vars->beamcodes,hdr[4],&bp);
		if (bcRet < 0) // beamcode not in list
		{
			gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects
			sprintf_s(msg,MSGSIZE,"beamcode: %x, index: %i",hdr[4],bp);
			PyObject_CallMethod(vars->log,"info","s",msg);
			PyGILState_Release(gstate);
			return ecBeamcodeNotInList;
		}

		s0 = vars->nlags*vars->ngates*COMPLEX;
		s1 = vars->ngates*COMPLEX;

		code = (char *) PyArray_DATA(vars->code);
		strincohcode(code,vars->codelength,hdr+13);
		pmat = PyArray_DATA(vars->lagmat);
		pind = PyArray_DATA(vars->lagind);
		pbuf = (float *)PyArray_DATA(vars->buf);
		cvmags(sd,pbuf,vars->calcstep,vars->ngates);

		pacfs = (float *)PyArray_DATA(vars->acfs)+bp*s0;
		cvadd(pbuf,pacfs,pacfs,vars->ngates);

		ppower = (float *)PyArray_DATA(vars->power)+bp*vars->ngates;
		cvaddreal(pbuf,ppower,ppower,vars->ngates);

		for (ilagno=1;ilagno<vars->nlags;ilagno++) 
		{
			floatZeroArray(vars->buf);
			pacfs = (float *)PyArray_DATA(vars->acfs)+bp*s0+ilagno*s1;
			cvFractionalLags(sd,pbuf,vars->calcstep,vars->baud,vars->fraction,pind[ilagno],pmat[ilagno],code,vars->ngates);
			cvadd(pacfs,pbuf,pacfs,vars->ngates);
		}
	}

	if (npulse == vars->npulsesint-1)
	{
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

		// reduce acfs to effective lags only
		acfs = createArrayZeros(PyArray_FLOAT,4,vars->beamcodes->npos,vars->elags,vars->ngates,COMPLEX);
		reduceLags(vars->acfs,vars->lagmat,acfs);

		gstate = PyGILState_Ensure(); // Make sure we have the GIL before accessing python objects

		h5Dynamic(vars->self,vars->root,"Beamcodes",beamcodes);
		h5Dynamic(vars->self,vars->root,"PulsesIntegrated",pulsesintegrated);

		h5Dynamic(vars->self,vars->root,"Power/Data",vars->power);
		h5Dynamic(vars->self,vars->root,"Acf/Data",acfs);

		PyGILState_Release(gstate);

		// release arrays
		// power
		Py_XDECREF(beamcodes);
		Py_XDECREF(pulsesintegrated);
		Py_XDECREF(vars->power);
		Py_XDECREF(vars->acfs);
		Py_XDECREF(acfs);
	};

	return ecOk;
};



// Python exported routines

static PyObject *ext_configure(PyObject *self, PyObject *args)
{
	PyObject *par,*pars,*parent,*var;
	PyObject *rangearray,*lagsarray,*pwrrangearray;
	int nx,bb,s,e,m,n,i,instance,*pmat,*pind;
	enum PyArray_TYPES arrType;
	char msg[MSGSIZE];
	char path[MAXPATH];
	double fr,dr,frpwr;
	float *range,*pwrrange,*lags;
	localvars *vars;
	char sres[STRINGVARSIZE];
	double dres,txbaud,pulsewidth;
	int ires,l;
	PyArrayObject *arr, *tmp;

	if (!PyArg_ParseTuple(args, "O",&parent))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure(self), self=mode class");
		goto error;
	};

	vars = modeInit(parent);

	//Read mode specific parameters

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

	//fraction
	if (!getInt(vars->pars,"fraction",&vars->fraction))
	{
		PyErr_SetString(PyExc_KeyError,"fraction not in parameters dict");
		goto error;
	};
	sprintf_s(msg,MSGSIZE,"fraction set to: %i",vars->fraction);
	PyObject_CallMethod(vars->log,"info","s",msg);

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

	//pulsewidth
	if (!getDouble(vars->pars,"pulsewidth",&pulsewidth))
	{
		PyErr_SetString(PyExc_KeyError,"pulsewidth not in parameters dict");
		goto error;
	}

	//txbaud
	if (!getDouble(vars->pars,"txbaud",&txbaud))
	{
		PyErr_SetString(PyExc_KeyError,"txbaud not in parameters dict");
		goto error;
	}

	// Compute nlags based on fraction and baud
	vars->nlags = (vars->baud-1)*(2*vars->fraction-1)+1; //+1 for the zerolag
	sprintf_s(msg,MSGSIZE,"nlags computed: %i",vars->nlags);
	PyObject_CallMethod(vars->log,"info","s",msg);

    //calculate and save range array
	fr = vars->firstrange-
	     vars->filterrangedelay+
		 vars->rangecorrection;
	sprintf_s(msg,MSGSIZE,"Firstrange: %f",fr);
	PyObject_CallMethod(vars->log,"info","s",msg);

	// correct power first range
	frpwr = fr+txbaud*1e-6/2.0*C/2.0;
	frpwr -= pulsewidth*1e-6/2.0*C/2.0;

	rangearray = createArray(PyArray_FLOAT,2,1,vars->ngates);
	pwrrangearray = createArray(PyArray_FLOAT,2,1,vars->ngates);
	range = (float *) PyArray_DATA(rangearray);
	pwrrange = (float *) PyArray_DATA(pwrrangearray);
	for (i=0;i<vars->ngates;i++)
	{
		range[i] = fr+(float)i*vars->samplespacing;
		pwrrange[i] = frpwr+(float)i*vars->samplespacing;
	}
	h5Static(vars->self,vars->root,"Acf/Range",rangearray);
	h5Attribute(vars->self,vars->root,"Acf/Range/Unit",Py_BuildValue("s","m"));
	h5Static(vars->self,vars->root,"Power/Range",pwrrangearray);
	h5Attribute(vars->self,vars->root,"Power/Range/Unit",Py_BuildValue("s","m"));

	// Various attributes
	h5Attribute(vars->self,vars->root,"Power/Data/Unit",Py_BuildValue("s","Samples^2"));
	h5Attribute(vars->self,vars->root,"Acf/Data/Unit",Py_BuildValue("s","Samples^2"));

	// Temp storage
	vars->buf = createArray(PyArray_FLOAT,1,(vars->nlags+vars->ngates+vars->codelength)*COMPLEX);
	vars->code = createArray(PyArray_CHAR,1,vars->codelength+1); //+1 leaves room for term #0

	// Build code matrices
	nx = 2*vars->fraction-1;
	vars->lagmat = createArrayZeros(PyArray_INT,1,(vars->baud-1)*nx+1);
	vars->lagind = createArrayZeros(PyArray_INT,1,(vars->baud-1)*nx+1);
	pmat = (int *)PyArray_DATA(vars->lagmat);
	pind = (int *)PyArray_DATA(vars->lagind);
	nx = 2*vars->fraction-1;
	for (bb=1;bb<vars->baud;bb++)
	{
		s = 1+(bb-1)*nx;
		e = 1+bb*nx;
		m = 0;
		for (n=s;n<e;n++)
		{
			pmat[n] = (bb-1)*vars->fraction+1+m;
			pind[n] = bb;
			m++;
		}
	}
//	h5Static(vars->self,vars->root,"Acf/Lagmat",vars->lagmat);
//	h5Static(vars->self,vars->root,"Acf/Lagind",vars->lagind);
			
	// Effective lags eg. lags in output array
	vars->elags = 0;
	for (l=0;l<vars->nlags;l++)
	{
		if (pmat[l] > vars->elags)
			vars->elags = pmat[l];
	}
	vars->elags += 1;

	//calculate and save lags array
	lagsarray = createArray(PyArray_FLOAT,2,1,vars->elags);
	lags = (float *)PyArray_DATA(lagsarray);
	for (i=0;i<vars->elags;i++)
		lags[i] = (float)i*vars->sampletime;
	h5Static(vars->self,vars->root,"Acf/Lags",lagsarray);
	h5Attribute(vars->self,vars->root,"Acf/Lags/Unit",Py_BuildValue("s","s"));

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
	// free beamcodes
	bcFree(vars->beamcodes);

	//Free integration arrays
	Py_XDECREF(vars->code); 
	Py_XDECREF(vars->buf);
	Py_XDECREF(vars->lagmat);
	Py_XDECREF(vars->lagind);

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
initincohcodefl1(void)
{
	PyObject *m;

	m = Py_InitModule("incohcodefl1", extMethods);
	if (m == NULL)
		return;
	import_array();
};

