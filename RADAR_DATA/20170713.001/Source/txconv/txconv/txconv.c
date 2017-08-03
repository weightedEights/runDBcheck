// Global definitions and includes

// Make sure we don't try to use python_d libraries
// Have also changed project settings to link against no debug libraries
#include "common.h"

// Mode specific includes
#include "txconv.h"

// Common mode helper includes
#include "modeCommon.h"


// dll exported process routine
DWORD process(int inst,int npulse, void *inbuf)
{
	localvars *vars;
	DWORD *hdr,*sdpulse,*sddata,*sd;
	int bp,bcRet;
	PyGILState_STATE gstate;
	Complex s;
	PyObject *pulsesintegrated,*beamcodes;
	int *pbc,*ppi;
	float *power;
	int i,mg;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		bcReset(vars->beamcodes);
		vars->power = createArrayZeros(PyArray_FLOAT,2,vars->beamcodes->npos,vars->ngates);
	};

	hdr = (DWORD *)inbuf;
	if (hdr[0] != 0x00000100) // header sanity check
		return ecSamplesMisalignment;
		
	mg = hdr[10];	

	// setup constants
	power = (float *)PyArray_DATA(vars->power);
	
	if ((mg == vars->modegroup) || (mg == vars->mgpulse)) // modegroup test
	{
		// Check to see if beamcode was configured
		bcRet = bcIndex(vars->beamcodes,hdr[4],&bp);
		if (bcRet < 0) // beamcode not in list
			return ecBeamcodeNotInList;

		fftwZeroComplex(vars->resp,vars->n2);
		fftwZeroComplex(vars->data,vars->n2);
		//fftwZeroComplex(vars->odata,vars-n2);
		//fftwZeroFloat(vars->pwr,vars->ngates);
			
		if (vars->interleaved)
		{
		    sdpulse = hdr+vars->nradacheaderwords+vars->indexsample;
		    sddata = hdr+vars->nradacheaderwords+vars->indexsample;
	    }
	    else
	    {	
		    sdpulse = hdr+vars->nradacheaderwords+vars->indexsample;
		    sddata = hdr+vars->nradacheaderwords+vars->indexsample+vars->txplen;
		}	
	    if ((mg == vars->mgpulse) || (!vars->interleaved))
	    {	
		    // pack response array
	        sd = sdpulse;
		    for (i=0;i<vars->txplen;i++)
		    {
		        s.c = *(sd+i);
		        vars->resp[i][0] = (float) s.s.r;
		        vars->resp[i][1] = (float) s.s.i;
	        }
        }
	    
	    if ((mg == vars->modegroup)||(!vars->interleaved))
	    {
	        // pack data array
	        sd = sddata;
	        for (i=0;i<vars->ngates;i++)
	        {
	            s.c = *(sd+i);
	            vars->data[i][0] = (float) s.s.r;
	            vars->data[i][1] = (float) s.s.i;
            }
        
            // do convolution
            fftwTxConv1(vars->p1,vars->p2,vars->p3,vars->n2,vars->ngates,vars->txplen,vars->data,vars->resp,vars->odata,vars->pwr);  
            
            // integrate
			cvAddFloats(vars->pwr,power+bp*vars->ngates,power+bp*vars->ngates,vars->ngates);

            //fftwAddReal(pwr,power+bp*vars->ngates,power+bp*vars->ngates,vars->ngates);
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
	int i,tl;
	char msg[MSGSIZE];
	float *range;
	localvars *vars;
	double fr;

	if (!PyArg_ParseTuple(args, "O",&parent))
	{
		PyErr_SetString(PyExc_TypeError,"Usage: configure(self), self=mode class");
		goto error;
	};

	vars = modeInit(parent);
	if (vars == NULL)
		goto error;

	//Read mode specific parameters
	
	if (!getInt(vars->pars,"interleaved",&vars->interleaved))
	    vars->interleaved = 0;
    sprintf_s(msg,MSGSIZE,"interleaved set to: %i",vars->interleaved);
    PyObject_CallMethod(vars->log,"info","s",msg);
	    
	if (vars->interleaved)
	{
        //modegroup for pulse
        if (!getInt(vars->pars,"txmodegroup",&vars->mgpulse))
        {
	        PyErr_SetString(PyExc_KeyError,"txmodegroup not in parameters dict");
	        goto error;
        };
	}
	else
	{
	    vars->mgpulse = vars->modegroup;
	}    
    sprintf_s(msg,MSGSIZE,"txmodegroup set to: %i",vars->mgpulse);
    PyObject_CallMethod(vars->log,"info","s",msg);

    //tx pulse length
    if (!getInt(vars->pars,"txplen",&vars->txplen))
    {
	    PyErr_SetString(PyExc_KeyError,"txplen not in parameters dict");
	    goto error;
    };
    sprintf_s(msg,MSGSIZE,"txplen set to: %i",vars->txplen);
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
		range[i] = (float) (fr+(float)i*vars->samplespacing);
	h5Static(vars->self,vars->root,"Power/Range",rangearray);
	h5Attribute(vars->self,vars->root,"Power/Range/Unit",Py_BuildValue("s","m"));

	// Various attributes
	h5Attribute(vars->self,vars->root,"Power/Data/Unit",Py_BuildValue("s","Samples^2"));

	// Compute n to the next power of 2;
	tl = vars->ngates+vars->txplen+vars->txplen/2;
	vars->n2 = (int) pow(2,ceil(log(tl)/log(2)));
    sprintf_s(msg,MSGSIZE,"n to the next power of 2: %i",vars->n2);
    PyObject_CallMethod(vars->log,"info","s",msg);

	// Temp storage
	//vars->resp = createArray(PyArray_FLOAT,2,vars->n2,2);
	//vars->data = createArray(PyArray_FLOAT,2,vars->n2,2);
	//vars->pwr = createArray(PyArray_FLOAT,1,vars->ngates);
	vars->resp = fftwf_malloc(vars->n2*sizeof(fftwf_complex));
	vars->data = fftwf_malloc(vars->n2*sizeof(fftwf_complex));
	vars->odata = fftwf_malloc(vars->n2*sizeof(fftwf_complex));
	vars->pwr = fftwf_malloc(vars->ngates*sizeof(float));
	vars->p1 = fftwf_plan_dft_1d(vars->n2,vars->data,vars->data,FFTW_FORWARD,FFTW_ESTIMATE);
	vars->p2 = fftwf_plan_dft_1d(vars->n2,vars->resp,vars->resp,FFTW_FORWARD,FFTW_ESTIMATE);
	vars->p3 = fftwf_plan_dft_1d(vars->n2,vars->odata,vars->odata,FFTW_BACKWARD,FFTW_ESTIMATE);

	PyObject_CallMethod(vars->log,"info","s","Configuration done");

	sprintf_s(msg,MSGSIZE,"Instance: %d",vars->instance);
	PyObject_CallMethod(vars->log,"info","s",msg);
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

	bcFree(vars->beamcodes);
	fftwf_free(vars->resp);
	fftwf_free(vars->data);
	fftwf_free(vars->odata);
	fftwf_free(vars->pwr);
    fftwf_destroy_plan(vars->p1);
    fftwf_destroy_plan(vars->p2);
    fftwf_destroy_plan(vars->p3);

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
inittxconv(void)
{
	PyObject *m;

	m = Py_InitModule("txconv", extMethods);
	if (m == NULL)
		return;
	import_array();
};

