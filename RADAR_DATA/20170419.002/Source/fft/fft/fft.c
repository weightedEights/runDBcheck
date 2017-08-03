// Global definitions and includes
#include "common.h"

// Mode specific includes
#include "fft.h"

// Common mode helper includes
#include "modeCommon.h"


// dll exported process routine
DWORD process(int inst,int npulse, void *inbuf)
{
	localvars *vars;
	DWORD *hdr,*sd;
	short *sr,*si,*sid;
	int bp,bcRet;
	PyGILState_STATE gstate;
    float *spec;
	PyObject *pulsesintegrated,*beamcodes;
	int *pbc,*ppi;
	int nsamples,ndatasamples;
	int i,j;

	vars = lv[inst];

	if (npulse == 0)  //first pulse in integration
	{
		bcReset(vars->beamcodes);
		vars->spectra = createArrayZeros(PyArray_FLOAT,2,vars->beamcodes->npos,vars->nlags);
	};

	hdr = (DWORD *)inbuf;
	if (hdr[0] != 0x00000100) // header sanity check
		return ecSamplesMisalignment;

	if (hdr[10] == vars->modegroup) // modegroup test
	{
	    sd = hdr+vars->nradacheaderwords+vars->indexsample;
	    sid = (short *)sd;
	    
	    sr = sid;
	    si = sid+1;
	    if (vars->fftsign < 0)
	    {
	        sr = sid+1;
	        si = sid;
        }
	    
		// Check to see if beamcode was configured
		bcRet = bcIndex(vars->beamcodes,hdr[4],&bp);
		if (bcRet < 0) // beamcode not in list
			return ecBeamcodeNotInList;
			
		// extract header values
		nsamples = (hdr[11]<<16)+hdr[12];
		ndatasamples = nsamples-vars->indexsample; //subtract off number of tx pulse samples
		
		// load data
		//fftwZeroComplex(vars->data,vars->nlags); 
		memset(vars->data,0,vars->nlags*sizeof(fftwf_complex)); //clear first
		for (i=0,j=0;i<ndatasamples;i++,j+=2)
		{
		    vars->data[i][0] = (float)sr[j];
		    vars->data[i][1] = (float)si[j];
	    }
	    fftwSpectra(vars->p1,vars->nlags,vars->data,vars->fdata);
	    spec = (float *)PyArray_DATA(vars->spectra)+bp*vars->nlags;
	    cvAddFloats(spec,vars->fdata,spec,vars->nlags);		
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

		h5Dynamic(vars->self,vars->root,"Spectra/Data",vars->spectra);

		PyGILState_Release(gstate);

		// release arrays
		Py_XDECREF(beamcodes);
		Py_XDECREF(pulsesintegrated);
		Py_XDECREF(vars->spectra);
	};

	return ecOk;
};



// Python exported routines

static PyObject *ext_configure(PyObject *self, PyObject *args)
{
	PyObject *parent;
	PyObject *freqarray;
	int i;
	char msg[MSGSIZE];
	float *freq;
	double fs,fb,f;
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

	//fftsign
	if (!getInt(vars->pars,"fftsign",&vars->fftsign))
	{
		sprintf_s(msg,MSGSIZE,"fftsign not in parameters dict. Defaults to 1");
		PyObject_CallMethod(vars->log,"info","s",msg);
		vars->fftsign = 1;
	};
	sprintf_s(msg,MSGSIZE,"fftsign set to: %i",vars->fftsign);
	PyObject_CallMethod(vars->log,"info","s",msg);
	
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
	h5Attribute(vars->self,vars->root,"Spectra/Data/Unit",Py_BuildValue("s","Samples^2"));

	// Temp storage
	vars->data = fftwf_malloc(vars->nlags*sizeof(fftwf_complex));
	vars->fdata = fftwf_malloc(vars->nlags*sizeof(float));
	vars->p1 = fftwf_plan_dft_1d(vars->nlags,vars->data,vars->data,FFTW_FORWARD,FFTW_ESTIMATE);
	
	PyObject_CallMethod(vars->log,"info","s","Configuration done");

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
	
	fftwf_free(vars->data);
	fftwf_free(vars->fdata);
    fftwf_destroy_plan(vars->p1);

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
initfft(void)
{
	PyObject *m;

	m = Py_InitModule("fft", extMethods);
	if (m == NULL)
		return;
	import_array();
};

