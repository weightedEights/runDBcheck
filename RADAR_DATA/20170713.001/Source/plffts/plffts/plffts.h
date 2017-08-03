
//Mode definitions
#define MODENAME "PLFFTS"
#define MODEVERSION "2.0"
#define MODEMODIFIED 20070927
#define MODETYPE 2010

//Misc definitions
#define COMPLEX 2
#define NCODES 19

//Current radac header values are: --
//  addr  Info
// 0     Header type 0x00000100
// 1     Number of header words
// 2     Frame Count
// 3     Pulse Count
// 4     Beam code
// 5     Time count
// 6     Clock status
// 7     Clock Msw
// 8     Clock Lsw
// 9     Datatype
// 10    Modegroup
// 11	   nspulse Msw
// 12    nspulse Lsw
// 13-31 Code
                       
typedef struct
{
	// Common parameters
	//Instance
	int instance;

	//Root path into H5 data file
	char root[MAXPATH];

	// Python objects
	PyObject *self;
	PyObject *log;
	PyObject *pars;

	// Integration variables
	BEAMCODES *beamcodes;
	int nradacheaderwords;
	int npulsesint; // number of pulses to integrate
	int nactivepulses; //number of active (modegroup) selected pulses to integrate
	int ntotalpulses; //total number of active pulses per integration
	int modegroup;  // only process samples to this modegroup
	int indexsample; // index of first sample to process
	int ngates;  // number of gates
	double samplespacing; // range between samples
	double sampletime;  // time between samples
    double firstrange;  // first range in meters
	double filterrangedelay;  // filter range delay in meters
	double rangecorrection;  // range correction factor in meters


	// Mode specific parameters
	int fftsign;
	int nlags;
	int npowergates;
	int gatestep;

	// Storage
	PyObject *gate;
	PyObject *gates;
	PyObject *power;
	PyObject *spectra;

} localvars;

//export process function so integrator can get to it...
#define dllexp __declspec(dllexport)

dllexp DWORD process(int inst,int npulse,void *inbuf);


