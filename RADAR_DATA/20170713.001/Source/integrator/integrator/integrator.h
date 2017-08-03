#include "windows.h"
#include "process.h"

#include <Python.h>
#include <numpy/arrayobject.h>

// Misc definitions
#define MSGSIZE 256

//
// Mode handling implementation
//
typedef DWORD (*moprocessfunc)(int instance,int npulse, void *samples);
typedef struct
{
	HINSTANCE lib;
	moprocessfunc process;
	PyObject *mode;
	int index;
} tmode;

typedef struct
{
	tmode **items;
	int count;
} tmodelist;

void listClear(tmodelist *list)
{
	int i;

	if (list->count > 0)
	{
		for (i=0;i<list->count;i++)
		{
			FreeLibrary(list->items[i]->lib);
			free(list->items[i]);
		};
		free(list->items);
		list->items = NULL;
		list->count = 0;
	}
	else
		list->items = NULL;	    
};


tmode *listNewItem(tmodelist *list)
{
	int indx;

	indx = list->count;
	list->count++;
	list->items = (tmode **)realloc(list->items,sizeof(void *)*list->count);
	list->items[indx] = (tmode *)malloc(sizeof(tmode));
	return list->items[indx];
};

//
// Integration and process thread wait objects
// Doing it this way allows me to wait for both a ready and a busy state
// without using cpu cycles in a loop
//
typedef struct
{
	HANDLE busy;
	HANDLE ready;
} waitobject;

void woSetBusy(waitobject obj)
{
	ResetEvent(obj.ready);
	SetEvent(obj.busy);
};

void woSetReady(waitobject obj)
{
	ResetEvent(obj.busy);
	SetEvent(obj.ready);
};

int woIsBusy(waitobject obj)
{
	DWORD busy,ready;

	busy = (WaitForSingleObject(obj.busy,0) == WAIT_OBJECT_0);
	ready = (WaitForSingleObject(obj.ready,0) == WAIT_OBJECT_0);
	return (busy || !ready);
};

int woIsReady(waitobject obj)
{
	DWORD busy,ready;

	busy = (WaitForSingleObject(obj.busy,0) == WAIT_OBJECT_0);
	ready = (WaitForSingleObject(obj.ready,0) == WAIT_OBJECT_0);
	return (ready && !busy);
};

DWORD woWaitReady(waitobject obj,DWORD timeout)
{
	return WaitForSingleObject(obj.ready,timeout);
};

DWORD woWaitBusy(waitobject obj,DWORD timeout)
{
	return WaitForSingleObject(obj.busy,timeout);
};

//
// Samplebuffer handling
//
typedef struct
{
	int size;
	DWORD *inbuf;
	DWORD *outbuf;
} dworddoublebuffer;

void sbReset(dworddoublebuffer *buf)
{
	buf->inbuf = NULL;
	buf->outbuf = NULL;
	buf->size = 0;
}

void sbSetSize(dworddoublebuffer *buf,int size)
{
	if (buf->inbuf != NULL)
		free(buf->inbuf);
	if (buf->outbuf != NULL)
		free(buf->outbuf);
	if (size > 0)
	{
		buf->inbuf = (DWORD *)malloc(size*sizeof(DWORD));
		buf->outbuf = (DWORD *)malloc(size*sizeof(DWORD));
		buf->size = size;
	}
	else
	{
		buf->inbuf = NULL;
		buf->outbuf = NULL;
		buf->size = 0;
	};
};

void sbSwap(dworddoublebuffer *buf)
{
	DWORD *tmp;

	tmp = buf->inbuf;
	buf->inbuf = buf->outbuf;
	buf->outbuf = tmp;
};

//
// local variables structure
//
typedef struct
{
	PyObject *self;
	PyObject *shell;
	PyObject *conf;
	PyObject *log;
	PyObject *integrationDone;

	// Thread variables
	CRITICAL_SECTION lock;
	HANDLE doneTrigger;
	HANDLE integrationThread;
	HANDLE processThread;
	HANDLE integrationThreadDone;
	HANDLE processThreadDone;
	HANDLE endIntegration;
	HANDLE processReady,doProcess;
	HANDLE stopIntegrator,runIntegrator;
	//waitobject integrationState;
	//waitobject processState;

	// Mode variables
	tmodelist *modelist;

	// Integration variables
	DWORD errorCode;
	int npulsesint;
	int missedPulses;
	double synctime;
	DWORD intcount;
	double starttime;
	double endtime;
	DWORD startmsw;
	DWORD startlsw;
	DWORD endmsw;
	DWORD endlsw;
	int pulseToProcess;
	void *samplesToProcess;
	dworddoublebuffer *samplebuffer;

} localvars;

