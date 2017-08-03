#ifndef __tracking 
#define __tracking

#include <Python.h>
#include <Windows.h>

enum tTrackState 
{
	tIdle=0,
	tInit=1,
	tRunning=2,
	tDone=3,
	tFinished=4
};

#define MAXTRACKS 256


typedef struct 
{
	enum tTrackState state;

	int nobjects;
	double starttime;
	double endtime;
	unsigned int *data;
} TRACK;

typedef struct
{
	int ntracks; //not used in new scheme
	CRITICAL_SECTION lock;

	int rdptr;
	int wrptr;

	TRACK **tracks;
} TRACKBLOCK;

#endif

//function prototypes
TRACKBLOCK *tbCreate();
TRACK *tbGetTrack(TRACKBLOCK *,double);
