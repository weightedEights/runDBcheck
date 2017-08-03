#include "tracking.h"

//
// track handling
//
TRACK *tCreate(PyObject *track)
{
	PyObject *pyst,*pyet;
	double st,et;
	int i,ncodes,nids;
	PyObject *pycodes,*pyids,*pycode,*pyid;
	unsigned int code,id;
	TRACK *ctrack;

	if (!PyDict_Check(track))
		return NULL;

	pyst = PyDict_GetItemString(track,"starttime");
	if (pyst == NULL)
		return NULL;
	st = PyFloat_AsDouble(pyst);

	pyet = PyDict_GetItemString(track,"endtime");
	if (pyet == NULL)
		return NULL;
	et = PyFloat_AsDouble(pyet);

	pycodes = PyDict_GetItemString(track,"codes");
	if (pycodes == NULL)
		return NULL;
	if (!PyList_Check(pycodes))
		return NULL;
	ncodes = PyList_Size(pycodes);

	pyids = PyDict_GetItemString(track,"ids");
	if (pyids == NULL)
		return 0;
	if (!PyList_Check(pyids))
		return NULL;
	nids = PyList_Size(pyids);

	if (!(ncodes==nids))
		return NULL;

	// allocate and fill the ctrack struct
	ctrack = (TRACK *)malloc(sizeof(TRACK));
	ctrack->state = tIdle;
	ctrack->starttime = st;
	ctrack->endtime = et;
	ctrack->nobjects = ncodes;
	ctrack->data = (unsigned int*)malloc(ncodes*sizeof(unsigned int));
	for (i=0;i<ncodes;i++)
	{
		pycode = PyList_GetItem(pycodes,i);
		pyid = PyList_GetItem(pyids,i);
		code = PyLong_AsLong(pycode);
		id = PyLong_AsLong(pyid);
		ctrack->data[i] = id*0x10000+code;
	}

	return ctrack;
}

int tFree(TRACK *ctrack)
{
	if (ctrack == NULL)
		return 1;

	free(ctrack->data);
	free(ctrack);
	return 1;
}


//
// track block handling
//

TRACKBLOCK *tbCreate()
{
	PyObject *track;
	TRACK *ctrack=NULL;
	int i,ntracks;
	TRACKBLOCK *trackblock;

	// allocate memory for table
	trackblock = (TRACKBLOCK *)malloc(sizeof(TRACKBLOCK));

	trackblock->tracks = malloc(MAXTRACKS*sizeof(void *));
	
	// reset track slots to unused
	for (i=0;i<MAXTRACKS;i++)
		trackblock->tracks[i] = NULL;

	trackblock->rdptr = 0;
	trackblock->wrptr = 0;

	InitializeCriticalSectionAndSpinCount(&trackblock->lock,2); //lock for track slots
	return trackblock;
}

int tbFree(TRACKBLOCK *trackblock)
{
	int i;

	if (trackblock == NULL)
		return 1;

	DeleteCriticalSection(&trackblock->lock);

	for (i=0;i<MAXTRACKS;i++)
		tFree(trackblock->tracks[i]);
	free(trackblock->tracks);
	free(trackblock);
	trackblock = NULL;
	return 1;
}


TRACK *tbGetTrack(TRACKBLOCK *trackblock,double radactime)
{
	TRACK *current;
	int notracks;

	notracks = 0;

	EnterCriticalSection(&trackblock->lock);
	current = trackblock->tracks[trackblock->rdptr];
	if (current == NULL)
		notracks = 1;
	LeaveCriticalSection(&trackblock->lock);

	if (notracks)
		return NULL;

	switch(current->state)
	{
		case tIdle:
			if (radactime >= current->endtime)
			{
				// too old just finish it
				current->state = tFinished;
				return current;
			}
			if (radactime >= current->starttime)
				current->state = tInit;
			return current;

		case tRunning:
			if (radactime >= current->endtime)
				current->state = tDone;
			return current;

		case tFinished:
			tFree(current);
			EnterCriticalSection(&trackblock->lock);
			trackblock->tracks[trackblock->rdptr] = NULL;
			LeaveCriticalSection(&trackblock->lock);

			trackblock->rdptr++;
			if (trackblock->rdptr >= MAXTRACKS)
				trackblock->rdptr = 0;

			return NULL;
		default:
			return current;
	}
	return current;
}

int tbAppend(TRACKBLOCK *trackblock,PyObject *info)
{
	PyObject *track;
	TRACK *ctrack=NULL;
	int i,ntracks,res;

	if (!PyList_Check(info))
		return 0;

	ntracks = PyList_Size(info);

	res = 1;  
	// now assemble and store tracks
	for (i=0;i<ntracks;i++)
	{
		track = PyList_GetItem(info,i);
		ctrack = tCreate(track);
		if (ctrack == NULL)
			continue;

		EnterCriticalSection(&trackblock->lock);
		if (trackblock->tracks[trackblock->wrptr] == NULL)
		{
			// only add if slot is not in use otherwise silently drop the track
			trackblock->tracks[trackblock->wrptr] = ctrack;
			trackblock->wrptr++;
			if (trackblock->wrptr >= MAXTRACKS)
				trackblock->wrptr = 0;
		}
		else
			res = 0; //not room for all tracks
		LeaveCriticalSection(&trackblock->lock);
	}

	return res;
}
