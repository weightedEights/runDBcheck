#include "radacDev.h"

// Static structure holding RADAC specific information
// populated when module is loaded
static RADACINFO radacInfo;


//
// Get handle to RADAC device driver
//
HANDLE devOpenfile()
{
	return CreateFile(DEVICEFILENAME,
		              GENERIC_WRITE | GENERIC_READ,
		              FILE_SHARE_WRITE | FILE_SHARE_READ,NULL,
					  OPEN_EXISTING,
					  FILE_ATTRIBUTE_NORMAL,0);
};

//
// devRead reads count number of 32 bit registers from mem starting at
// address addr into buffer.
//
DWORD devRead(HANDLE dev,ULONG mem,ULONG addr,ULONG count,ULONG *buffer)
{
	DEVTRANSFER ti;
	DWORD nBytesReturned;
	DWORD err;
	char mbuf[MESSAGEBUFFERSIZE],errmsg[MESSAGEBUFFERSIZE];

	ti.Memory = mem;
	ti.Access = DEVICEREAD;
	ti.Address = addr;
	ti.Length = count;
	if (!DeviceIoControl(dev,IOCTL_TRANSFER,&ti,sizeof(ti),buffer,count*sizeof(ULONG),&nBytesReturned,NULL))
		return 0;

	return 1;
};

//
// Write count number of 32 bit values from buffer to mem starting at address addr
//
DWORD devWrite(HANDLE dev,ULONG mem,ULONG addr,ULONG count,ULONG *buffer)
{
	pDEVTRANSFER pti;
	DWORD nBytesReturned;
    int i,nbytes;

	nbytes = sizeof(DEVTRANSFER)+count*sizeof(ULONG);
	pti = (pDEVTRANSFER)malloc(nbytes);
	pti->Memory = mem;
	pti->Access = DEVICEWRITE;
	pti->Address = addr;
	pti->Length = count;
	for (i=0;i<count;i++)
		pti->Data[i] = buffer[i];

	if (!DeviceIoControl(dev,IOCTL_TRANSFER,pti,nbytes,NULL,0,&nBytesReturned,NULL))
	{
		free(pti);
		return 0;
	}
	free(pti);
	return 1;
};

//
// devReadTx reads 7 32 bit registers from FPGA txconfig mem into buffer
//
DWORD devReadTx(HANDLE dev,ULONG *buffer)
{
	XFERTX ti;
	DWORD nBytesReturned;

	ti.Write = FALSE;
	ti.Length = 7;
	if (!DeviceIoControl(dev,IOCTL_TX_CONFIG,&ti,sizeof(ti),buffer,ti.Length*sizeof(ULONG),&nBytesReturned,NULL))
		return 0;

	return 1;
};


//
// devWriteTx writes 7 32 bit values from buffer to FPGA txconfig mem
//
DWORD devWriteTx(HANDLE dev,ULONG *buffer)
{
	XFERTX ti;
	DWORD nBytesReturned;
	int i;

	ti.Write = TRUE;
	ti.Length = 7;
	for (i=0;i<ti.Length;i++)
		ti.Data[i] = buffer[i];
	if (!DeviceIoControl(dev,IOCTL_TX_CONFIG,&ti,sizeof(ti),NULL,0,&nBytesReturned,NULL))
		return 0;

	return 1;
};

//
// devReadRx reads value from Rx from addr into buffer
//
DWORD devReadRx(HANDLE dev,ULONG addr,ULONG *buffer)
{
	XFERRX ti;
	DWORD nBytesReturned;
	
	ti.Write = FALSE;
	ti.Length = 2;
	ti.Data[0] = 0x00000800 | (addr & 0x000003ff);
	ti.Data[1] = 0;
	
	if (!DeviceIoControl(dev,IOCTL_RX_CONFIG,&ti,sizeof(ti),buffer,ti.Length*sizeof(ULONG),&nBytesReturned,NULL))
		return 0;
		
	if (addr <= 0xff)
	{
		buffer[0] = buffer[0] & 0xFFFFF;  //20 bits
		buffer[1] = 0;
	}
	else if ((addr >= 0x100) && (addr <= 0x1ff))  //36 bits
	{
		buffer[1] = buffer[1] & 0x0000000f; 
	}
	else if (addr == 0x300)  //8 bits
	{
		buffer[0] = buffer[0] & 0x000000ff; 
		buffer[1] = 0;
	}
	else if (addr == 0x301)  //3 bits
	{
		buffer[0] = buffer[0] & 0x00000007;  
		buffer[1] = 0;
	}
	else if ((addr == 0x302) || (addr == 0x303))  //32 bits
	{
		buffer[1] = 0;                    
	}
	else if (addr == 0x304)  //16 bits
	{
		buffer[0] = buffer[0] & 0x0000ffff;  
		buffer[1] = 0;
	}
	else if ((addr == 0x305) || (addr == 0x306))  //8 bits
	{
		buffer[0] = buffer[0] & 0x000000ff;  
		buffer[1] = 0;
	}
	else if (addr == 0x307)  //5 bits
	{
		buffer[0] = buffer[0] & 0x0000001f;  
		buffer[1] = 0;
	}
	else if (addr == 0x308)  //8 bits
	{
		buffer[0] = buffer[0] & 0x000000ff;
		buffer[1] = 0;
	}
	else if (addr == 0x309)  //4 bits
	{
		buffer[0] = buffer[0] & 0x0000000f;
		buffer[1] = 0;
	}
	else if ((addr >= 0x30a) && (addr <= 0x30d))  //8 bits
	{
		buffer[0] = buffer[0] & 0x000000ff;
		buffer[1] = 0;
	}
	return 1;
};

//
// devWriteRx writes buffer to Rx addr
//
DWORD devWriteRx(HANDLE dev,ULONG addr,ULONG *buffer)
{
	XFERRX ti;
	DWORD nBytesReturned;

	ti.Write = TRUE;
	ti.Length = 3;
	ti.Data[0] = buffer[0];
	ti.Data[1] = buffer[1];
	ti.Data[2] = 0x00000c00 | (addr & 0x000003ff);


	if (!DeviceIoControl(dev,IOCTL_RX_CONFIG,&ti,sizeof(ti),NULL,0,&nBytesReturned,NULL))
		return 0;

	return 1;
};

//
// devCollectSamples collect count samples from Rx
//
DWORD devCollectSamples(HANDLE dev,ULONG count,ULONG *buffer)
{
	DMATRANSFER dt;
	DWORD nBytesReturned;
	DWORD err;
	char mbuf[MESSAGEBUFFERSIZE],errmsg[MESSAGEBUFFERSIZE];
	BOOL res;

	dt.Length = count;

	res = DeviceIoControl(dev,IOCTL_COLLECT_SAMPLES,&dt,sizeof(dt),buffer,count*sizeof(ULONG),&nBytesReturned,NULL); 
	if (!res)
		return 0;

	return 1;
};




//Exported funtions


//
// readRegisters reads count number of 32 bit values from mem
//
DWORD readRegisters(DWORD mem,DWORD addr,DWORD count,DWORD *outbuf)
{
	ULONG res,adr,buffer[BUFFERSIZE];
	long full,rest,indx,i,j;
	HANDLE dev;

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
		goto error;

	full = count / BUFFERSIZE;
	rest = count % BUFFERSIZE;
	adr = addr;
	indx = 0;
	for (i=0;i<full;i++)
	{
		if (!devRead(dev,mem,adr,BUFFERSIZE,buffer)) 
			goto error;

		for (j=0;j<BUFFERSIZE;j++)
		{
			outbuf[indx] = buffer[j];
			indx += 1;
		}
		adr += BUFFERSIZE;
	};
	if (rest > 0)
	{
		if (!devRead(dev,mem,adr,rest,buffer))
			goto error;

		for (j=0;j<rest;j++)
		{
			outbuf[indx] = buffer[j];
			indx += 1;
		};
	};

cleanup:
	CloseHandle(dev);
	return 0;
error:
	CloseHandle(dev);
	return GetLastError();
}


//
// writeRegisters writes count words to
// the memory specified by mem starting at address addr
//
DWORD writeRegisters(DWORD mem,DWORD addr,DWORD count,DWORD *inbuf)
{
	ULONG res,adr,buffer[BUFFERSIZE];
	long full,rest,i,j,indx;
	HANDLE dev;
	
	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
		goto error;

	full = count / BUFFERSIZE;
	rest = count % BUFFERSIZE;
	adr = addr;
	indx = 0;
	for (i=0;i<full;i++)
	{
		for (j=0;j<BUFFERSIZE;j++)
		{
			buffer[j] = inbuf[indx];
			indx++;
		}

		if (!devWrite(dev,mem,adr,BUFFERSIZE,buffer)) 
			goto error;

		adr += BUFFERSIZE;
	};
	if (rest > 0)
	{
		for (j=0;j<rest;j++)
		{
			buffer[j] = inbuf[indx];
			indx++;
		};

		if (!devWrite(dev,mem,adr,rest,buffer))
			goto error;

	};

cleanup:
	CloseHandle(dev);
	return 0;
error:
	CloseHandle(dev);
	return GetLastError();
}

//
// device_readTx reads 7 numbers of 32 bit values from FPGA
// txconfig mem 
//
DWORD readTx(DWORD *buffer)
{
	ULONG i;
	HANDLE dev;

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
		goto error;

	if (!devReadTx(dev,buffer)) 
		goto error;

cleanup:
	CloseHandle(dev);
	return 0;
error:
	CloseHandle(dev);
	return GetLastError();
}

//
// device_writeTx writes 7 32 bit to
// the FPGA tx config memory.
//
DWORD writeTx(DWORD *buffer)
{
	char mbuf[MESSAGEBUFFERSIZE];
	long i,count;
	HANDLE dev;

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
		goto error;

	if (!devWriteTx(dev,buffer)) 
		goto error;

cleanup:
	CloseHandle(dev);
	return 0;
error:
	CloseHandle(dev);
	return GetLastError();
}


//
// readRx value(s) from Rx chip at addr into buffer
//
DWORD readRx(DWORD addr,DWORD *buffer)
{
	ULONG i;
	HANDLE dev;

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
		goto error;

	if (!devReadRx(dev,addr,buffer)) 
		goto error;

cleanup:
	CloseHandle(dev);
	return 0;
error:
	CloseHandle(dev);
	return GetLastError();
}

//
// device_writeRx writes values from buffer to
// the Rx chip at addr.
//
DWORD writeRx(DWORD addr,DWORD *buffer)
{
	long i,count;
	HANDLE dev;

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
		goto error;

	if (!devWriteRx(dev,addr,buffer)) 
		goto error;

cleanup:
	CloseHandle(dev);
	return 0;
error:
	CloseHandle(dev);
	return GetLastError();
}

DWORD collectSamples(DWORD count,DWORD *buffer)
{
	HANDLE dev;

	// Collect the samples
	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
		goto error;

	if (!devCollectSamples(dev,count,buffer)) 
		goto error;

cleanup:
	CloseHandle(dev);
	return 0;
error:
	CloseHandle(dev);
	return GetLastError();
};

/**********************************************************************************/
/* Python section																  */
/**********************************************************************************/

//
// Python helper functions
//
int convertSamplesToComplex(ULONG *buffer,PyArrayObject *header,PyArrayObject *samplebuffer)
{
	ULONG bufindx,i,j,size;
	npy_cfloat *data;
	ULONG *hdrv;

	// Extract RADAC header values to numpy uint32 array
	bufindx = 0;
	size = PyArray_SIZE(header);
	hdrv = (ULONG *) PyArray_DATA(header);
	for (i=0;i<size;i++)
		hdrv[i] = buffer[i];

	bufindx = i;
	size = PyArray_SIZE(samplebuffer);
	data = (npy_cfloat *) PyArray_DATA(samplebuffer);
	for (j=0;j<size;j++)
	{
		data[j].real = (short)(buffer[bufindx] >> 16);
		data[j].imag = (short)(buffer[bufindx] & 0x0000ffff);
		bufindx++;
	};

	return 1;
};


// Various low level RADAC device driver access routines + helper functions

//
// Translate a win32 error message to a string
//
int TranslateError(DWORD code,char *ret,int bufsize)
{
	return FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM,NULL,code,0,ret,bufsize,NULL);
};

//
//Python exported funtions
//

//
// device_readRegisters reads count number of 32 bit values from mem
// returning the values in a python list of python long(s)
//
static PyObject *device_readRegisters(PyObject *self, PyObject *args)
{
	ULONG res,mem,addr,adr,count,buffer[BUFFERSIZE];
	long full,rest,indx,i,j;
	HANDLE dev;

	PyObject *list;	
	
	if (!PyArg_ParseTuple(args, "III", &mem,&addr,&count))
	{
		PyErr_SetString(PyExc_TypeError,"deviceReadRegisters: Usage: deviceReadRegisters(memory,address,count).");
		return NULL;
	};

	list = PyList_New(count);

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
	{
		PyErr_SetString(PyExc_IOError,"Invalid device handle");
		goto error;
	}

	full = count / BUFFERSIZE;
	rest = count % BUFFERSIZE;
	adr = addr;
	indx = 0;
	for (i=0;i<full;i++)
	{
		if (!devRead(dev,mem,adr,BUFFERSIZE,buffer)) 
		{
			PyErr_SetString(PyExc_RuntimeError,"deviceReadRegisters: devRead failed");
			goto error;
		};

		for (j=0;j<BUFFERSIZE;j++)
		{
			PyList_SetItem(list,indx,PyLong_FromUnsignedLong(buffer[j]));
			indx += 1;
		}
		adr += BUFFERSIZE;
	};
	if (rest > 0)
	{
		if (!devRead(dev,mem,adr,rest,buffer))
		{
			PyErr_SetString(PyExc_RuntimeError,"deviceReadRegisters: devRead failed");
			goto error;
		};

		for (j=0;j<rest;j++)
		{
			PyList_SetItem(list,indx,PyLong_FromUnsignedLong(buffer[j]));
			indx += 1;
		};
	};

cleanup:
	CloseHandle(dev);
	return list;
error:
	Py_XDECREF(list);  //Make sure list object is removed
	CloseHandle(dev);
	return NULL;
}


//
// device_writeRegisters writes a python list of python long(s) to
// the memory specified by mem starting at address addr
//
static PyObject *device_writeRegisters(PyObject *self, PyObject *args)
{
	ULONG res,mem,addr,adr,count,buffer[BUFFERSIZE];
	long full,rest,i,j,indx;
	HANDLE dev;

	PyObject *list=NULL,*item=NULL;	
	
	if (!PyArg_ParseTuple(args, "IIO", &mem,&addr,&list))
	{
		PyErr_SetString(PyExc_TypeError,"deviceWriteRegister: Usage: deviceWriteRegister(memory,address,[...]).");
		return NULL;
	};

	count = PyList_Size(list);

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
	{
		PyErr_SetString(PyExc_IOError,"Invalid device handle");
		return NULL;
	}

	full = count / BUFFERSIZE;
	rest = count % BUFFERSIZE;
	adr = addr;
	indx = 0;
	for (i=0;i<full;i++)
	{
		for (j=0;j<BUFFERSIZE;j++)
		{
			item = PyList_GetItem(list,indx);
			buffer[j] = PyLong_AsUnsignedLong(item);
			indx++;
		}

		if (!devWrite(dev,mem,adr,BUFFERSIZE,buffer)) 
		{
			PyErr_SetString(PyExc_RuntimeError,"deviceWriteRegisters: devWrite failed");
			goto error;
		};

		adr += BUFFERSIZE;
	};
	if (rest > 0)
	{
		for (j=0;j<rest;j++)
		{
			item = PyList_GetItem(list,indx);
			buffer[j] = PyLong_AsUnsignedLong(item);
			indx++;
		};

		if (!devWrite(dev,mem,adr,rest,buffer))
		{
			PyErr_SetString(PyExc_RuntimeError,"deviceWriteRegisters: devWrite failed");
			goto error;
		};

	};

cleanup:
	CloseHandle(dev);
	Py_RETURN_NONE;
error:
	CloseHandle(dev);
	return NULL;
}

//
// device_readTx reads 7 numbers of 32 bit values from FPGA
// txconfig mem returning the values in a python list of python long(s)
//
static PyObject *device_readTx(PyObject *self, PyObject *args)
{
	ULONG i,buffer[TXCONFIGSIZE];
	HANDLE dev;

	PyObject *list;	
	
	if (!PyArg_ParseTuple(args, ""))
	{
		PyErr_SetString(PyExc_TypeError,"deviceReadTx: Usage: deviceReadTx().");
		return NULL;
	};

	list = PyList_New(TXCONFIGSIZE);

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
	{
		PyErr_SetString(PyExc_IOError,"Invalid device handle");
		goto error;
	}

	if (!devReadTx(dev,buffer)) 
	{
		PyErr_SetString(PyExc_RuntimeError,"deviceReadTx: devReadTx failed");
		goto error;
	};

	for (i=0;i<TXCONFIGSIZE;i++)
		PyList_SetItem(list,i,PyLong_FromUnsignedLong(buffer[i]));

cleanup:
	CloseHandle(dev);
	return list;
error:
	Py_XDECREF(list);  //Make sure list object is removed
	CloseHandle(dev);
	return NULL;
}

//
// device_writeTx writes 7 32 bit values from python list of python long(s) to
// the FPGA tx config memory.
//
static PyObject *device_writeTx(PyObject *self, PyObject *args)
{
	ULONG buffer[TXCONFIGSIZE];
	char mbuf[MESSAGEBUFFERSIZE];
	long i,count;
	HANDLE dev;

	PyObject *list=NULL,*item=NULL;	
	
	if (!PyArg_ParseTuple(args, "O",&list))
	{
		PyErr_SetString(PyExc_TypeError,"deviceWriteTx: Usage: deviceWriteTx([...]).");
		return NULL;
	};

	count = PyList_Size(list);
	if (count != TXCONFIGSIZE)
	{
		_snprintf_s(mbuf,MESSAGEBUFFERSIZE,MESSAGEBUFFERSIZE,"deviceWriteTx: Needs a list of %i longs.",TXCONFIGSIZE);
		PyErr_SetString(PyExc_ValueError,mbuf);
		return NULL;
	};

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
	{
		PyErr_SetString(PyExc_IOError,"Invalid device handle");
		return NULL;
	}

	for (i=0;i<TXCONFIGSIZE;i++)
	{
		item = PyList_GetItem(list,i);
		buffer[i] = PyLong_AsUnsignedLong(item);
	}

	if (!devWriteTx(dev,buffer)) 
	{
		PyErr_SetString(PyExc_RuntimeError,"deviceWriteTx: devWriteTx failed");
		goto error;
	};


cleanup:
	CloseHandle(dev);
	Py_RETURN_NONE;
error:
	CloseHandle(dev);
	return NULL;
}


//
// device_readRx value(s) from Rx chip at addr into a python list of python longs
//
static PyObject *device_readRx(PyObject *self, PyObject *args)
{
	ULONG i,buffer[RXBUFFERSIZE];
	HANDLE dev;
	ULONG addr;

	PyObject *list;	
	
	if (!PyArg_ParseTuple(args, "I",&addr))
	{
		PyErr_SetString(PyExc_TypeError,"deviceReadRx: Usage: deviceReadRx(addr).");
		return NULL;
	};

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
	{
		PyErr_SetString(PyExc_IOError,"Invalid device handle");
		goto error;
	}

	list = PyList_New(RXBUFFERSIZE);
	if (!devReadRx(dev,addr,buffer)) 
	{
		PyErr_SetString(PyExc_RuntimeError,"deviceReadRx: devReadRx failed");
		goto error;
	};

	for (i=0;i<RXBUFFERSIZE;i++)
		PyList_SetItem(list,i,PyLong_FromUnsignedLong(buffer[i]));

cleanup:
	CloseHandle(dev);
	return list;
error:
	Py_XDECREF(list);  //Make sure list object is removed
	CloseHandle(dev);
	return NULL;
}

//
// device_writeRx writes values from python list of python long(s) to
// the Rx chip at addr.
//
static PyObject *device_writeRx(PyObject *self, PyObject *args)
{
	ULONG buffer[RXBUFFERSIZE];
	char mbuf[MESSAGEBUFFERSIZE];
	long i,count;
	HANDLE dev;
	ULONG addr;

	PyObject *list=NULL,*item=NULL;	
	
	if (!PyArg_ParseTuple(args, "IO",&addr,&list))
	{
		PyErr_SetString(PyExc_TypeError,"deviceWriteRx: Usage: deviceWriteRx(addr,[..])");
		return NULL;
	};

	count = PyList_Size(list);
	if (count != RXBUFFERSIZE)
	{
		_snprintf_s(mbuf,MESSAGEBUFFERSIZE,MESSAGEBUFFERSIZE,"deviceWriteRx: Needs a list of %i longs",RXBUFFERSIZE);
		PyErr_SetString(PyExc_ValueError,mbuf);
		return NULL;
	};

	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
	{
		PyErr_SetString(PyExc_IOError,"Invalid device handle");
		return NULL;
	}

	for (i=0;i<RXBUFFERSIZE;i++)
	{
		item = PyList_GetItem(list,i);
		buffer[i] = PyLong_AsUnsignedLong(item);
	}

	if (!devWriteRx(dev,addr,buffer)) 
	{
		PyErr_SetString(PyExc_RuntimeError,"deviceWriteRx: devWriteRx failed");
		goto error;
	};

cleanup:
	CloseHandle(dev);
	Py_RETURN_NONE;
error:
	CloseHandle(dev);
	return NULL;
}

static PyObject *device_collectSamples(PyObject *self, PyObject *args)
{
	PyArrayObject *header,*samplebuffer;
	ULONG count,bytecount,*buffer,i,j;
	HANDLE dev;

	if (!PyArg_ParseTuple(args, "O!O!",&PyArray_Type,&header,&PyArray_Type,&samplebuffer))
	{
		PyErr_SetString(PyExc_TypeError,"deviceCollectSamples: Usage: deviceCollectSamples(<uint32 numpy header array>,<complex float numpy samplebuffer>)");
		return NULL;
	};

	if (header->descr->type_num != PyArray_UINT32) 
	{
		PyErr_SetString(PyExc_ValueError,"deviceCollectSamples needs a <uint32 numpy array> as argument 1 to return header values in.");
		return NULL;
	};
    if (samplebuffer->descr->type_num != PyArray_CFLOAT) 
	{
		PyErr_SetString(PyExc_ValueError,"deviceCollectSamples needs a <complex float numpy array> as argument 2 to return samples in.");
		return NULL;
	};

	// Create ULONG buffer to hold samples. Make sure to
	// make room for radac header words
	count = PyArray_SIZE(header)+PyArray_SIZE(samplebuffer);
	bytecount = count*sizeof(ULONG);
	buffer = (ULONG *) malloc(bytecount);

	// Collect the samples
	dev = devOpenfile();
	if (dev == INVALID_HANDLE_VALUE)
	{
		PyErr_SetString(PyExc_IOError,"Invalid device handle");
		goto error;
	}

	if (!devCollectSamples(dev,count,buffer)) 
	{
		PyErr_SetString(PyExc_RuntimeError,"deviceCollectSamples: devCollectSamples failed");
		goto error;
	};

	// Extract radac header values and convert samples to complex float array
	if (!convertSamplesToComplex(buffer,header,samplebuffer))
		goto error;

cleanup:
	free(buffer);
	CloseHandle(dev);
	Py_RETURN_NONE;
error:
	free(buffer);
	CloseHandle(dev);
	return NULL;
};

static PyObject *device_Info(PyObject *self, PyObject *args)
{
	PyObject *info;
	ULONG buffer[15];
	HANDLE dev;

	if (!PyArg_ParseTuple(args, ""))
	{
		PyErr_SetString(PyExc_TypeError,"deviceInfo: Usage: {<device info>} = deviceInfo()");
		return NULL;
	};
	dev = devOpenfile();
	devRead(dev,MEMREG,rcControl,15,buffer);
	CloseHandle(dev);

	return Py_BuildValue("{s:i,s:i,s:i}","versiondate",buffer[6],"versionnumber",buffer[7],
										"radacheaderwords",buffer[4]);
};


    

// Python Initialization code

//
// Function called from module initialization routine to
// populate the static radacInfo structure
//

void radacInit()
{
	ULONG buffer[2];
	HANDLE dev;

	dev = devOpenfile();
	devRead(dev,MEMREG,rcVerDate,2,buffer);
	radacInfo.VersionDate = buffer[0];
	radacInfo.VersionNumber = buffer[1];

	devRead(dev,MEMREG,rcNHeaderWords,1,&radacInfo.RadacHeaderWords);
	CloseHandle(dev);
};

static PyMethodDef deviceMethods[] = 
{
	{"deviceReadRegisters", device_readRegisters, METH_VARARGS, "deviceReadRegisters(mem,addr,count). Reads count words from mem to list"},
	{"deviceWriteRegisters", device_writeRegisters, METH_VARARGS, "deviceWriteRegisters(mem,addr,[...]). Writes [...] words to mem starting at address addr"},
	{"deviceReadTx", device_readTx, METH_VARARGS,"deviceReadTx(). Returning Tx config memory in list"},
	{"deviceWriteTx", device_writeTx, METH_VARARGS, "deviceWriteTx([...]). Writes [...] to Tx config memory"},
	{"deviceReadRx", device_readRx, METH_VARARGS, "deviceReadRx(address). Returning values in list"},
	{"deviceWriteRx", device_writeRx, METH_VARARGS, "deviceWriteRx(address,[...]). Writes [...] to Rx at address"},
	{"deviceCollectSamples", device_collectSamples, METH_VARARGS, "deviceCollectSamples(<complex float numpy samplebuffer>)"},
	{"deviceInfo", device_Info, METH_VARARGS, "deviceInfo(). Returning info about RADAC device in python dictionary"},
	{NULL, NULL, 0, NULL} /* Sentinel */
};


PyMODINIT_FUNC
initradacDev(void)
{
	PyObject *m;

	m = Py_InitModule("radacDev", deviceMethods);
	if (m == NULL)
		return;
	import_array();
//	radacInit();
};
