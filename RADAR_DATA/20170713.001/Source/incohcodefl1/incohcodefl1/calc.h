
// Reduce number of lags in output acfs by adding together
// lags with the same lag time
void reduceLags(PyObject *ain, PyObject *amat, PyObject *aout)
{
	float *pin,*pout;
	int *pmat;
	int npos,nlags,ngates,elags;
	int ips,ops,ls,gs;
	int pos,lag,gate,olag;

	pin = (float *)PyArray_DATA(ain);
	pout = (float *)PyArray_DATA(aout);
	pmat = (int *)PyArray_DATA(amat);

	npos = PyArray_DIM(ain,0); //Get number of positions
	nlags = PyArray_DIM(ain,1); //Get number of lags in full array
	ngates = PyArray_DIM(ain,2); //number of gates
	elags = PyArray_DIM(aout,1); //Get number of lags in reduced array

	ips = nlags*ngates*2;
	ops = elags*ngates*2;
	ls = ngates*2;
	gs = 2;

	for (pos=0;pos<npos;pos++)
	{
		for (lag=0;lag<nlags;lag++)
		{
			olag = pmat[lag];
			for (gate=0;gate<ngates;gate++)
			{
				pout[pos*ops+olag*ls+gate*gs]   += pin[pos*ips+lag*ls+gate*gs];
			    pout[pos*ops+olag*ls+gate*gs+1] += pin[pos*ips+lag*ls+gate*gs+1];
			}
		}
	}
}