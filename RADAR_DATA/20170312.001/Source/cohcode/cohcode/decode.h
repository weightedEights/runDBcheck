// Function to decode static codes
				  
void decode(int codelength, char *code, float *inbuf,float *outbuf,int ngates)
{
	int c;
	char s;
	
	for (c=0;c<2*ngates;c++)
		outbuf[c] = 0.0;
	for (c=0;c<codelength;c++)
	{
		s = code[c];
		if (s == '+')
			cvadd(outbuf,inbuf+2*c,outbuf,ngates);
		else
			cvsub(outbuf,inbuf+2*c,outbuf,ngates);
	}
}
