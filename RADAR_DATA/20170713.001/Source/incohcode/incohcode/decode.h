int decode(char *code,int nlag,int offset)
{
	return code[offset]*code[offset+nlag];
}



