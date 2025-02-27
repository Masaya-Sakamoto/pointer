void *mymemset(void *dst, int c, long n)
{
	if (n != 0) {
		unsigned char *d = (unsigned char *)dst;

		do
			*d++ = (unsigned char)c;
		while (--n != 0);
	}
	return (dst);
}
