#include <sys/time.h>
#include <netdb.h>
#include <stdio.h>

/*
double	timer_()
{
     int elapTicks;
     double elapMilli;
     
     elapTicks = clock() * CLK_TCK;      //start the timer  
     elapMilli = elapTicks/1000;     //milliseconds from Begin to End
     return elapMilli/1000.0;
}
*/

double	timer_()
{
	struct	timeval	time;

	gettimeofday(&time, 0);

	return time.tv_sec + time.tv_usec/1000000.0;
}
