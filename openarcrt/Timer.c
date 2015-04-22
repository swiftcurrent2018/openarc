#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define OneM 1048576 
#define OneK 1024
#define MAX_SLEEP 8

//#define DEBUG_ON

int main(int argc, char **argv) {
#ifdef DEBUG_ON
	time_t t1;
	time_t tpoint1,tpoint2,tpoint3;
#endif
	int i,k,cnt;
	int max_mem_length;
	int mem_length;
	int* data_ptr;
	int sleep_time;
	float fraction;

#ifdef DEBUG_ON
	(void)time(&t1);
	tpoint2 = t1;
#endif

	if( argc != 2 )
	{
		fprintf(stdout, "====> Usage: \n");
		fprintf(stdout, "      Timer time_to_wait(sec) \n");
		exit(-1);
	}
	sleep_time = atoi(argv[1]);
	fprintf(stdout, "====> Wait for %d seconds \n",sleep_time);
	sleep(sleep_time);

#ifdef DEBUG_ON
	(void)time(&t1);
	tpoint3 = t1;
	fprintf(stdout, "====> Actual sleeping time is %d \n",(tpoint3-tpoint2));
#endif
}
