#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef _RESOLUTION_
#define _RESOLUTION_ 5000
#endif
#ifndef _NB_ITER_
#define _NB_ITER_ 40000
#endif

// complexity :  nb_iters*resolution^2

 #pragma acc routine
float mdb(float x0, float y0) {

float temp;
float x=x0;
float y=y0;

#pragma acc loop seq
for(int i=0; i < _NB_ITER_; i++) {
    temp = x;
    x = x*x - y*y + x0;
    y = 2*temp*y + y0;
}

return x*x+y*y;
}

double my_timer ()
{
    struct timeval time;
    gettimeofday (&time, 0); 
    return time.tv_sec + time.tv_usec / 1000000.0;
}

int main() {

const int resolution = _RESOLUTION_;
const int incr = resolution/100;
float *res;
int i,j;
double strt_time, done_time;

res = (float *)malloc(sizeof(float)*resolution*resolution);

strt_time = my_timer ();

 #pragma acc data copyout(res[0:resolution*resolution])
 #pragma acc parallel loop 
for(i=0; i<resolution; i++) { 
 #pragma acc loop 
for(j=0; j<resolution; j++) 
{
float x = -2.0+(4.0/resolution)*i;
float y = -2.0+(4.0/resolution)*j;
res[i*resolution+j] = mdb(x,y);
}
}

done_time = my_timer ();

//print
printf("\n");
for(i=incr*22; i<(resolution-22*incr); i+=incr)
{
for(j=incr*5; j<resolution; j+=incr) {
 if (res[j*resolution+i] < 4) {
 printf("X");
 } else {
 printf(" ");
 }
}
printf("\n");
}

printf("\nResolution = %d\n", resolution);
printf("Number of iterations = %d\n", _NB_ITER_);
printf ("Accelerator Elapsed time = %lf sec\n", done_time - strt_time);

}
