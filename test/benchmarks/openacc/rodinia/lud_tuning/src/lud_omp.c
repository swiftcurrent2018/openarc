#include <stdio.h>

#ifndef _M_SIZE
#define _M_SIZE 4096
#ifdef _OPENARC_
#pragma openarc #define _M_SIZE 4096
#endif
#endif

#ifndef ENABLE_OPENACC
#define ENABLE_OPENACC 1
#endif

extern int omp_num_threads;

void lud_omp(float * a, int size)
{
     int i,j,k;
     float sum;
	 //printf("num of threads = %d\n", omp_num_threads);
#if ENABLE_OPENACC == 1
#pragma acc data copy(a[0:_M_SIZE])
#endif
     for (i=0; i <size; i++){
#if ENABLE_OPENACC == 1
#pragma acc kernels loop gang, worker, private(j, k, sum)
#endif
         for (j=i; j <size; j++){
             sum=a[i*size+j];
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
         }

#if ENABLE_OPENACC == 1
#pragma acc kernels loop if(i+1<size) gang, worker, private(j, k, sum)
#endif
         for (j=i+1;j<size; j++){
             sum=a[j*size+i];
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
         }
     }
}
