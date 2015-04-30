#include <stdio.h>
#include <omp.h>

#ifndef _M_SIZE
#define _M_SIZE 4194304
#endif

#ifdef _OPENARC_

#if _M_SIZE == 4096
#pragma openarc #define _M_SIZE 4096
#elif _M_SIZE == 65536
#pragma openarc #define _M_SIZE 65536
#elif _M_SIZE == 262144
#pragma openarc #define _M_SIZE 262144
#elif _M_SIZE == 4194304
#pragma openarc #define _M_SIZE 4194304
#endif

#endif

extern int omp_num_threads;

void lud_omp(float * a, int size)
{
     int i,j,k;
     float sum;
	 printf("num of threads = %d\n", omp_num_threads);
#pragma acc data copy(a[0:_M_SIZE])
     for (i=0; i <size; i++){
#pragma acc kernels loop gang, worker, private(j, k, sum)
         for (j=i; j <size; j++){
             sum=a[i*size+j];
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
         }

#pragma acc kernels loop if(i+1<size) gang, worker, private(j, k, sum)
         for (j=i+1;j<size; j++){
             sum=a[j*size+i];
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
         }
     }
}
