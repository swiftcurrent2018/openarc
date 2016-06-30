#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>
//#define _OPENACCM
#ifdef _OPENACCM
#include <openacc.h>
#endif

#ifdef __cplusplus
#define restrict __restrict__
#endif

#ifndef _N_
#define _N_ 8192
#endif


int N = _N_;
int M = N;
int P = N;

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}


void
MatrixMultiplication_openacc(float * restrict a,float * restrict b, float * restrict c)
{
  int i, j, k ;

#ifdef _OPENACCM
  acc_init(acc_device_default);
#endif
#pragma acc data copyout(a[0:(M*N)]), copyin(b[0:(M*P)],c[0:(P*N)])
  {
#pragma acc kernels loop independent gang, vector(8)
    for (i=0; i<M; i++){
#pragma acc loop gang, vector (64)
      for (j=0; j<N; j++)
        {
	  float sum = 0.0 ;
#pragma acc loop seq
	  for (k=0; k<P; k++) {
	    sum += b[i*P+k]*c[k*N+j] ;
      }
	  a[i*N+j] = sum ;
        }
    }
  }
#ifdef _OPENACCM
  acc_shutdown(acc_device_default);
#endif
}


void
MatrixMultiplication_openmp(float * restrict a,float * restrict b, float * restrict c)
{
  int i, j, k ;
  int chunk = N/4;


#pragma omp parallel shared(a,b,c,chunk) private(i,j,k)
  {
#ifdef _OPENMP
	if(omp_get_thread_num() == 0) {
		printf("Number of OpenMP threads %d\n", omp_get_num_threads());
	}
#endif
//#pragma omp for schedule(static,chunk)
//#pragma omp for schedule(dynamic)
#pragma omp for
    for (i=0; i<M; i++){
      for (j=0; j<N; j++)
        {
	  float sum = 0.0 ;
	  for (k=0; k<P; k++)
	    sum += b[i*P+k]*c[k*N+j] ;
	  a[i*N+j] = sum ;
        }
    }
  }
}


int main()
{
  float *a, *b, *c;
  int i;
  double elapsed_time;

  a = (float *) malloc(M*N*4);
  b = (float *) malloc(M*P*4);
  c = (float *) malloc(P*N*4);

  for (i = 0; i <  M*N; i++) {
    a[i] = (float) 0.0;
  }
  for (i = 0; i <  M*P; i++) {
    b[i] = (float) i;
  }
  for (i = 0; i <  P*N; i++) {
    c[i] = (float) 1.0;
  }

  elapsed_time = my_timer();
  MatrixMultiplication_openmp(a,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("CPU Elapsed time = %lf sec\n", elapsed_time);
  elapsed_time = my_timer();
#pragma aspen modelregion
  MatrixMultiplication_openacc(a,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time = %lf sec\n", elapsed_time);

  free(a);
  free(b);
  free(c);

  return 0;
} 

