#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
//#define _OPENACCM
#ifdef _OPENACCM
#include <openacc.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef _N_
#define _N_ 512
#endif

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

#ifndef TEST_BIND
#define TEST_BIND 1
#endif


int N = _N_;
int M = _N_;
int P = _N_;

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}

// TEST_BIND == 0: do not use the bind clause.
// TEST_BIND == 1: apply the bind clause for a fuction whose definition exists (global variables are implicitly accessed).
// TEST_BIND == 2: apply the bind clause for a fuction whose declaration exists (global variables are implicitly accessed).
// TEST_BIND == 3: apply the bind clause for a fuction whose declaration exists (global variables are explicitly passed as parameters).
// TEST_BIND == 4: apply the bind clause for a fuction whose declaration/definition does not exist (global variables are implicitly accessed).

// To bind an external library function, all global variables should be explicitly
// passed as function parameters, but the reference procedure below has implicit
// global variable accesses, and thus the "TEST_BIND == 2" case will not work.
// TEST_BIND = 0, 1, and 3 will work correctly. (Case 1 works because the compiler can modify the definition of the user function.)

#if TEST_BIND == 1
float my_reduction( float *a, float *b, int i, int j) {
	int m;
	float tsum = 0.0F;
	for (m=0; m<P; m++) {
		tsum += a[i*P+m]*b[m*N+j] ;
	}
	return tsum;
}
#elif TEST_BIND == 2
float my_reduction( float *a, float *b, int i, int j);
#elif TEST_BIND == 3
float my_reduction( float *a, float *b, int i, int j, int N, int P);
#endif

#if TEST_BIND == 3
#pragma acc routine seq bind(my_reduction)
float reduction( float *b, float *c, int i, int j, int N, int P) {
	float sum = 0.0F;
	int k;
#pragma acc loop seq
	for (k=0; k<P; k++) {
		sum += b[i*P+k]*c[k*N+j] ;
	}
	return sum;
}
#else
#if TEST_BIND != 0
#pragma acc routine seq bind(my_reduction)
#endif
float reduction( float *b, float *c, int i, int j) {
	float sum = 0.0F;
	int k;
#pragma acc loop seq
	for (k=0; k<P; k++) {
		sum += b[i*P+k]*c[k*N+j] ;
	}
	return sum;
}
#endif


void
MatrixMultiplication_openacc(float * a, float * b, float * c)
{
  int i, j, k ;

#ifdef _OPENACCM
  acc_init(acc_device_default);
#endif
#pragma acc kernels loop independent gang worker collapse(2) copyout(a[0:(M*N)]), copyin(b[0:(M*P)],c[0:(P*N)])
    for (i=0; i<M; i++){
      for (j=0; j<N; j++)
        {
#if TEST_BIND == 3
	  		a[i*N+j] = reduction(b,c,i,j,N,P);
#else
	  		a[i*N+j] = reduction(b,c,i,j);
#endif
        }
    }
#ifdef _OPENACCM
  acc_shutdown(acc_device_default);
#endif
}


void
MatrixMultiplication_openmp(float * a,float * b, float * c)
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
  float *a_CPU, *b_CPU, *c_CPU;
  int i,j;
  double elapsed_time;

  a = (float *) malloc(M*N*sizeof(float));
  b = (float *) malloc(M*P*sizeof(float));
  c = (float *) malloc(P*N*sizeof(float));
  a_CPU = (float *) malloc(M*N*sizeof(float));
  b_CPU = (float *) malloc(M*P*sizeof(float));
  c_CPU = (float *) malloc(P*N*sizeof(float));

  for (i = 0; i <  M*N; i++) {
    a[i] = (float) 0.0F;
    a_CPU[i] = (float) 0.0F;
  }
  for (i = 0; i <  M*P; i++) {
    b[i] = (float) i;
    b_CPU[i] = (float) i;
  }
  for (i = 0; i <  P*N; i++) {
    c[i] = (float) 1.0F;
    c_CPU[i] = (float) 1.0F;
  }

#if VERIFICATION == 1
  elapsed_time = my_timer();
  MatrixMultiplication_openmp(a_CPU,b_CPU,c_CPU);
  elapsed_time = my_timer() - elapsed_time;
  printf("CPU Elapsed time = %lf sec\n", elapsed_time);
#endif
  elapsed_time = my_timer();
  MatrixMultiplication_openacc(a,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time = %lf sec\n", elapsed_time);

#if VERIFICATION == 1
	{
		double cpu_sum = 0.0;
		double gpu_sum = 0.0;
    	double rel_err = 0.0;

    	for (i=0; i<M*N; i++){
			cpu_sum += a_CPU[i]*a_CPU[i];
			gpu_sum += a[i]*a[i];
		}

		cpu_sum = sqrt(cpu_sum);
		gpu_sum = sqrt(gpu_sum);
		if( cpu_sum > gpu_sum ) {
			rel_err = (cpu_sum-gpu_sum)/cpu_sum;
		} else {
			rel_err = (gpu_sum-cpu_sum)/cpu_sum;
		}

		if(rel_err < 1e-6)
		{
	    	printf("Verification Successful err = %e\n", rel_err);
		}
		else
		{
	    	printf("Verification Fail err = %e\n", rel_err);
		}
	}
#endif

  free(a_CPU);
  free(b_CPU);
  free(c_CPU);
  free(a);
  free(b);
  free(c);

  return 0;
} 

