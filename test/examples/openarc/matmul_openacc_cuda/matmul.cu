#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include "openacc.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "acc_helper.h"

#ifndef _N_
#define _N_ 512
#endif

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

__global__ void MatrixMultiplication_cuda (float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int P);

static int N = _N_;
static int M = _N_;
static int P = _N_;

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}

void
MatrixMultiplication_openmp(float * a,float * b, float * c)
{
  int i, j, k ;


#pragma omp parallel shared(a,b,c) private(i,j,k)
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
  float *a_GPU, *b_GPU, *c_GPU;
  int i;
  double elapsed_time;
  dim3 dG, dB;

  //If below function is enabled, CUDA driver API creats a CUDA
  //context fist.
  //Otherwise, CUDA runtime API creates a CUDA context first. 
  acc_helper_setup();

  a = (float *) malloc(M*N*sizeof(float));
  b = (float *) malloc(M*P*sizeof(float));
  c = (float *) malloc(P*N*sizeof(float));
  a_CPU = (float *) malloc(M*N*sizeof(float));
  b_CPU = (float *) malloc(M*P*sizeof(float));
  c_CPU = (float *) malloc(P*N*sizeof(float));
  cudaMalloc((void **)&a_GPU, M*N*sizeof(float));
  cudaMalloc((void **)&b_GPU, M*P*sizeof(float));
  cudaMalloc((void **)&c_GPU, P*N*sizeof(float));

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
  //Enable below if this is not called previously.
  //acc_helper_setup();
  cudaMemcpy(b_GPU, b, M*P*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_GPU, c, P*N*sizeof(float), cudaMemcpyHostToDevice);
  dG.x = (int)ceil((((float)(M*N))/32.0F));
  dG.y = 1;
  dG.z = 1;
  dB.x = 32;
  dB.y = 1; 
  dB.z = 1;
  MatrixMultiplication_cuda<<<dG,dB>>>(a_GPU,b_GPU,c_GPU, M, N, P);
  cudaMemcpy(a, a_GPU, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time (CUDA) = %lf sec\n", elapsed_time);

  elapsed_time = my_timer();
  //Enable below if this is not called previously.
  //acc_helper_setup();
  cudaMemcpy(b_GPU, b, M*P*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_GPU, c, P*N*sizeof(float), cudaMemcpyHostToDevice);
  acc_map_data(a, a_GPU, M*N*sizeof(float));
  acc_map_data(b, b_GPU, M*P*sizeof(float));
  acc_map_data(c, c_GPU, P*N*sizeof(float));
  MatrixMultiplication_openacc(a,b,c);
  cudaMemcpy(a, a_GPU, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time (OpenACC) = %lf sec\n", elapsed_time);

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

