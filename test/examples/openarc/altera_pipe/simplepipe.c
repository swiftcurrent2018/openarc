#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#if OMP == 1
#include <omp.h>
#endif

#ifndef _N_
#define _N_ 8192000
#endif

#ifndef HOST_MEM_ALIGNMENT
#define HOST_MEM_ALIGNMENT 1
#endif

#if HOST_MEM_ALIGNMENT == 1
#define AOCL_ALIGNMENT 64
#endif

int N = _N_;

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}


void
simplepipe_acc(float * a, float * b, float * c)
{
  int i;
#pragma acc data copyout(a[0:N]), pipe(b[0:N]) copyin(c[0:N])
  {
#pragma acc kernels loop gang worker pipeout(b)
    for (i=0; i<N; i++){
	  b[i] = c[i]*c[i];
    }
#pragma acc kernels loop gang worker pipein(b)
    for (i=0; i<N; i++){
	  a[i] = b[i];
    }
  }
}


void
simplepipe_openmp(float * a,float * b, float * c)
{
  int i;
#pragma omp parallel shared(a,b) private(i)
  {
#pragma omp for
    for (i=0; i<N; i++){
	  b[i] = c[i]*c[i];
    }
#pragma omp for
    for (i=0; i<N; i++){
	  a[i] = b[i];
    }
  }
}


int main()
{
  float *a, *b, *c;
  float *ref_a, *ref_b, *ref_c;
  double cpu_sum = 0.0;
  double gpu_sum = 0.0;
  double rel_err = 0.0;

#if HOST_MEM_ALIGNMENT == 1
  void *p;
#endif
  int i;
  double elapsed_time;

#if HOST_MEM_ALIGNMENT == 1
  posix_memalign(&p, AOCL_ALIGNMENT, N*sizeof(float));
  a = (float *)p;
  posix_memalign(&p, AOCL_ALIGNMENT, N*sizeof(float));
  b = (float *)p;
  posix_memalign(&p, AOCL_ALIGNMENT, N*sizeof(float));
  c = (float *)p;
  posix_memalign(&p, AOCL_ALIGNMENT, N*sizeof(float));
  ref_a = (float *)p;
  posix_memalign(&p, AOCL_ALIGNMENT, N*sizeof(float));
  ref_b = (float *)p;
  posix_memalign(&p, AOCL_ALIGNMENT, N*sizeof(float));
  ref_c = (float *)p;
#else
  a = (float *) malloc(N*sizeof(float));
  b = (float *) malloc(N*sizeof(float));
  c = (float *) malloc(N*sizeof(float));
  ref_a = (float *) malloc(N*sizeof(float));
  ref_b = (float *) malloc(N*sizeof(float));
  ref_c = (float *) malloc(N*sizeof(float));
#endif

  for (i = 0; i <  N; i++) {
    c[i] = (float) i;
    ref_c[i] = (float) i;
  }

  elapsed_time = my_timer();
  simplepipe_openmp(ref_a,ref_b,ref_c);
  elapsed_time = my_timer() - elapsed_time;
  printf("CPU Elapsed time = %lf sec\n", elapsed_time);
  elapsed_time = my_timer();
  simplepipe_acc(a,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time = %lf sec\n", elapsed_time);


  for (i = 0; i < N; i++)
  {   
    cpu_sum += ref_a[i]*ref_a[i];
    gpu_sum += a[i]*a[i];
  }   

  cpu_sum = sqrt(cpu_sum);
  gpu_sum = sqrt(gpu_sum);
  if( cpu_sum > gpu_sum) {
    rel_err = (cpu_sum-gpu_sum)/cpu_sum;
  } else {
    rel_err = (gpu_sum-cpu_sum)/cpu_sum;
  }   
  if(rel_err < 1e-9)
  {   
    printf("Verification Successful err = %e\n", rel_err);
  }   
  else
  {   
    printf("Verification Fail err = %e\n", rel_err);
  }   

  free(a);
  free(b);
  free(c);
  free(ref_a);
  free(ref_b);
  free(ref_c);

  return 0;
} 

