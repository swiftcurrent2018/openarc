#include "openacc.h"

#ifndef _N_
#define _N_ 512
#endif

static int N = _N_;
static int M = _N_;
static int P = _N_;

void acc_helper_setup() {
	acc_init(acc_device_default);
}

void MatrixMultiplication_openacc(float * a, float * b, float * c)
{
  int i, j, k ; 

#pragma acc kernels loop independent gang worker collapse(2) present(a[0:(M*N)]), present(b[0:(M*P)],c[0:(P*N)])
    for (i=0; i<M; i++){
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

