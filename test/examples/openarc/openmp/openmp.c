#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#ifdef _OPENARC_
#include <openacc.h>
#endif

#ifndef _N_
#define _N_ 10000000;
#endif

int main() {
	int N = _N_;
	int *A, *B, *C;
	int i,j,k, is_sync;

	A = (int *)malloc(N*sizeof(int));
	B = (int *)malloc(N*sizeof(int));
	
	for(i=0; i<N; i++) {
		A[i] = i;
		B[i] = i;
	}
	
	#pragma omp parallel for num_threads(2) schedule(static) private(is_sync,i,j,k)
	for(i=0; i<omp_get_num_threads(); i++) {
		int thread_id = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		printf("OpenMP thread %d is running\n", thread_id);
		if( thread_id == 0 ) {
			#pragma acc parallel loop  gang worker copy(A[0:N]) async(thread_id)
			for(j=0; j<N; j++) {
				A[j] = A[j] + 1;
			}
		} else if( thread_id == 1 ) {
			#pragma acc parallel loop  gang worker copy(B[0:N]) async(thread_id)
			for(j=0; j<N; j++) {
				B[j] = B[j] + 1;
			}
		}
#ifdef _OPENACC
		is_sync = acc_async_test(thread_id);
		if( is_sync != 0 ) {
			printf("Thread %d fails async test1\n", thread_id);
		}

		#pragma acc wait(thread_id);
		printf("OpenMP thread %d finishes\n", thread_id);

		is_sync = acc_async_test(thread_id);
		if( is_sync == 0 ) {
			printf("Thread %d fails async test2\n", thread_id);
		}
#endif
	}
	for(i=0; i<N; i++) {
		if( A[i] != i+1 ) {
			printf("Error in array A computation!\n");
			exit(1);
		};
		if( B[i] != i+1 ) {
			printf("Error in array B computation!\n");
			exit(1);
		};
	}

	free(A);
	free(B);

	return 0;
}
