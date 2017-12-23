#include <stdio.h>
#include <stdlib.h>

#ifndef WORKERS 
#define WORKERS 256
#pragma openarc #define WORKERS 256
#endif

#ifndef GANGS 
#define GANGS 16
#pragma openarc #define GANGS 16
#endif

#define MSIZE 4096
#pragma openarc #define MSIZE 4096

int main(int argc, char** argv) {
    int size;
    float *A, *B, *C;
	float D = 0.0F;
	float D_ref = 0.0F;
    int i;
    int error = 0;

    size = MSIZE;

    A = (float*) malloc(size * sizeof(float));
    B = (float*) malloc(size * sizeof(float));
    C = (float*) malloc(size * sizeof(float));

    for (i = 0; i < size; i++) {
        A[i] = (float) i;
        B[i] = (float) i * 100;
        C[i] = 1.0F;
		D_ref += (float)i;
    }

#pragma acc data copyin(A[0:size], B[0:size])
    {
#pragma acc kernels loop independent gang(GANGS) worker(WORKERS) reduction(+:C[0:MSIZE], D)
        for (i = 0; i < size; i++) {
            C[i] += A[i] + B[i];
			D += (float)i;
        }
    }

    for (i = 0; i < size; i++) {
        if (C[i] != (float) i + (float) i * 100 + 1.0F) error++;
    }

    printf("workers:%d, gangs:%d, size:%d, array reduction error:%d\n", WORKERS, GANGS, size, error);
	if( D != D_ref ) {
		printf("scalar reduction failed!\n");
	} else {
		printf("scalar reduction succeeded!\n");
	}

    return 0;
}

