#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int workers, gangs, size;
    float *A, *B, *C;
    int i;
    int error = 0;

    workers = 16;
    gangs = 256;
    size = workers * gangs;

    A = (float*) malloc(size * sizeof(float));
    B = (float*) malloc(size * sizeof(float));
    C = (float*) malloc(size * sizeof(float));

    for (i = 0; i < size; i++) {
        A[i] = (float) i;
        B[i] = (float) i * 100;
    }

#pragma acc data copyin(A[0:size], B[0:size]) copyout(C[0:size])
    {
#pragma acc kernels loop independent gang worker(16)
        for (i = 0; i < size; i++) {
            C[i] = A[i] + B[i];
        }
    }

    for (i = 0; i < size; i++) {
        if (C[i] != (float) i + (float) i * 100) error++;
    }

    printf("workers:%d, gangs:%d, size:%d, error:%d\n", workers, gangs, size, error);

    return 0;
}

