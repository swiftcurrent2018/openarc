#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE    (8 * 256)
#pragma openarc #define SIZE (8 * 256)

double *A, *B, *C;
MPI_Request *req;

double wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.e-6 * tv.tv_usec;
}

int main(int argc, char** argv) {
    int size, rank;
    int i, j, k;
    double t0, t1, t2, t3;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    req = (MPI_Request*) malloc(size * sizeof(MPI_Request));

    if (rank == 0) {
        A = (double*) calloc(SIZE * SIZE, sizeof(double));
        B = (double*) calloc(SIZE * SIZE, sizeof(double));
        C = (double*) calloc(SIZE * SIZE, sizeof(double));
    } else {
        A = (double*) calloc(SIZE * SIZE / size, sizeof(double));
        B = (double*) calloc(SIZE * SIZE, sizeof(double));
        C = (double*) calloc(SIZE * SIZE / size, sizeof(double));
    }

    if (rank == 0) {
        for (i = 0; i < SIZE; i++) {
            for (j = 0; j < SIZE; j++) {
                A[i * SIZE + j] = i + j;
                B[i * SIZE + j] = 1;
            }
        }
    }

    t0 = wtime();
    MPI_Bcast(B, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (i = 1; i < size; i++) {
            MPI_Isend(A + i * (SIZE * SIZE / size), SIZE * SIZE / size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, req + i - 1);
        }
        MPI_Waitall(size - 1, req, MPI_STATUSES_IGNORE);
    } else {
        MPI_Recv(A, SIZE * SIZE / size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    t2 = wtime();
#pragma acc parallel loop gang worker independent copyin(A[0:SIZE*SIZE/size], B[0:SIZE*SIZE]) copyout(C[0:SIZE*SIZE/size]) private(i, j, k)
    for (i = 0; i < SIZE / size; i++) {
        for (j = 0; j < SIZE; j++) {
            double sum = 0;
            for (k = 0; k < SIZE; k++) {
                sum += A[i * SIZE + k] * B[k * SIZE + j];
            }
            C[i * SIZE + j] = sum;
        }
    }
    t3 = wtime();

    if (rank == 0) {
        for (i = 1; i < size; i++) {
            MPI_Irecv(C + i * (SIZE * SIZE / size), SIZE * SIZE / size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, req + i - 1);
        }
        MPI_Waitall(size - 1, req, MPI_STATUSES_IGNORE);
    } else  {
        MPI_Send(C, SIZE * SIZE / size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    t1 = wtime();

    printf("rank[%d] total: %lf secs. kernel %lf secs\n", rank, t1 - t0, t3 - t2);

#if 1
    if (rank == 0) {
        printf("verification...\n");
        for (i = 0; i < SIZE; i++) {
            int running = 1;
            for (j = 0; j < SIZE; j++) {
                double sum = 0;
                for (k = 0; k < SIZE; k++) {
                    sum += A[i * SIZE + k] * B[k * SIZE + j];
                }
                if (C[i * SIZE + j] != sum)
                {
                    printf("C[%d][%d] = %lf vs %lf\n", i, j, C[i * SIZE + j], sum);
                    running = 0;
                    break;
                }
            }
            if (!running) break;
        }
    }
#endif

    MPI_Finalize();

    return 0;
}

