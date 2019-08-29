#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#ifndef _MAXN_
#define _MAXN_      (1 * 1024)
#endif
#define MAXITCNT    (10)
#define EXPITCNT    (1000)
#define DIFFNORM    (1.0e-2)

int MAXN = _MAXN_;

int main(int argc, char** argv)
{
    int        rank, value, size, errcnt, toterr, i, j, itcnt;
    int        i_first, i_last;
    double     diffnorm, gdiffnorm;

    double (*xlocal)[_MAXN_];
    double (*xnew)[_MAXN_];

    char name[256];
    double time0, time1;
    double ttime0, ttime1;
    double time_init, time_loop = 0.0;

    int direct_dev = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(name, sizeof(name));
    printf("[%s:%d] %d/%d [%s]\n", __FILE__, __LINE__, rank, size, name);

    xlocal = (double(*)[_MAXN_]) malloc(((_MAXN_/size)+2) * _MAXN_ * sizeof(double));
    xnew = (double(*)[_MAXN_]) malloc(((_MAXN_/size)+2) * _MAXN_ * sizeof(double));

    /* xlocal[][0] is lower ghostpoints, xlocal[][_MAXN_+2] is upper */

    /* Note that top and bottom processes have one less row of interior points */
    i_first = 1;
    i_last  = _MAXN_/size;
    if (rank == 0)        i_first++;
    if (rank == size - 1) i_last--;

    /* Fill the data as specified */
    for (i = 1; i <= _MAXN_/size; i++) 
        for (j = 0; j < _MAXN_; j++) 
            xlocal[i][j] = j;

    for (j = 0; j < _MAXN_; j++) {
        xlocal[i_first-1][j] = -1;
        xlocal[i_last+1][j] = -1;
    }

    itcnt = 0;

    time0 = MPI_Wtime();

#pragma acc data create(xnew[0:(MAXN/size)+2][0:MAXN]) copyin(xlocal[0:(MAXN/size)+2][0:MAXN])
    do {
        ttime0 = MPI_Wtime();
        /* Send up unless I'm at the top, then receive from below */
        /* Note the use of xlocal[i] for &xlocal[i][0] */
        if (rank < size - 1) 
            MPI_Send(xlocal[_MAXN_ / size], _MAXN_, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        if (rank > 0)
            MPI_Recv(xlocal[0], _MAXN_, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send down unless I'm at the bottom */
        if (rank > 0) 
            MPI_Send(xlocal[1], _MAXN_, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
        if (rank < size - 1) 
            MPI_Recv(xlocal[_MAXN_ / size + 1], _MAXN_, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Compute new values (but not on boundary) */
        itcnt ++;
        diffnorm = 0.0;
#pragma acc update device(xlocal[0:(MAXN/size)+2][0:MAXN]) if(!direct_dev)
        {
#pragma acc kernels loop independent reduction(+:diffnorm)
            for (j = 1; j < _MAXN_ - 1; j++) {
                for (i = i_first; i <= i_last; i++) {
                    xnew[i][j] = (xlocal[i][j+1] + xlocal[i][j-1] + xlocal[i+1][j] + xlocal[i-1][j]) / 4.0;
                    diffnorm += (xnew[i][j] - xlocal[i][j]) * (xnew[i][j] - xlocal[i][j]);
                }
            }

        /* Only transfer the interior points */
#pragma acc kernels loop gang worker independent
            for (j = 1; j < _MAXN_ - 1; j++) 
                for (i = i_first; i <= i_last; i++) 
                    xlocal[i][j] = xnew[i][j];
        }

        MPI_Allreduce(&diffnorm, &gdiffnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        gdiffnorm = sqrt(gdiffnorm);
        ttime1 = MPI_Wtime();
        if (rank == 0) printf("At iteration %d, diff is %e [%.5f]\n", itcnt, gdiffnorm, ttime1 - ttime0);
        if (itcnt == 1) time_init = ttime1 - ttime0;
        else time_loop += (ttime1 - ttime0);
    } while (gdiffnorm > DIFFNORM && itcnt < MAXITCNT);

    time1 = MPI_Wtime();

    printf("%lf\n", time1 - time0);
    if (rank == 0) printf("size[%d] %lf seconds time_init[%lf] time_loop[%lf] exploop[%d] exp_total[%lf]\n", size, time1 - time0, time_init, time_loop / (double) (itcnt - 1), EXPITCNT, time_init + EXPITCNT * (time_loop / (double) (itcnt - 1)));

    MPI_Finalize();

    return 0;
}

