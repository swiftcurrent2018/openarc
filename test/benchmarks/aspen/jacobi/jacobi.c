#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
//#define _OPENACCM
#ifdef _OPENACCM
#include <openacc.h>
#endif

#ifndef VERIFICATION
#define VERIFICATION 0
#endif

#if VERIFICATION == 1
#define RUN_CPUVERSION 1
#endif

#define ITER 	10

#define CHECK_RESULT

#ifndef SIZE
//#define SIZE 	2048 //128 * 16
//#define SIZE    4096 //256 * 16
//#define SIZE    8192 //256 * 32
#define SIZE  8192 //256 * 48
#endif

#ifdef _OPENARC_

#if SIZE == 4096
#pragma openarc #define SIZE 4096
#elif SIZE == 8192
#pragma openarc #define SIZE 8192
#elif SIZE == 12288
#pragma openarc #define SIZE 12288
#elif SIZE == 16384
#pragma openarc #define SIZE 16384
#endif

#pragma openarc #define SIZE_2 (2+\SIZE)

#endif

#define SIZE_1 	(SIZE+1)
#define SIZE_2 	(SIZE+2)

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0);

    return time.tv_sec + time.tv_usec / 1000000.0;
}



int m_size = SIZE;

int main (int argc, char *argv[])
{
    int i, j, k;
    //int c;
    float sum = 0.0f;

#if RUN_CPUVERSION == 1
	float (*a_CPU)[SIZE_2];
	float (*b_CPU)[SIZE_2];
#endif 
    double strt_time, done_time;
	float (*a)[SIZE_2] = (float (*)[SIZE_2])malloc(sizeof(float) * (m_size+2) * (m_size+2));
	float (*b)[SIZE_2] = (float (*)[SIZE_2])malloc(sizeof(float) * (m_size+2) * (m_size+2));

    //while ((c = getopt (argc, argv, "")) != -1);

    for (i = 0; i < m_size+2; i++)
    {
        for (j = 0; j < m_size+2; j++)
        {
            b[i][j] = 0;
        }
    }

    for (j = 0; j <= SIZE_1; j++)
    {
        b[j][0] = 1.0;
        b[j][SIZE_1] = 1.0;
    }
    for (i = 1; i <= SIZE; i++)
    {
        b[0][i] = 1.0;
        b[SIZE_1][i] = 1.0;
    }

    printf ("Performing %d iterations on a %d by %d array\n", ITER, SIZE, SIZE);

    /* -- Timing starts before the main loop -- */
    printf("-------------------------------------------------------------\n");

    strt_time = my_timer ();

#pragma aspen enter modelregion
#ifdef _OPENACCM
	acc_init(acc_device_default);
#endif

#pragma acc data copy(b[0:m_size+2][0:m_size+2]), create(a[0:m_size+2][0:m_size+2])
    for (k = 0; k < ITER; k++)
    {
#pragma acc kernels loop gang, worker
#pragma openarc transform permute(j,i)
        for (i = 1; i <= m_size; i++)
        {
            for (j = 1; j <= m_size; j++)
            {
                a[i][j] = (b[i - 1][j] + b[i + 1][j] + b[i][j - 1] + b[i][j + 1]) / 4.0f;
            }
        }

#pragma acc kernels loop gang worker
#pragma openarc transform permute(j,i)
        for (i = 1; i <= m_size; i++)
        {
            for (j = 1; j <= m_size; j++)
            {
                b[i][j] = a[i][j];
            }
        }
    }
#ifdef _OPENACCM
	acc_shutdown(acc_device_default);
#endif
#pragma aspen exit modelregion

    done_time = my_timer ();
    printf ("Accelerator Elapsed time = %lf sec\n", done_time - strt_time);

#ifdef CHECK_RESULT
    for (i = 1; i <= SIZE; i++)
    {
        sum += b[i][i];
    }
    printf("Diagonal sum = %.10E\n", sum);
#endif

#if RUN_CPUVERSION == 1
#if VERIFICATION != 1
	printf("free a and b\n");
	free(a);
	free(b);
#endif
	a_CPU = (float (*)[SIZE_2])malloc(sizeof(float) * (m_size+2) * (m_size+2));
	b_CPU = (float (*)[SIZE_2])malloc(sizeof(float) * (m_size+2) * (m_size+2));
    for (i = 0; i < m_size+2; i++)
    {
        for (j = 0; j < m_size+2; j++)
        {
			b_CPU[i][j] = 0;
        }
    }

    for (j = 0; j <= SIZE_1; j++)
    {
		b_CPU[j][0] = 1.0;
		b_CPU[j][SIZE_1] = 1.0;

    }
    for (i = 1; i <= SIZE; i++)
    {
		b_CPU[0][i] = 1.0;
		b_CPU[SIZE_1][i] = 1.0;
    }

    strt_time = my_timer ();

    for (k = 0; k < ITER; k++)
    {
#pragma omp parallel for private(i,j)
        for (i = 1; i <= m_size; i++)
        {
            for (j = 1; j <= m_size; j++)
            {
                a_CPU[i][j] = (b_CPU[i - 1][j] + b_CPU[i + 1][j] + b_CPU[i][j - 1] + b_CPU[i][j + 1]) / 4.0f;
            }
        }

#pragma omp parallel for private(i,j)
        for (i = 1; i <= m_size; i++)
        {
            for (j = 1; j <= m_size; j++)
            {
                b_CPU[i][j] = a_CPU[i][j];
            }
        }
    }
    done_time = my_timer ();
    printf ("CPU Elapsed time = %lf sec\n", done_time - strt_time);

#if VERIFICATION == 1
	{
		double cpu_sum = 0.0;
		double gpu_sum = 0.0;
    	double rel_err = 0.0;

		for (i = 1; i <= SIZE; i++)
    	{
        	cpu_sum += b_CPU[i][i]*b_CPU[i][i];
			gpu_sum += b[i][i]*b[i][i];
    	}

		cpu_sum = sqrt(cpu_sum);
		gpu_sum = sqrt(gpu_sum);
		rel_err = (cpu_sum-gpu_sum)/cpu_sum;

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
#ifdef CHECK_RESULT
	sum = 0.0;
    for (i = 1; i <= SIZE; i++)
    {
        sum += b_CPU[i][i];
    }
    printf("Diagonal sum = %.10E\n", sum);
#endif
#endif


    //printf ("done_time = %lf\n", done_time);

    return 0;
}

