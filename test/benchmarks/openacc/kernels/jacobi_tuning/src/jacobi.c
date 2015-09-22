#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#ifndef VERIFICATION
#define VERIFICATION 0
#endif

#define ITER 	1000

#ifndef SIZE
#define SIZE 	8192
#ifdef _OPENARC_
#pragma openarc #define SIZE 8192
#endif
#endif

#define SIZE_1 	(SIZE+1)
#define SIZE_2 	(SIZE+2)

#ifdef _OPENARC_
#pragma openarc #define SIZE_2 (2+SIZE)
#endif

#define CHECK_RESULT

#if HOST_MEM_ALIGNMENT == 1
#include <malloc.h>
#define malloc(size) valloc(size)
#endif

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0);

    return time.tv_sec + time.tv_usec / 1000000.0;
}

float (*a)[SIZE_2];
float (*b)[SIZE_2];

int main (int argc, char *argv[])
{
    int i, j, k;
    //int c;
    float sum = 0.0f;

    double strt_time, done_time;

	float* a_GPU = (float*)malloc(sizeof(float) * SIZE_2 * SIZE_2);
	float* b_GPU = (float*)malloc(sizeof(float) * SIZE_2 * SIZE_2);

    a = (float (*)[SIZE_2]) a_GPU;
    b = (float (*)[SIZE_2]) b_GPU;

#if VERIFICATION >= 1
	float** a_CPU = (float**)malloc(sizeof(float*) * SIZE_2);
	float** b_CPU = (float**)malloc(sizeof(float*) * SIZE_2);

	float* a_data = (float*)malloc(sizeof(float) * SIZE_2 * SIZE_2);
	float* b_data = (float*)malloc(sizeof(float) * SIZE_2 * SIZE_2);

	for(i = 0; i < SIZE_2; i++)
	{
		a_CPU[i] = &a_data[i * SIZE_2];
		b_CPU[i] = &b_data[i * SIZE_2];
	}

#endif 

    //while ((c = getopt (argc, argv, "")) != -1);

    for (i = 0; i < SIZE_2; i++)
    {
        for (j = 0; j < SIZE_2; j++)
        {
            b[i][j] = 0;
#if VERIFICATION >= 1
			b_CPU[i][j] = 0;
#endif 
        }
    }

    for (j = 0; j <= SIZE_1; j++)
    {
        b[j][0] = 1.0;
        b[j][SIZE_1] = 1.0;

#if VERIFICATION >= 1
		b_CPU[j][0] = 1.0;
		b_CPU[j][SIZE_1] = 1.0;
#endif 

    }
    for (i = 1; i <= SIZE; i++)
    {
        b[0][i] = 1.0;
        b[SIZE_1][i] = 1.0;

#if VERIFICATION >= 1
		b_CPU[0][i] = 1.0;
		b_CPU[SIZE_1][i] = 1.0;
#endif 
    }

    printf ("Performing %d iterations on a %d by %d array\n", ITER, SIZE, SIZE);

    /* -- Timing starts before the main loop -- */
    printf("-------------------------------------------------------------\n");

    strt_time = my_timer ();

#pragma acc data copy(b[0:SIZE_2][0:SIZE_2]), create(a[0:SIZE_2][0:SIZE_2])
    for (k = 0; k < ITER; k++)
    {
#ifdef _OPENARC_
#pragma openarc transform permute(j,i)
#endif
#pragma acc kernels loop gang, worker
        for (i = 1; i <= SIZE; i++)
        {
#if UNROLLING == 1
#pragma unroll 2
#endif
            for (j = 1; j <= SIZE; j++)
            {
                a[i][j] = (b[i - 1][j] + b[i + 1][j] + b[i][j - 1] + b[i][j + 1]) / 4.0f;
            }
        }

#pragma acc kernels loop gang worker
#pragma openarc transform permute(j,i)
        for (i = 1; i <= SIZE; i++)
        {
#if UNROLLING == 1
#pragma unroll 2
#endif
            for (j = 1; j <= SIZE; j++)
            {
                b[i][j] = a[i][j];
            }
        }
    }

    done_time = my_timer ();

#if VERIFICATION >= 1

    for (k = 0; k < ITER; k++)
    {
        for (i = 1; i <= SIZE; i++)
        {
            for (j = 1; j <= SIZE; j++)
            {
                a_CPU[i][j] = (b_CPU[i - 1][j] + b_CPU[i + 1][j] + b_CPU[i][j - 1] + b_CPU[i][j + 1]) / 4.0f;
            }
        }

        for (i = 1; i <= SIZE; i++)
        {
            for (j = 1; j <= SIZE; j++)
            {
                b_CPU[i][j] = a_CPU[i][j];
            }
        }
    }
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
	}
#else
	{
		double cpu_sum = 0.0;
		double gpu_sum = 0.0;
    	double rel_err = 0.0;
		int error_found = 0;

        for (i = 1; i <= SIZE; i++)
        {
            for (j = 1; j <= SIZE; j++)
            {
        		cpu_sum = b_CPU[i][j];
				gpu_sum = b[i][j];
				if( cpu_sum == gpu_sum ) {
					continue;
				}
				if( cpu_sum > gpu_sum) {
					if( cpu_sum == 0.0 ) {
						rel_err = cpu_sum-gpu_sum;
					} else {
						rel_err = (cpu_sum-gpu_sum)/cpu_sum;
					}
				} else {
					if( cpu_sum == 0.0 ) {
						rel_err = gpu_sum-cpu_sum;
					} else {
						rel_err = (gpu_sum-cpu_sum)/cpu_sum;
					}
				}
				if(rel_err < 0.0) {
					rel_err = -1*rel_err;
				}

				if(rel_err >= 1e-9)
				{
					error_found = 1;
					break;
				}
			}
			if( error_found == 1 ) {
				break;
			}
		}
		if( error_found == 0 )
		{
	    	printf("Verification Successful\n");
		}
		else
		{
	    	printf("Verification Fail err = %e\n", rel_err);
		}
	}
#endif
#endif


#ifdef CHECK_RESULT
    for (i = 1; i <= SIZE; i++)
    {
        sum += b[i][i];
    }
    printf("Diagonal sum = %.10E\n", sum);
#endif
    //printf ("done_time = %lf\n", done_time);
    printf ("Accelerator Elapsed time = %lf sec\n", done_time - strt_time);

    return 0;
}

