/*
 *  Copyright 2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <math.h>
#include <string.h>
#include "timer.h"
#include <stdio.h>
#ifdef _OPENACCM
#include <openacc.h>
#endif


#ifndef VERIFICATION
#define VERIFICATION 0
#endif

#ifndef NN
#define NN 1024
#endif 

#ifdef _OPENARC_

#if NN == 1024
#pragma openarc #define NN 1024
#elif NN == 2048
#pragma openarc #define NN 2048
#elif NN == 3072
#pragma openarc #define NN 3072
#elif NN == 3072
#pragma openarc #define NN 3072
#elif NN == 4096
#pragma openarc #define NN 4096
#endif

#endif

#if VERIFICATION == 1
double A_CPU[NN][NN];
double Anew_CPU[NN][NN];
#endif

int main(int argc, char** argv)
{
    int n = NN;
    int m = n;
    int iter_max = 10;

    double tol = 1.0e-6;
    double error     = 1.0;
    int i, j;
    int iter = 0;
    double runtime;

	double (*A)[NN] = (double (*)[NN])malloc(sizeof(double)*n*n);
	double (*Anew)[NN] = (double (*)[NN])malloc(sizeof(double)*n*n);
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));

#if VERIFICATION == 1
    memset(A_CPU, 0, n * m * sizeof(double));
    memset(Anew_CPU, 0, n * m * sizeof(double));
#endif

    for (j = 0; j < n; j++)
    {
        A[j][0]    = 1.0;
        Anew[j][0] = 1.0;

#if VERIFICATION == 1
        A_CPU[j][0] = 1.0;
        Anew_CPU[j][0] = 1.0;
#endif
    }

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

    StartTimer();

#pragma aspen enter modelregion
#ifdef _OPENACCM
    acc_init(acc_device_default);
#endif

//aspen_param_whilecnt = 1000 for NN = NN = 4096
//aspen_param_whilecnt = 1000 for NN = NN = 8192
#pragma aspen declare param(aspen_param_whilecnt:10)
#pragma aspen control loop(aspen_param_whilecnt)
#pragma acc data copy(A[0:n][0:n]), create(Anew[0:n][0:n])
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

//#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc parallel num_gangs(16) num_workers(32) reduction(max:error) private(j)
        {
			double lerror = 0.0;
#pragma acc loop gang
            for( j = 1; j < n-1; j++)
            {
#pragma acc loop worker reduction(max:lerror)
                for( i = 1; i < m-1; i++ )
                {
                    Anew[j][i] = 0.25 * ( A[j][i+1] + A[j][i-1]
                                          + A[j-1][i] + A[j+1][i]);
                    lerror = fmax( lerror, fabs(Anew[j][i] - A[j][i]));
                }
//[DEBUG] intentionally ignore to flatten nested map constructs.
#pragma aspen control ignore
				error = fmax(error, lerror);
            }
        }

//#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels loop gang
        for( j = 1; j < n-1; j++)
        {
#pragma acc loop worker
            for( i = 1; i < m-1; i++ )
            {
                A[j][i] = Anew[j][i];
            }
        }

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);

        iter++;
    }

#ifdef _OPENACCM
    acc_shutdown(acc_device_default);
#endif
#pragma aspen exit modelregion

	printf("iter: %d\n", iter);

    runtime = GetTimer();
    printf("Accelerator Elapsed time %f s\n", runtime / 1000);

#if VERIFICATION == 1
    {
    	StartTimer();
        error = 1.0;
        iter = 0;
        while ( error > tol && iter < iter_max )
        {
            error = 0.0;

            {
				#pragma omp parallel for private(j, i)
                for( j = 1; j < n-1; j++)
                {
					double lerror = 0.0;
                    for( i = 1; i < m-1; i++ )
                    {
                        Anew_CPU[j][i] = 0.25 * ( A_CPU[j][i+1] + A_CPU[j][i-1]
                                                  + A_CPU[j-1][i] + A_CPU[j+1][i]);
                        lerror = fmax( lerror, fabs(Anew_CPU[j][i] - A_CPU[j][i]));
                    }
					#pragma omp critical
					error = fmax(error,lerror);
                }
            }
			#pragma omp parallel for private(j, i)
            for( j = 1; j < n-1; j++)
            {
                for( i = 1; i < m-1; i++ )
                {
                    A_CPU[j][i] = Anew_CPU[j][i];
                }
            }
            if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
            iter++;
        }
    	runtime = GetTimer();
    	printf("CPU Elapsed time %f s\n", runtime / 1000);

        {
            double cpu_sum = 0.0f;
            double gpu_sum = 0.0f;
            double rel_err = 0.0f;

            for (i = 1; i < m-1; i++)
            {
                cpu_sum += A_CPU[i][i]*A_CPU[i][i];
                gpu_sum += A[i][i]*A[i][i];
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
    }
#endif

}
