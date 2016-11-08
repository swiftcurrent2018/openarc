#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

extern double timer_();

//////////////////////////
// Macros for debugging //
//////////////////////////
//#define DEBUG_ON1
#define DEBUG_ON2

#ifndef SPMUL_INPUTDIR
#define SPMUL_INPUTDIR        "/home/f6l/SPMULInput/"
#endif
#define ITER	100

/*
#define INPUTFILE  "nlpkkt240.rbC"
#define SIZE  27993600
#define SIZE2  27993600 //debugging purpose (should be replaced with SIZE)
#define NZR    401232976
#ifdef _OPENARC_
#pragma openarc #define SIZE  27993600
#pragma openarc #define SIZE2  27993600 //debugging purpose (should be replaced with SIZE)
#pragma openarc #define NZR    401232976
#endif
*/

/*
#define INPUTFILE  "af23560.rbC"
#define SIZE  23560
#define SIZE2  23560 //debugging purpose (should be replaced with SIZE)
#define NZR    484256
#ifdef _OPENARC_
#pragma openarc #define SIZE  23560
#pragma openarc #define SIZE2  23560 //debugging purpose (should be replaced with SIZE)
#pragma openarc #define NZR    484256
#endif
*/

/*
#define INPUTFILE	"rajat31.rbC"
#define SIZE	4690002
#define SIZE2	4690002 //debugging purpose (should be replaced with SIZE)
#define NZR		20316253
#ifdef _OPENARC_
#pragma openarc #define SIZE  4690002
#pragma openarc #define SIZE2  4690002 //debugging purpose (should be replaced with SIZE)
#pragma openarc #define NZR    20316253
#endif
*/

/*
#define INPUTFILE	"af_shell10.rbC"
#define SIZE	1508065
#define SIZE2	1508065 //debugging purpose (should be replaced with SIZE)
#define NZR		27090195
#ifdef _OPENARC_
#pragma openarc #define SIZE  1508065
#pragma openarc #define SIZE2  1508065 //debugging purpose (should be replaced with SIZE)
#pragma openarc #define NZR    27090195
#endif
*/

/*
#define INPUTFILE   "hood.rbC"
#define SIZE    220542  
#define SIZE2   220542  
#define NZR 5494489 
#ifdef _OPENARC_
#pragma openarc #define SIZE	220542
#pragma openarc #define SIZE2	220542 //debugging purpose (should be replaced with SIZE)
#pragma openarc #define NZR		5494489
#endif
*/

#define INPUTFILE	"kkt_power.rbC"
#define SIZE	2063494
#define SIZE2	2063494 //debugging purpose (should be replaced with SIZE)
#define NZR		8130343
#ifdef _OPENARC_
#pragma openarc #define SIZE	2063494
#pragma openarc #define SIZE2	2063494 //debugging purpose (should be replaced with SIZE)
#pragma openarc #define NZR		8130343
#endif

/*
#define ITER	500
#define INPUTFILE	"msdoor.rbC"
#define SIZE	415863
#define SIZE2	415863 
#define NZR		10328399
*/

/*
#define INPUTFILE	"appu.rbC"
//#define INPUTFILE	"appu.rbCR"
//#define INPUTFILE	"appu.rbCRP"
#define SIZE	14000
#define SIZE2	14000 
#define NZR		1853104
//#define NZR		1857600
*/

/*
#define INPUTFILE	"nd24k.rbC"
#define SIZE	72000
#define SIZE2	72000 
#define NZR		14393817

//#define INPUTFILE	"F1.rbC"
#define INPUTFILE	"F1.rbCRP"
#define SIZE	343791
#define SIZE2	343791 
//#define NZR		13590452
#define NZR		13596431

//#define INPUTFILE	"ASIC_680k.rbC"
#define INPUTFILE	"ASIC_680k.rbCR"
#define SIZE	682862
#define SIZE2	682862
#define NZR		3871773

#define INPUTFILE	"ASIC_680ks.rbC"
#define SIZE	682712
#define SIZE2	682712 
#define NZR		2329176

#define INPUTFILE	"crankseg_2.rbC"
#define SIZE	63838
#define SIZE2	63838 
#define NZR		7106348

#define INPUTFILE	"darcy003.rbC"
#define SIZE	389874
#define SIZE2	389874 
#define NZR		1167685

#define INPUTFILE	"Si41Ge41H72.rbC"
#define SIZE	185639
#define SIZE2	185639 
#define NZR		7598452

#define INPUTFILE	"SiO2.rbC"
#define SIZE	155331
#define SIZE2	155331 
#define NZR		5719417
*/
/*
#define INPUTFILE   "sparsine.rbCR"
#define SIZE    50000
#define SIZE2   50000
#define NZR 799494

#define INPUTFILE   "sparsine.rbCRPF"
#define SIZE    50000   
#define SIZE2   50000   
#define NZR 3200000 
*/
/*
#define INPUTFILE   "ns3Da.rbCRPF"
#define SIZE    20414   
#define SIZE2   20414   
#define NZR 6533120 
*/

/*
#define INPUTFILE   "af23560.rand51M"
#define SIZE    100000  
#define SIZE2   100000
#define NZR 6400000
*/

/*
#define INPUTFILE   "af23560.rand200M"
#define SIZE    100000  
#define SIZE2   100000
#define NZR 25600000
*/

/*
int colind[NZR];
int rowptr[SIZE+1];
float values[NZR];
float x[SIZE];
float y[SIZE];
*/
int *colind;
int *rowptr;
float *values;
float *x;
float *y;

int main() {
	FILE *fp10;
	//FILE *fp12; //Result writing part is disabled
	char filename1[96] = SPMUL_INPUTDIR; 
	char filename2[32] = INPUTFILE; 

	float temp, x_sum;
	double s_time1, e_time1, s_time2, e_time2;
	double s_time3, e_time3;
	int exp0, i, j, k;
	int r_ncol, r_nnzero, r_nrow;
	int cpumemsize = 0;



	colind = (int *)malloc(sizeof(int)*NZR);	
	rowptr = (int *)malloc(sizeof(int)*(SIZE+1));	
	values = (float *)malloc(sizeof(int)*NZR);	
	x = (float *)malloc(sizeof(int)*SIZE);	
	y = (float *)malloc(sizeof(int)*SIZE);	
  printf("**** SerialSpmul starts! ****\n");

#if defined(_OPENMP)
	omp_set_num_threads(8);
#endif
	strcat(filename1, filename2);

	printf("Input file: %s\n", filename2);

	s_time1 = timer_();
	s_time2 = timer_();
	if( (fp10 = fopen(filename1, "r")) == NULL ) {
		printf("FILE %s DOES NOT EXIST; STOP\n", filename1);
		exit(1);
	}
/*
	if( (fp12 = fopen("spmulSP.out", "w")) == NULL ) {
		exit(1);
	}
*/
	printf("FILE open done\n");

	fscanf(fp10, "%d %d %d", &r_nrow, &r_ncol, &r_nnzero);
	if (r_nrow != SIZE) {
		printf("alarm: incorrect row\n");
		exit(1);
	}
	if (r_ncol != SIZE) {
		printf("alarm: incorrect col\n");
		exit(1);
	}
	if (r_nnzero != NZR) {
		printf("alarm: incorrect nzero\n");
		exit(1);
	}
	for( i=0; i<=SIZE; i++ ) {
		fscanf(fp10, "%d", (rowptr+i));
	}

	for( i=0; i<NZR; i++ ) {
		fscanf(fp10, "%d", (colind+i));
	}

	for( i=0; i<NZR; i++ ) {
		fscanf(fp10, "%E", (values+i)); //for float variables
	}
	fclose(fp10);

	j = 0;
    for( i=0; i<SIZE; i++ ) {
LB99:
		temp = values[j];
		if( ((-0.1f) < temp)&&(temp < 0.1f) ) {
			j += 1;
			//goto LB99;
			//Added by SYLee
			if( temp == 0.0f )
				goto LB99;
			x[i] = temp; 
			continue;
		}
		exp0 = (int)(log10f(fabsf(temp)));
		x[i] = temp;
		if( (-exp0) <= 0 ) {
			for( k=1; k<=(1+exp0); k++ ) {
				x[i] = x[i]/10.0f;
			}
		} else if( (1+exp0) <= 0 ) {
			k = -1;
			for( k=1; k<=(-exp0); k++ ) {
				x[i] = 10.0f*x[i];
			}
		}
		if( (1.0f < x[i])||(x[i] < (-1.0f)) ) {
			printf("alarm initial i = %d\n", i);
			printf("x = %E\n", x[i]); 
			printf("value = %E\n", values[1000+i]);
			printf("exp = %d\n", exp0);
			exit(1);
		}
		j += 1;
	}

#ifdef DEBUG_ON1
	x_sum = 0.0f;
	for( i=0; i<SIZE; i++ ) {
		x_sum += x[i];
	}
	printf("0: x_sum = %.12E\n", x_sum);
#endif
	cpumemsize += sizeof(int) * (NZR + SIZE + 1);
	cpumemsize += sizeof(float) * (NZR + 2*SIZE);
	printf("Used CPU memory: %d bytes\n", cpumemsize);

	printf("initialization done\n");
	e_time2 = timer_();
	s_time3 = timer_();


////////////////////////////////////////////////////////////////////////
// This version has more fork-join overhead, but easier for GPU trans.//
////////////////////////////////////////////////////////////////////////
// Loop iteration variable in the for-loop of a for or parallel for   //
// construct is private in that construct. (predetermined atrributes) //
////////////////////////////////////////////////////////////////////////
// exp0 and j should be explicitly declased as private.               //
// i will be private automatically, but added in private list         //
////////////////////////////////////////////////////////////////////////
#pragma acc data copy(x[0:SIZE]), copyin(values[0:NZR], colind[0:NZR], rowptr[0:SIZE+1]), create(y[0:SIZE])
	for( k=0; k<ITER; k++ ) {
#pragma acc kernels loop gang, worker
        for( i=0; i<SIZE2; i++ ) { 
            y[i] = 0.0f;
            for( j=0; j<(rowptr[1+i]-rowptr[i]); j++ ) { 
                y[i] = y[i] + values[rowptr[i]+j-1]*x[colind[rowptr[i]+j-1]-1];
            }   
        } //barrier with flush is implied 

/////////////////////////////////////////////////////////////////////
// The above implicit barrier is needed because y is written in    //
// the first for-loop and read in the second for-loop.             //
// (All threads in the second loop should read y that was modified //
// in the first loop.)                                             //
/////////////////////////////////////////////////////////////////////
#pragma acc kernels loop gang, worker
		for( i=0; i<SIZE2; i++ ) {
			float tmp = y[i];
			x[i] = tmp;
			if( tmp != 0.0f ) {
				exp0 = (int)(log10f(fabsf(tmp)));
				if( exp0 >= 0 ) {
					for( j=1; j<=(1+exp0); j++ ) {
						x[i] = x[i]/10.0f;
					}
				} else if( exp0 <= -1 ) {
					j = -1;
					for( j=1; j<=(-exp0); j++ ) {
						x[i] = 10.0f*x[i];
					}
				} 
			}
		} //barrier with flush is implied 
////////////////////////////////////////////////////////////////////
// The above implicit barrier is needed because x is read in the  //
// first for-loop and modified in the second for-loop.            //
// (All threads in the first loop should read x that was modified //
// in the previous iteration.)                                    //
////////////////////////////////////////////////////////////////////
	} //end of k-loop

	e_time3 = timer_();
	e_time1 = timer_();
	printf("Total time = %f seconds\n", (e_time1 - s_time1));
	printf("Initialize time = %f seconds\n", (e_time2 - s_time2));
	printf("Accelerator Elapsed time = %f seconds\n", (e_time3 - s_time3));

#ifdef DEBUG_ON2
	x_sum = 0.0f;
	for( i=0; i<SIZE2; i++ ) {
		x_sum += x[i];
	}
	printf("%d: x_sum = %.12E\n",(k+1), x_sum);
#endif

/*
	for( i=0; i< SIZE; i++ ) {
		fprintf(fp12, "%.9E\n", x[i]);
	} 

	fclose(fp12);
*/

	return 0;
}
