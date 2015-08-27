/*--------------------------------------------------------------------

  NAS Parallel Benchmarks 2.3 OpenMP C versions - EP

  This benchmark is an OpenMP C version of the NPB EP code.

  The OpenMP C versions are developed by RWCP and derived from the serial
  Fortran versions in "NPB 2.3-serial" developed by NAS.

  Permission to use, copy, distribute and modify this software for any
  purpose with or without fee is hereby granted.
  This software is provided "as is" without express or implied warranty.

  Send comments on the OpenMP C versions to pdp-openmp@rwcp.or.jp

  Information on OpenMP activities at RWCP is available at:

http://pdplab.trc.rwcp.or.jp/pdperf/Omni/

Information on NAS Parallel Benchmarks 2.3 is available at:

http://www.nas.nasa.gov/NAS/NPB/

--------------------------------------------------------------------*/
/*--------------------------------------------------------------------

Author: P. O. Frederickson 
D. H. Bailey
A. C. Woo

OpenMP C version: S. Satoh

--------------------------------------------------------------------*/
//#define DEBUG_ON_x
//#define DEBUG_ON_t1
//#define DEBUG_ON_q

#include "npb-C.h"
#include "npbparams.h"
#include "openacc.h"

/* parameters */
#define	MK		12
#define	MM		(M - MK)
#define	NN		(1 << MM)
#define	NK		(1 << MK)
#define	NQ		10
#define EPSILON		1.0e-8
#define	A		1220703125.0
#define	S		271828183.0
//#define	TIMERS_ENABLED	FALSE

#if defined(USE_POW)
#define r23 pow(0.5, 23.0)
#define r46 (r23*r23)
#define t23 pow(2.0, 23.0)
#define t46 (t23*t23)
#else
#define r23 (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)
#define r46 (r23*r23)
#define t23 (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)
#define t46 (t23*t23)
#endif

#ifndef _UNROLLFAC_
#define _UNROLLFAC_	1
#endif

#ifdef _OPENARC_
#if _UNROLLFAC_ == 1
#pragma openarc #define _UNROLLFAC_ 1
#elif _UNROLLFAC_ == 6
#pragma openarc #define _UNROLLFAC_ 6
#elif _UNROLLFAC_ == 8
#pragma openarc #define _UNROLLFAC_ 8
#elif _UNROLLFAC_ == 32
#pragma openarc #define _UNROLLFAC_ 32
#elif _UNROLLFAC_ == 128
#pragma openarc #define _UNROLLFAC_ 128
#elif _UNROLLFAC_ == 1024
#pragma openarc #define _UNROLLFAC_ 1024
#endif

#pragma openarc #define NK 4096
#endif


/* global variables */
/* common /storage/ */
//static double x[2*NK];
double *x;
//#pragma omp threadprivate(x)
static double q[NQ];

/*--------------------------------------------------------------------
  program EMBAR
  c-------------------------------------------------------------------*/
/*
   c   This is the serial version of the APP Benchmark 1,
   c   the "embarassingly parallel" benchmark.
   c
   c   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
   c   numbers.  MK is the Log_2 of the size of each batch of uniform random
   c   numbers.  MK can be set for convenience on a given system, since it does
   c   not affect the results.
 */
int main(int argc, char **argv) {

		double Mops, t1, t2, t3, t4, x1, x2, sx, sy, tm, an, tt, gc;
		double dum[3] = { 1.0, 1.0, 1.0 };
		int np, ierr, node, no_nodes, i, ik, kk, l, k, nit, ierrcode,
			no_large_nodes, np_add, k_offset, j;
		int nthreads = 1;
		boolean verified;
		char size[13+1];	/* character*13 */

		//double t1, t2, t3, t4, x1, x2;
		//int kk, i, ik, l;
		//double qq[NQ];		/* private copy of q[0:NQ-1] */
		double qq0;
		double qq1;
		double qq2;
		double qq3;
		double qq4;
		double qq5;
		double qq6;
		double qq7;
		double qq8;
		double qq9;
		double t1_randlc,t2_randlc,t3_randlc,t4_randlc,a1_randlc,a2_randlc,x1_randlc,x2_randlc,z_randlc, a_randlc;
		int i_vranlc;
		double x_vranlc;
		double (*xx)[(NN/_UNROLLFAC_)];
		int m;



		/*
		   c   Because the size of the problem is too large to store in a 32-bit
		   c   integer for some classes, we put it into a string (for printing).
		   c   Have to strip off the decimal point put in there by the floating
		   c   point print statement (internal file)
		 */

		printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"
						" - EP Benchmark\n");
		sprintf(size, "%12.0f", pow(2.0, M+1));
		for (j = 13; j >= 1; j--) {
				if (size[j] == '.') size[j] = ' ';
		}
		printf(" Number of random numbers generated: %13s\n", size);

		verified = FALSE;
		x = (double *)acc_create_unified(NULL, sizeof(double)*2*NK);

		/*
		   c   Compute the number of "batches" of random number pairs generated 
		   c   per processor. Adjust if the number of processors does not evenly 
		   c   divide the total number
		 */
		np = NN;

		/*
		   c   Call the random number generator functions and initialize
		   c   the x-array to reduce the effects of paging on the timings.
		   c   Also, call all mathematical functions that are used. Make
		   c   sure these initializations cannot be eliminated as dead code.
		 */
		vranlc(0, &(dum[0]), dum[1], &(dum[2]));
		dum[0] = randlc(&(dum[1]), dum[2]);
		for (i = 0; i < 2*NK; i++) x[i] = -1.0e38;
		Mops = log(sqrt(fabs(max(1.0, 1.0))));

		timer_clear(1);
		timer_clear(2);
		timer_clear(3);
		timer_start(1);

		vranlc(0, &t1, A, x);

		/*   Compute AN = A ^ (2 * NK) (mod 2^46). */

		t1 = A;

		for ( i = 1; i <= MK+1; i++) {
				t2 = randlc(&t1, t1);
		}

		an = t1;
		tt = S;
		gc = 0.0;
		sx = 0.0;
		sy = 0.0;

		for ( i = 0; i <= NQ - 1; i++) {
				q[i] = 0.0;
		}
		qq0 = 0.0;
		qq1 = 0.0;
		qq2 = 0.0;
		qq3 = 0.0;
		qq4 = 0.0;
		qq5 = 0.0;
		qq6 = 0.0;
		qq7 = 0.0;
		qq8 = 0.0;
		qq9 = 0.0;

		/*
		   c   Each instance of this loop may be performed independently. We compute
		   c   the k offsets separately to take into account the fact that some nodes
		   c   have more numbers to generate than others
		 */
		k_offset = -1;
		xx = (double (*)[NN/_UNROLLFAC_])acc_create_unified(NULL, (2*NK)*(NN/_UNROLLFAC_)*sizeof(double));

#pragma acc kernels loop gang, worker, \
		copyin(x[0:2*NK]), create(xx[0:2*NK][0:(NN/_UNROLLFAC_)]), \
		private(t1, t2, t3, t4, x1, x2, k, kk, i, ik, m), \
		private(l, t1_randlc, t2_randlc, t3_randlc, t4_randlc, a1_randlc, a2_randlc), \
		private(x1_randlc, x2_randlc, z_randlc, a_randlc, i_vranlc, x_vranlc)
		for (m = 0; m < (NN/_UNROLLFAC_); m++)
		{
				for (i = 0; i < 2*NK; i++) xx[i][m] = x[i];
				for (k = 0; k <_UNROLLFAC_; k++)
				{

						//for (i = 0; i < NQ; i++) qq[i] = 0.0f;

						//#pragma omp for reduction(+:sx,sy) schedule(static) nowait 
						kk = k_offset + (m+k*(NN/_UNROLLFAC_)) + 1;
						t1 = S;
						t2 = an;

						/*      Find starting seed t1 for this kk. */

						for (i = 1; i <= 100; i++) {
								ik = kk / 2;
								if (2 * ik != kk) {
										//t3 = randlc(&t1, t2);
										a_randlc = t2;
										t1_randlc = r23 * a_randlc;
										a1_randlc = (int)t1_randlc;
										a2_randlc = a_randlc - t23 * a1_randlc;

										t1_randlc = r23 * t1;
										x1_randlc = (int)t1_randlc;
										x2_randlc = t1 - t23 * x1_randlc;
										t1_randlc = a1_randlc * x2_randlc + a2_randlc * x1_randlc;
										t2_randlc = (int)(r23 * t1_randlc);
										z_randlc = t1_randlc - t23 * t2_randlc;
										t3_randlc = t23 * z_randlc + a2_randlc * x2_randlc;
										t4_randlc = (int)(r46 * t3_randlc);
										t1 = t3_randlc - t46 * t4_randlc;

										t3 = (r46 * t1);
								}
								if (ik == 0) break;
								//t3 = randlc(&t2, t2);
								a_randlc = t2;
								t1_randlc = r23 * a_randlc;
								a1_randlc = (int)t1_randlc;
								a2_randlc = a_randlc - t23 * a1_randlc;

								t1_randlc = r23 * t2;
								x1_randlc = (int)t1_randlc;
								x2_randlc = t2 - t23 * x1_randlc;
								t1_randlc = a1_randlc * x2_randlc + a2_randlc * x1_randlc;
								t2_randlc = (int)(r23 * t1_randlc);
								z_randlc = t1_randlc - t23 * t2_randlc;
								t3_randlc = t23 * z_randlc + a2_randlc * x2_randlc;
								t4_randlc = (int)(r46 * t3_randlc);
								t2 = t3_randlc - t46 * t4_randlc;

								t3 = (r46 * t2);
								kk = ik;
						}
#ifdef DEBUG_ON_t1
						printf("k = %d: t1 = %f\n", k-1, t1);
#endif

						/*      Compute uniform pseudorandom numbers. */

#ifdef TIMERS_ENABLED
						if (TIMERS_ENABLED == TRUE) timer_start(3);
#endif
						//vranlc(2*NK, &t1, A, x-1);
						t1_randlc = r23 * A;
						a1_randlc = (int)t1_randlc;
						a2_randlc = A - t23 * a1_randlc;
						x_vranlc = t1;

						for (i_vranlc = 1; i_vranlc <= 2*NK; i_vranlc++) {

								t1_randlc = r23 * x_vranlc;
								x1_randlc = (int)t1_randlc;
								x2_randlc = x_vranlc - t23 * x1_randlc;
								t1_randlc = a1_randlc * x2_randlc + a2_randlc * x1_randlc;
								t2_randlc = (int)(r23 * t1_randlc);
								z_randlc = t1_randlc - t23 * t2_randlc;
								t3_randlc = t23 * z_randlc + a2_randlc * x2_randlc;
								t4_randlc = (int)(r46 * t3_randlc);
								x_vranlc = t3_randlc - t46 * t4_randlc;
								xx[i_vranlc-1][m] = r46 * x_vranlc;
						}   
						t1 = x_vranlc;

#ifdef TIMERS_ENABLED
						if (TIMERS_ENABLED == TRUE) timer_stop(3);
#endif
#ifdef DEBUG_ON_x	
						if( (3 <= k)&&(k <= 5) )
								for (i = 30; i < 40; i++) printf("x[%d][%d] = %f\n",k-1,i,x[i]); 
#endif

						/*
						   c       Compute Gaussian deviates by acceptance-rejection method and 
						   c       tally counts in concentric square annuli.  This loop is not 
						   c       vectorizable.
						 */
#ifdef TIMERS_ENABLED
						if (TIMERS_ENABLED == TRUE) timer_start(2);
#endif

						for ( i = 0; i < NK; i++) {
								x1 = 2.0 * xx[2*i][m] - 1.0;
								x2 = 2.0 * xx[2*i+1][m] - 1.0;
								t1 = pow2(x1) + pow2(x2);
								if (t1 <= 1.0) {
										t2 = sqrt(-2.0 * log(t1) / t1);
										t3 = (x1 * t2);				/* Xi */
										t4 = (x2 * t2);				/* Yi */
										l = max(fabs(t3), fabs(t4));
										//qq[l] += 1.0;				/* counts */
										if( l == 0 ) { qq0 += 1.0; }
										else if( l == 1 ) { qq1 += 1.0; }
										else if( l == 2 ) { qq2 += 1.0; }
										else if( l == 3 ) { qq3 += 1.0; }
										else if( l == 4 ) { qq4 += 1.0; }
										else if( l == 5 ) { qq5 += 1.0; }
										else if( l == 6 ) { qq6 += 1.0; }
										else if( l == 7 ) { qq7 += 1.0; }
										else if( l == 8 ) { qq8 += 1.0; }
										else { qq9 += 1.0; }
										sx = sx + t3;				/* sum of Xi */
										sy = sy + t4;				/* sum of Yi */
								}
						}
#ifdef TIMERS_ENABLED
						if (TIMERS_ENABLED == TRUE) timer_stop(2);
#endif
#ifdef DEBUG_ON_q
						printf("k = %d\n", k);
						for (i = 0; i <= NQ - 1; i++) printf("qq[%d] = %f\n",i,qq[i]);
#endif
						/*
						//#pragma omp critical
						{
						for (i = 0; i <= NQ - 1; i++) q[i] += qq[i];
						}
						 */
				}
		} /* end of parallel region */    
		q[0] = qq0;
		q[1] = qq1;
		q[2] = qq2;
		q[3] = qq3;
		q[4] = qq4;
		q[5] = qq5;
		q[6] = qq6;
		q[7] = qq7;
		q[8] = qq8;
		q[9] = qq9;

		for (i = 0; i <= NQ-1; i++) {
				gc = gc + q[i];
		}

		timer_stop(1);
		tm = timer_read(1);

		nit = 0;
		if (M == 24) {
		if((fabs((sx- (-3.247834652034740e3))/sx) <= EPSILON) &&
		(fabs((sy- (-6.958407078382297e3))/sy) <= EPSILON)) {
			verified = TRUE;
		}   
		} else if (M == 25) {
		if ((fabs((sx- (-2.863319731645753e3))/sx) <= EPSILON) &&
        (fabs((sy- (-6.320053679109499e3))/sy) <= EPSILON)) {
        	verified = TRUE;
		}   
		} else if (M == 28) {
		if ((fabs((sx- (-4.295875165629892e3))/sx) <= EPSILON) &&
        (fabs((sy- (-1.580732573678431e4))/sy) <= EPSILON)) {
        	verified = TRUE;
		}   
		} else if (M == 30) {
		if ((fabs((sx- (4.033815542441498e4))/sx) <= EPSILON) &&
        (fabs((sy- (-2.660669192809235e4))/sy) <= EPSILON)) {
        	verified = TRUE;
		}   
		} else if (M == 32) {
		if ((fabs((sx- (4.764367927995374e4))/sx) <= EPSILON) &&
        (fabs((sy- (-8.084072988043731e4))/sy) <= EPSILON)) {
        	verified = TRUE;
		}   
		}   

		Mops = pow(2.0, M+1)/tm/1000000.0;

		printf("EP Benchmark Results: \n"
						"Accelerator Elapsed Time = %10.4f\n"
						"N = 2^%5d\n"
						"No. Gaussian Pairs = %15.0f\n"
						"Sums = %25.15e %25.15e\n"
						"Counts:\n",
						tm, M, gc, sx, sy);
		for (i = 0; i  <= NQ-1; i++) {
				printf("%3d %15.0f\n", i, q[i]);
		}

		c_print_results("EP", CLASS, M+1, 0, 0, nit, nthreads,
						tm, Mops, 	
						"Random numbers generated",
						verified, NPBVERSION, COMPILETIME,
						CS1, CS2, CS3, CS4, CS5, CS6, CS7);

#ifdef TIMERS_ENABLED
		if (TIMERS_ENABLED == TRUE) {
				printf("Total time:     %f", timer_read(1));
				printf("Gaussian pairs: %f", timer_read(2));
				printf("Random numbers: %f", timer_read(3));
		}
#endif
		return 0;
}
