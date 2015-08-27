/*--------------------------------------------------------------------

  NAS Parallel Benchmarks 2.3 OpenMP C versions - FT

  This benchmark is an OpenMP C version of the NPB FT code.

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

Authors: D. Bailey
W. Saphir

OpenMP C version: S. Satoh

--------------------------------------------------------------------*/

#include "npb-C.h"

/* global variables */
#include "global.h"

/* function declarations */
static void compute_initial_conditions (float u0_r[NTOTAL],
				float u0_i[NTOTAL]);
static void ipow46 (float a, int exponent, float *result);
static void setup (void);
static void print_timers (void);
static void fft (int dir, float x1_r[NTOTAL], float x1_i[NTOTAL],
				float x2_r[NTOTAL], float x2_i[NTOTAL]);
static void fft_init (int n);
static void cfftz (int is, int m, int n, float x_r[NTOTAL],
				float x_i[NTOTAL], float y_r[NTOTAL], float y_i[NTOTAL],
				int di1, int di2);
static void fftz2 (int is, int l, int m, int n,
				float u_r[NX], float u_i[NX],
				float x_r[NTOTAL], float x_i[NTOTAL],
				float y_r[NTOTAL], float y_i[NTOTAL],
				int di1, int di2);
static int ilog2 (int n);
static void verify (int d1, int d2, int d3, int nt,
				boolean * verified, char *classT);

/*--------------------------------------------------------------------
  c FT benchmark
  c-------------------------------------------------------------------*/

		int
main (int argc, char **argv)
{

		/*c-------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		int i_main, ierr;

		/*------------------------------------------------------------------
		  c u0, u1, u2 are the main arrays in the problem. 
		  c Depending on the decomposition, these arrays will have different 
		  c dimensions. To accomodate all possibilities, we allocate them as 
		  c one-dimensional arrays and pass them to subroutines for different 
		  c views
		  c  - u0 contains the initial (transformed) initial condition
		  c  - u1 and u2 are working arrays
		  c  - indexmap maps i,j,k of u0 to the correct i^2+j^2+k^2 for the
		  c    time evolution operator. 
		  c-----------------------------------------------------------------*/

		/*--------------------------------------------------------------------
		  c Large arrays are in common so that they are allocated on the
		  c heap rather than the stack. This common block is not
		  c referenced directly anywhere else. Padding is to avoid accidental 
		  c cache problems, since all array sizes are powers of two.
		  c-------------------------------------------------------------------*/
		static float u0_r[NTOTAL];		//u0_r[NZ][NY][NX];
		static float u0_i[NTOTAL];		//u0_i[NZ][NY][NX];
		static float u1_r[NTOTAL];		//u1_r[NZ][NY][NX];
		static float u1_i[NTOTAL];		//u1_i[NZ][NY][NX];
		static float u2_r[NTOTAL];		//u2_r[NZ][NY][NX];
		static float u2_i[NTOTAL];		//u2_i[NZ][NY][NX];
		static int indexmap[NTOTAL];		//indexmap[NZ][NY][NX];

		int iter;
		int nthreads = 1;
		double total_time, mflops;
		boolean verified;
		char classT;

		//////////////////////////////////
		// Used for compute_indexmap(). //
		//////////////////////////////////
		int i, j, k, ii, ii2, jj, ij2, kk;
		int m;
		float ap;

		////////////////////////
		// Used for evolve(). //
		////////////////////////
		//int i, j, k;

		//////////////////////////
		// Used for checksum(). //
		//////////////////////////
		//int m, j, 
		int q, r, s;
		float chk_r, chk_i;

		/////////////////////
		// Used for fft(). //
		/////////////////////
		int dir;
		static float y0_r[NTOTAL];
		static float y0_i[NTOTAL];
		static float y1_r[NTOTAL];
		static float y1_i[NTOTAL];
		int logNX, logNY, logNZ;

		///////////////////////
		//Used for cffts1(). //
		///////////////////////
		//int i, j, k, jj, m;
		int id;
		int is;
		///////////////////////
		// Used for cfftz(). //
		///////////////////////
		int l;

		///////////////////////
		// Used for fftz2(). //
		///////////////////////
		int k_fftz2, n1, li, lj, lk, ku, i_fftz2, i11, i12, i21, i22;
		float u1_rf, x11_r, x21_r;
		float u1_if, x11_i, x21_i;

		int idx, p, nn;
		float x11real, x11imag, x21real, x21imag;

		/*--------------------------------------------------------------------
		  c Run the entire problem once to make sure all data is touched. 
		  c This reduces variable startup costs, which is important for such a 
		  c short benchmark. The other NPB 2 implementations are similar. 
		  c-------------------------------------------------------------------*/
		for (i_main = 0; i_main < T_MAX; i_main++)
		{
				timer_clear (i_main);
		}
		setup ();
		{
				//compute_indexmap (indexmap);
				/*--------------------------------------------------------------------
				  c compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
				  c for time evolution exponent. 
				  c-------------------------------------------------------------------*/


				/*--------------------------------------------------------------------
				  c basically we want to convert the fortran indices 
				  c   1 2 3 4 5 6 7 8 
				  c to 
				  c   0 1 2 3 -4 -3 -2 -1
				  c The following magic formula does the trick:
				  c mod(i-1+n/2, n) - n/2
				  c-------------------------------------------------------------------*/

#pragma acc kernels loop gang worker independent
				for (m = 0; m < NTOTAL; m++)
				{
						i = m % NX;
						k = m / NX;
						j = k % NY;
						k = k / NY;

						ii = (i + NX / 2) % NX - NX / 2;
						ii2 = ii * ii;
						jj = (j + NY / 2) % NY - NY / 2;
						ij2 = jj * jj + ii2;
						kk = (k + NZ / 2) % NZ - NZ / 2;

						indexmap[m] = kk * kk + ij2;
				}

				/*--------------------------------------------------------------------
				  c compute array of exponentials for time evolution. 
				  c-------------------------------------------------------------------*/
				{
						ap = -4.0F * ALPHA * PI * PI;

						ex[0] = 1.0F;
						ex[1] = exp (ap);
						for (i = 2; i <= EXPMAX; i++)
						{
								ex[i] = ex[i - 1] * ex[1];
						}
				}				/* end single */
				{
						compute_initial_conditions (u1_r, u1_i);
						fft_init (dims[0][0]);
				}
				fft (1, u1_r, u1_i, u0_r, u0_i);
		}				/* end parallel */

		/*--------------------------------------------------------------------
		  c Start over from the beginning. Note that all operations must
		  c be timed, in contrast to other benchmarks. 
		  c-------------------------------------------------------------------*/
		for (i_main = 0; i_main < T_MAX; i_main++)
		{
				timer_clear (i_main);
		}

		timer_start (T_TOTAL);
		if (TIMERS_ENABLED == TRUE)
				timer_start (T_SETUP);

//#pragma omp parallel private(iter) firstprivate(niter)
#pragma acc data \
				create(ex[0:EXPMAX+1]) \
                create(indexmap[0:NTOTAL]) \
				create(u_r[0:NX], u_i[0:NX]) \
				create(u1_r[0:NTOTAL], u1_i[0:NTOTAL]) \
				create(u0_r[0:NTOTAL], u0_i[0:NTOTAL]) \
			    create(u2_r[0:NTOTAL], u2_i[0:NTOTAL]) \
				create(y0_r[0:NTOTAL], y0_i[0:NTOTAL]) \
				create(y1_r[0:NTOTAL], y1_i[0:NTOTAL])
		{
				//compute_indexmap (indexmap);
				/*--------------------------------------------------------------------
				  c compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
				  c for time evolution exponent. 
				  c-------------------------------------------------------------------*/

				/*--------------------------------------------------------------------
				  c basically we want to convert the fortran indices 
				  c   1 2 3 4 5 6 7 8 
				  c to 
				  c   0 1 2 3 -4 -3 -2 -1
				  c The following magic formula does the trick:
				  c mod(i-1+n/2, n) - n/2
				  c-------------------------------------------------------------------*/

#pragma acc kernels loop gang worker independent 
				for (m = 0; m < NTOTAL; m++)
				{
						i = m % NX;
						k = m / NX;
						j = k % NY;
						k = k / NY;

						ii = (i + NX / 2) % NX - NX / 2;
						ii2 = ii * ii;
						jj = (j + NY / 2) % NY - NY / 2;
						ij2 = jj * jj + ii2;
						kk = (k + NZ / 2) % NZ - NZ / 2;

						indexmap[m] = kk * kk + ij2;
				}

				/*--------------------------------------------------------------------
				  c compute array of exponentials for time evolution. 
				  c-------------------------------------------------------------------*/
				{
						ap = -4.0F * ALPHA * PI * PI;

						ex[0] = 1.0F;
						ex[1] = exp (ap);
						for (i = 2; i <= EXPMAX; i++)
						{
								ex[i] = ex[i - 1] * ex[1];
						}
				}				/* end single */

#pragma acc update device(ex[0:EXPMAX+1])

				{
						compute_initial_conditions (u1_r, u1_i);

						fft_init (dims[0][0]);
				}

#pragma acc update device(u_r[0:NX], u_i[0:NX], \
				u1_r[0:NTOTAL], u1_i[0:NTOTAL]) 

				if (TIMERS_ENABLED == TRUE)
				{
						timer_stop (T_SETUP);
				}
				if (TIMERS_ENABLED == TRUE)
				{
						timer_start (T_FFT);
				}
				//fft (1, u1_r, u1_i, u0_r, u0_i);
				//START_FFT//
				dir = 1;
				logNX = ilog2(NX);	
				logNY = ilog2(NY);	
				logNZ = ilog2(NZ);	

				/*--------------------------------------------------------------------
				  c note: args u1, x2 must be different arrays
				  c note: args for cfftsx are (direction, layout, xin, xout, scratch)
				  c       xin/xout may be the same and it can be somewhat faster
				  c       if they are
				  c-------------------------------------------------------------------*/

				{
						if (dir == 1)
						{
								//cffts1 (1, logNX, u1_r, u1_i, u1_r, u1_i, y0_r, y0_i, y1_r, y1_i);	/* u1 -> u1 */
								is = 1;

#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = i * NZ * NY + j * NZ + k;

										y0_r[id] = u1_r[m];
										y0_i[id] = u1_i[m];
								}

								//cfftz (is, logNX, NX, y0_r, y0_i, y1_r, y1_i, NZ, NY);
								/*--------------------------------------------------------------------
								  c   Perform one variant of the Stockham FFT.
								  c-------------------------------------------------------------------*/
								for (l = 1; l <= logNX; l += 2)
								{
										//fftz2 (is, l, logNX, NX, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NZ, NY);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NX; idx++)
										{

												n1 = NX / 2;
												if (l - 1 == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l - 1) - 1);
												}

												if (logNX - l == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNX - l) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y0_r[i11 + p];
																x11imag = y0_i[i11 + p];
																x21real = y0_r[i12 + p];
																x21imag = y0_i[i12 + p];
																y1_r[i21 + p] = x11real + x21real;
																y1_i[i21 + p] = x11imag + x21imag;
																y1_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y1_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
										if (l == logNX)
												break;
										//fftz2 (is, l + 1, logNX, NX, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NZ, NY);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NX; idx++)
										{

												n1 = NX / 2;
												if (l == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l) - 1);
												}

												if (logNX - (l+1) == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNX - (l+1)) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y1_r[i11 + p];
																x11imag = y1_i[i11 + p];
																x21real = y1_r[i12 + p];
																x21imag = y1_i[i12 + p];
																y0_r[i21 + p] = x11real + x21real;
																y0_i[i21 + p] = x11imag + x21imag;
																y0_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y0_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
								}

								if (logNX % 2 == 1)
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = i * NZ * NY + j * NZ + k;

												u1_r[m] = y1_r[id];
												u1_i[m] = y1_i[id];
										}
								}
								else
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = i * NZ * NY + j * NZ + k;

												u1_r[m] = y0_r[id];
												u1_i[m] = y0_i[id];
										}
								}

								//cffts2 (1, logNY, u1_r, u1_i, u1_r, u1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
								is = 1;

#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = j * NX * NZ + k * NX + i;

										y0_r[id] = u1_r[m];
										y0_i[id] = u1_i[m];
								}

								//cfftz (is, logNY, NY, y0_r, y0_i, y1_r, y1_i, NX, NZ);
								/*--------------------------------------------------------------------
								  c   Perform one variant of the Stockham FFT.
								  c-------------------------------------------------------------------*/
								for (l = 1; l <= logNY; l += 2)
								{
										//fftz2 (is, l, logNY, NY, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NX, NZ);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NY; idx++)
										{

												n1 = NY / 2;
												if (l - 1 == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l - 1) - 1);
												}

												if (logNY - l == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNY - l) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y0_r[i11 + p];
																x11imag = y0_i[i11 + p];
																x21real = y0_r[i12 + p];
																x21imag = y0_i[i12 + p];
																y1_r[i21 + p] = x11real + x21real;
																y1_i[i21 + p] = x11imag + x21imag;
																y1_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y1_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
										if (l == logNY)
												break;
										//fftz2 (is, l + 1, logNY, NY, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NX, NZ);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NY; idx++)
										{

												n1 = NY / 2;
												if (l == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l) - 1);
												}

												if (logNY - (l+1) == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNY - (l+1)) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y1_r[i11 + p];
																x11imag = y1_i[i11 + p];
																x21real = y1_r[i12 + p];
																x21imag = y1_i[i12 + p];
																y0_r[i21 + p] = x11real + x21real;
																y0_i[i21 + p] = x11imag + x21imag;
																y0_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y0_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
								}

								if (logNY % 2 == 1)
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = j * NX * NZ + k * NX + i;

												u1_r[m] = y1_r[id];
												u1_i[m] = y1_i[id];
										}
								}
								else
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = j * NX * NZ + k * NX + i;

												u1_r[m] = y0_r[id];
												u1_i[m] = y0_i[id];
										}
								}

								//cffts3 (1, logNZ, u1_r, u1_i, u0_r, u0_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x2 */
								is = 1;

								//cfftz (is, logNZ, NZ, u1_r, u1_i, y1_r, y1_i, NX, NY);
								/*--------------------------------------------------------------------
								  c   Perform one variant of the Stockham FFT.
								  c-------------------------------------------------------------------*/
								for (l = 1; l <= logNZ; l += 2)
								{
										//fftz2 (is, l, logNZ, NZ, u_r, u_i, u1_r, u1_i, y1_r, y1_i, NX, NY);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NZ; idx++)
										{

												n1 = NZ / 2;
												if (l - 1 == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l - 1) - 1);
												}

												if (logNZ - l == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNZ - l) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = u1_r[i11 + p];
																x11imag = u1_i[i11 + p];
																x21real = u1_r[i12 + p];
																x21imag = u1_i[i12 + p];
																y1_r[i21 + p] = x11real + x21real;
																y1_i[i21 + p] = x11imag + x21imag;
																y1_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y1_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
										if (l == logNZ)
												break;
										//fftz2 (is, l + 1, logNZ, NZ, u_r, u_i, y1_r, y1_i, u1_r, u1_i, NX, NY);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NZ; idx++)
										{

												n1 = NZ / 2;
												if (l == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l) - 1);
												}

												if (logNZ - (l+1) == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNZ - (l+1)) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y1_r[i11 + p];
																x11imag = y1_i[i11 + p];
																x21real = y1_r[i12 + p];
																x21imag = y1_i[i12 + p];
																u1_r[i21 + p] = x11real + x21real;
																u1_i[i21 + p] = x11imag + x21imag;
																u1_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																u1_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
								}

								if (logNZ % 2 == 1)
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												u0_r[m] = y1_r[m];
												u0_i[m] = y1_i[m];
										}
								}
								else 
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												u0_r[m] = u1_r[m];
												u0_i[m] = u1_i[m];
										}
								}

						}
						else
						{
								//cffts3 (-1, logNZ, u1_r, u1_i, u1_r, u1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
								is = -1;

								//cfftz (is, logNZ, NZ, u1_r, u1_i, y1_r, y1_i, NX, NY);
								/*--------------------------------------------------------------------
								  c   Perform one variant of the Stockham FFT.
								  c-------------------------------------------------------------------*/
								for (l = 1; l <= logNZ; l += 2)
								{
										//fftz2 (is, l, logNZ, NZ, u_r, u_i, u1_r, u1_i, y1_r, y1_i, NX, NY);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NZ; idx++)
										{

												n1 = NZ / 2;
												if (l - 1 == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l - 1) - 1);
												}

												if (logNZ - l == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNZ - l) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = u1_r[i11 + p];
																x11imag = u1_i[i11 + p];
																x21real = u1_r[i12 + p];
																x21imag = u1_i[i12 + p];
																y1_r[i21 + p] = x11real + x21real;
																y1_i[i21 + p] = x11imag + x21imag;
																y1_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y1_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
										if (l == logNZ)
												break;
										//fftz2 (is, l + 1, logNZ, NZ, u_r, u_i, y1_r, y1_i, u1_r, u1_i, NX, NY);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NZ; idx++)
										{

												n1 = NZ / 2;
												if (l == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l) - 1);
												}

												if (logNZ - (l+1) == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNZ - (l+1)) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y1_r[i11 + p];
																x11imag = y1_i[i11 + p];
																x21real = y1_r[i12 + p];
																x21imag = y1_i[i12 + p];
																u1_r[i21 + p] = x11real + x21real;
																u1_i[i21 + p] = x11imag + x21imag;
																u1_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																u1_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
								}

								if (logNZ % 2 == 1)
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												u1_r[m] = y1_r[m];
												u1_i[m] = y1_i[m];
										}
								}
								else 
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												u1_r[m] = u1_r[m];
												u1_i[m] = u1_i[m];
										}
								}

								//cffts2 (-1, logNY, u1_r, u1_i, u1_r, u1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
								is = -1;
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = j * NX * NZ + k * NX + i;

										y0_r[id] = u1_r[m];
										y0_i[id] = u1_i[m];
								}

								//cfftz (is, logNY, NY, y0_r, y0_i, y1_r, y1_i, NX, NZ);
								/*--------------------------------------------------------------------
								  c   Perform one variant of the Stockham FFT.
								  c-------------------------------------------------------------------*/
								for (l = 1; l <= logNY; l += 2)
								{
										//fftz2 (is, l, logNY, NY, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NX, NZ);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NY; idx++)
										{

												n1 = NY / 2;
												if (l - 1 == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l - 1) - 1);
												}

												if (logNY - l == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNY - l) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y0_r[i11 + p];
																x11imag = y0_i[i11 + p];
																x21real = y0_r[i12 + p];
																x21imag = y0_i[i12 + p];
																y1_r[i21 + p] = x11real + x21real;
																y1_i[i21 + p] = x11imag + x21imag;
																y1_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y1_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
										if (l == logNY)
												break;
										//fftz2 (is, l + 1, logNY, NY, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NX, NZ);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NY; idx++)
										{

												n1 = NY / 2;
												if (l == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l) - 1);
												}

												if (logNY - (l+1) == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNY - (l+1)) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y1_r[i11 + p];
																x11imag = y1_i[i11 + p];
																x21real = y1_r[i12 + p];
																x21imag = y1_i[i12 + p];
																y0_r[i21 + p] = x11real + x21real;
																y0_i[i21 + p] = x11imag + x21imag;
																y0_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y0_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
								}

								if (logNY % 2 == 1)
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = j * NX * NZ + k * NX + i;

												u1_r[m] = y1_r[id];
												u1_i[m] = y1_i[id];
										}
								}
								else
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = j * NX * NZ + k * NX + i;

												u1_r[m] = y0_r[id];
												u1_i[m] = y0_i[id];
										}
								}

								//cffts1 (-1, logNX, u1_r, u1_i, u0_r, u0_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x2 */
								is = -1;

#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = i * NZ * NY + j * NZ + k;

										y0_r[id] = u1_r[m];
										y0_i[id] = u1_i[m];
								}

								//cfftz (is, logNX, NX, y0_r, y0_i, y1_r, y1_i, NZ, NY);
								/*--------------------------------------------------------------------
								  c   Perform one variant of the Stockham FFT.
								  c-------------------------------------------------------------------*/
								for (l = 1; l <= logNX; l += 2)
								{
										//fftz2 (is, l, logNX, NX, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NZ, NY);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NX; idx++)
										{

												n1 = NX / 2;
												if (l - 1 == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l - 1) - 1);
												}

												if (logNX - l == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNX - l) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y0_r[i11 + p];
																x11imag = y0_i[i11 + p];
																x21real = y0_r[i12 + p];
																x21imag = y0_i[i12 + p];
																y1_r[i21 + p] = x11real + x21real;
																y1_i[i21 + p] = x11imag + x21imag;
																y1_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y1_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
										if (l == logNX)
												break;
										//fftz2 (is, l + 1, logNX, NX, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NZ, NY);
										/*--------------------------------------------------------------------
										  c   Set initial parameters.
										  c-------------------------------------------------------------------*/
										nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
										for (idx = 0; idx < NTOTAL / NX; idx++)
										{

												n1 = NX / 2;
												if (l == 0)
												{
														lk = 1;
												}
												else
												{
														lk = 2 << ((l) - 1);
												}

												if (logNX - (l+1) == 0)
												{
														li = 1;
												}
												else
												{
														li = 2 << ((logNX - (l+1)) - 1);
												}

												lj = 2 * lk;
												ku = li;

												for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
												{
														i11 = idx + i_fftz2 * lk * nn;
														i12 = i11 + n1 * nn;
														i21 = idx + i_fftz2 * lj * nn;
														i22 = i21 + lk * nn;

														if (is >= 1)
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = u_i[ku + i_fftz2];
														}
														else
														{
																u1_rf = u_r[ku + i_fftz2];
																u1_if = -u_i[ku + i_fftz2];
														}

														/*--------------------------------------------------------------------
														  c   This loop is vectorizable.
														  c-------------------------------------------------------------------*/
														for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
														{
																p = k_fftz2 * nn;
																x11real = y1_r[i11 + p];
																x11imag = y1_i[i11 + p];
																x21real = y1_r[i12 + p];
																x21imag = y1_i[i12 + p];
																y0_r[i21 + p] = x11real + x21real;
																y0_i[i21 + p] = x11imag + x21imag;
																y0_r[i22 + p] =
																		u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																y0_i[i22 + p] =
																		u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
														}
												}
										}
								}

								if (logNX % 2 == 1)
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = i * NZ * NY + j * NZ + k;

												u0_r[m] = y1_r[id];
												u0_i[m] = y1_i[id];
										}
								}
								else
								{
#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = i * NZ * NY + j * NZ + k;

												u0_r[m] = y0_r[id];
												u0_i[m] = y0_i[id];
										}
								}
						}
				}
				//END_FFT//
				if (TIMERS_ENABLED == TRUE)
				{
						timer_stop (T_FFT);
				}

				for (iter = 1; iter <= niter; iter++)
				{
						if (TIMERS_ENABLED == TRUE)
						{
								timer_start (T_EVOLVE);
						}

						//evolve (u0_r, u0_i, u1_r, u1_i, iter, indexmap);
						/*--------------------------------------------------------------------
						  c evolve u0 -> u1 (iter time steps) in fourier space
						  c-------------------------------------------------------------------*/


#pragma acc kernels loop gang worker independent
						for (i = 0; i < NTOTAL; i++)
						{
								u1_r[i] = u0_r[i] * ex[iter * indexmap[i]];
								u1_i[i] = u0_i[i] * ex[iter * indexmap[i]];
						}

						if (TIMERS_ENABLED == TRUE)
						{
								timer_stop (T_EVOLVE);
						}
						if (TIMERS_ENABLED == TRUE)
						{
								timer_start (T_FFT);
						}

						//fft (-1, u1_r, u1_i, u2_r, u2_i);
						//START_FFT//
						dir = -1;
						logNX = ilog2(NX);	
						logNY = ilog2(NY);	
						logNZ = ilog2(NZ);	

						/*--------------------------------------------------------------------
						  c note: args x1, x2 must be different arrays
						  c note: args for cfftsx are (direction, layout, xin, xout, scratch)
						  c       xin/xout may be the same and it can be somewhat faster
						  c       if they are
						  c-------------------------------------------------------------------*/

						{
								if (dir == 1)
								{
										//cffts1 (1, logNX, u1_r, u1_i, u1_r, u1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
										is = 1;

#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = i * NZ * NY + j * NZ + k;

												y0_r[id] = u1_r[m];
												y0_i[id] = u1_i[m];
										}

										//cfftz (is, logNX, NX, y0_r, y0_i, y1_r, y1_i, NZ, NY);
										/*--------------------------------------------------------------------
										  c   Perform one variant of the Stockham FFT.
										  c-------------------------------------------------------------------*/
										for (l = 1; l <= logNX; l += 2)
										{
												//fftz2 (is, l, logNX, NX, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NZ, NY);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NX; idx++)
												{

														n1 = NX / 2;
														if (l - 1 == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l - 1) - 1);
														}

														if (logNX - l == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNX - l) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y0_r[i11 + p];
																		x11imag = y0_i[i11 + p];
																		x21real = y0_r[i12 + p];
																		x21imag = y0_i[i12 + p];
																		y1_r[i21 + p] = x11real + x21real;
																		y1_i[i21 + p] = x11imag + x21imag;
																		y1_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y1_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
												if (l == logNX)
														break;
												//fftz2 (is, l + 1, logNX, NX, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NZ, NY);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NX; idx++)
												{

														n1 = NX / 2;
														if (l == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l) - 1);
														}

														if (logNX - (l+1) == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNX - (l+1)) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y1_r[i11 + p];
																		x11imag = y1_i[i11 + p];
																		x21real = y1_r[i12 + p];
																		x21imag = y1_i[i12 + p];
																		y0_r[i21 + p] = x11real + x21real;
																		y0_i[i21 + p] = x11imag + x21imag;
																		y0_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y0_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
										}

										if (logNX % 2 == 1)
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														i = m % NX;
														k = m / NX;
														j = k % NY;
														k = k / NY;
														id = i * NZ * NY + j * NZ + k;

														u1_r[m] = y1_r[id];
														u1_i[m] = y1_i[id];
												}
										}
										else
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														i = m % NX;
														k = m / NX;
														j = k % NY;
														k = k / NY;
														id = i * NZ * NY + j * NZ + k;

														u1_r[m] = y0_r[id];
														u1_i[m] = y0_i[id];
												}
										}

										//cffts2 (1, logNY, u1_r, u1_i, u1_r, u1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
										is = 1;

#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = j * NX * NZ + k * NX + i;

												y0_r[id] = u1_r[m];
												y0_i[id] = u1_i[m];
										}

										//cfftz (is, logNY, NY, y0_r, y0_i, y1_r, y1_i, NX, NZ);
										/*--------------------------------------------------------------------
										  c   Perform one variant of the Stockham FFT.
										  c-------------------------------------------------------------------*/
										for (l = 1; l <= logNY; l += 2)
										{
												//fftz2 (is, l, logNY, NY, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NX, NZ);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NY; idx++)
												{

														n1 = NY / 2;
														if (l - 1 == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l - 1) - 1);
														}

														if (logNY - l == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNY - l) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y0_r[i11 + p];
																		x11imag = y0_i[i11 + p];
																		x21real = y0_r[i12 + p];
																		x21imag = y0_i[i12 + p];
																		y1_r[i21 + p] = x11real + x21real;
																		y1_i[i21 + p] = x11imag + x21imag;
																		y1_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y1_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
												if (l == logNY)
														break;
												//fftz2 (is, l + 1, logNY, NY, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NX, NZ);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NY; idx++)
												{

														n1 = NY / 2;
														if (l == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l) - 1);
														}

														if (logNY - (l+1) == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNY - (l+1)) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y1_r[i11 + p];
																		x11imag = y1_i[i11 + p];
																		x21real = y1_r[i12 + p];
																		x21imag = y1_i[i12 + p];
																		y0_r[i21 + p] = x11real + x21real;
																		y0_i[i21 + p] = x11imag + x21imag;
																		y0_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y0_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
										}

										if (logNY % 2 == 1)
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														i = m % NX;
														k = m / NX;
														j = k % NY;
														k = k / NY;
														id = j * NX * NZ + k * NX + i;

														u1_r[m] = y1_r[id];
														u1_i[m] = y1_i[id];
												}
										}
										else
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														i = m % NX;
														k = m / NX;
														j = k % NY;
														k = k / NY;
														id = j * NX * NZ + k * NX + i;

														u1_r[m] = y0_r[id];
														u1_i[m] = y0_i[id];
												}
										}

										//cffts3 (1, logNZ, u1_r, u1_i, u2_r, u2_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x2 */
										is = 1;

										//cfftz (is, logNZ, NZ, u1_r, u1_i, y1_r, y1_i, NX, NY);
										/*--------------------------------------------------------------------
										  c   Perform one variant of the Stockham FFT.
										  c-------------------------------------------------------------------*/
										for (l = 1; l <= logNZ; l += 2)
										{
												//fftz2 (is, l, logNZ, NZ, u_r, u_i, u1_r, u1_i, y1_r, y1_i, NX, NY);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NZ; idx++)
												{

														n1 = NZ / 2;
														if (l - 1 == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l - 1) - 1);
														}

														if (logNZ - l == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNZ - l) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = u1_r[i11 + p];
																		x11imag = u1_i[i11 + p];
																		x21real = u1_r[i12 + p];
																		x21imag = u1_i[i12 + p];
																		y1_r[i21 + p] = x11real + x21real;
																		y1_i[i21 + p] = x11imag + x21imag;
																		y1_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y1_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
												if (l == logNZ)
														break;
												//fftz2 (is, l + 1, logNZ, NZ, u_r, u_i, y1_r, y1_i, u1_r, u1_i, NX, NY);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NZ; idx++)
												{

														n1 = NZ / 2;
														if (l == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l) - 1);
														}

														if (logNZ - (l+1) == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNZ - (l+1)) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y1_r[i11 + p];
																		x11imag = y1_i[i11 + p];
																		x21real = y1_r[i12 + p];
																		x21imag = y1_i[i12 + p];
																		u1_r[i21 + p] = x11real + x21real;
																		u1_i[i21 + p] = x11imag + x21imag;
																		u1_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		u1_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
										}

										if (logNZ % 2 == 1)
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														u2_r[m] = y1_r[m];
														u2_i[m] = y1_i[m];
												}
										}
										else 
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														u2_r[m] = u1_r[m];
														u2_i[m] = u1_i[m];
												}
										}
								}
								else
								{
										//cffts3 (-1, logNZ, u1_r, u1_i, u1_r, u1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
										is = -1;

										//cfftz (is, logNZ, NZ, u1_r, u1_i, y1_r, y1_i, NX, NY);
										/*--------------------------------------------------------------------
										  c   Perform one variant of the Stockham FFT.
										  c-------------------------------------------------------------------*/
										for (l = 1; l <= logNZ; l += 2)
										{
												//fftz2 (is, l, logNZ, NZ, u_r, u_i, u1_r, u1_i, y1_r, y1_i, NX, NY);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NZ; idx++)
												{

														n1 = NZ / 2;
														if (l - 1 == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l - 1) - 1);
														}

														if (logNZ - l == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNZ - l) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = u1_r[i11 + p];
																		x11imag = u1_i[i11 + p];
																		x21real = u1_r[i12 + p];
																		x21imag = u1_i[i12 + p];
																		y1_r[i21 + p] = x11real + x21real;
																		y1_i[i21 + p] = x11imag + x21imag;
																		y1_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y1_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
												if (l == logNZ)
														break;
												//fftz2 (is, l + 1, logNZ, NZ, u_r, u_i, y1_r, y1_i, u1_r, u1_i, NX, NY);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NZ; idx++)
												{

														n1 = NZ / 2;
														if (l == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l) - 1);
														}

														if (logNZ - (l+1) == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNZ - (l+1)) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y1_r[i11 + p];
																		x11imag = y1_i[i11 + p];
																		x21real = y1_r[i12 + p];
																		x21imag = y1_i[i12 + p];
																		u1_r[i21 + p] = x11real + x21real;
																		u1_i[i21 + p] = x11imag + x21imag;
																		u1_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		u1_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
										}

										if (logNZ % 2 == 1)
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														u1_r[m] = y1_r[m];
														u1_i[m] = y1_i[m];
												}
										}
										else 
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														u1_r[m] = u1_r[m];
														u1_i[m] = u1_i[m];
												}
										}

										//cffts2 (-1, logNY, u1_r, u1_i, u1_r, u1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
										is = -1;

#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = j * NX * NZ + k * NX + i;

												y0_r[id] = u1_r[m];
												y0_i[id] = u1_i[m];
										}

										//cfftz (is, logNY, NY, y0_r, y0_i, y1_r, y1_i, NX, NZ);
										/*--------------------------------------------------------------------
										  c   Perform one variant of the Stockham FFT.
										  c-------------------------------------------------------------------*/
										for (l = 1; l <= logNY; l += 2)
										{
												//fftz2 (is, l, logNY, NY, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NX, NZ);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NY; idx++)
												{

														n1 = NY / 2;
														if (l - 1 == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l - 1) - 1);
														}

														if (logNY - l == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNY - l) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y0_r[i11 + p];
																		x11imag = y0_i[i11 + p];
																		x21real = y0_r[i12 + p];
																		x21imag = y0_i[i12 + p];
																		y1_r[i21 + p] = x11real + x21real;
																		y1_i[i21 + p] = x11imag + x21imag;
																		y1_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y1_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
												if (l == logNY)
														break;
												//fftz2 (is, l + 1, logNY, NY, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NX, NZ);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NY; idx++)
												{

														n1 = NY / 2;
														if (l == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l) - 1);
														}

														if (logNY - (l+1) == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNY - (l+1)) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y1_r[i11 + p];
																		x11imag = y1_i[i11 + p];
																		x21real = y1_r[i12 + p];
																		x21imag = y1_i[i12 + p];
																		y0_r[i21 + p] = x11real + x21real;
																		y0_i[i21 + p] = x11imag + x21imag;
																		y0_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y0_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
										}

										if (logNY % 2 == 1)
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														i = m % NX;
														k = m / NX;
														j = k % NY;
														k = k / NY;
														id = j * NX * NZ + k * NX + i;

														u1_r[m] = y1_r[id];
														u1_i[m] = y1_i[id];
												}
										}
										else
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														i = m % NX;
														k = m / NX;
														j = k % NY;
														k = k / NY;
														id = j * NX * NZ + k * NX + i;

														u1_r[m] = y0_r[id];
														u1_i[m] = y0_i[id];
												}
										}

										//cffts1 (-1, logNX, u1_r, u1_i, u2_r, u2_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x2 */
										is = -1;

#pragma acc kernels loop gang worker independent
										for (m = 0; m < NTOTAL; m++)
										{
												i = m % NX;
												k = m / NX;
												j = k % NY;
												k = k / NY;
												id = i * NZ * NY + j * NZ + k;

												y0_r[id] = u1_r[m];
												y0_i[id] = u1_i[m];
										}

										//cfftz (is, logNX, NX, y0_r, y0_i, y1_r, y1_i, NZ, NY);
										/*--------------------------------------------------------------------
										  c   Perform one variant of the Stockham FFT.
										  c-------------------------------------------------------------------*/
										for (l = 1; l <= logNX; l += 2)
										{
												//fftz2 (is, l, logNX, NX, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NZ, NY);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NX; idx++)
												{

														n1 = NX / 2;
														if (l - 1 == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l - 1) - 1);
														}

														if (logNX - l == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNX - l) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y0_r[i11 + p];
																		x11imag = y0_i[i11 + p];
																		x21real = y0_r[i12 + p];
																		x21imag = y0_i[i12 + p];
																		y1_r[i21 + p] = x11real + x21real;
																		y1_i[i21 + p] = x11imag + x21imag;
																		y1_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y1_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
												if (l == logNX)
														break;
												//fftz2 (is, l + 1, logNX, NX, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NZ, NY);
												/*--------------------------------------------------------------------
												  c   Set initial parameters.
												  c-------------------------------------------------------------------*/
												nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
												for (idx = 0; idx < NTOTAL / NX; idx++)
												{

														n1 = NX / 2;
														if (l == 0)
														{
																lk = 1;
														}
														else
														{
																lk = 2 << ((l) - 1);
														}

														if (logNX - (l+1) == 0)
														{
																li = 1;
														}
														else
														{
																li = 2 << ((logNX - (l+1)) - 1);
														}

														lj = 2 * lk;
														ku = li;

														for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
														{
																i11 = idx + i_fftz2 * lk * nn;
																i12 = i11 + n1 * nn;
																i21 = idx + i_fftz2 * lj * nn;
																i22 = i21 + lk * nn;

																if (is >= 1)
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = u_i[ku + i_fftz2];
																}
																else
																{
																		u1_rf = u_r[ku + i_fftz2];
																		u1_if = -u_i[ku + i_fftz2];
																}

																/*--------------------------------------------------------------------
																  c   This loop is vectorizable.
																  c-------------------------------------------------------------------*/
																for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
																{
																		p = k_fftz2 * nn;
																		x11real = y1_r[i11 + p];
																		x11imag = y1_i[i11 + p];
																		x21real = y1_r[i12 + p];
																		x21imag = y1_i[i12 + p];
																		y0_r[i21 + p] = x11real + x21real;
																		y0_i[i21 + p] = x11imag + x21imag;
																		y0_r[i22 + p] =
																				u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
																		y0_i[i22 + p] =
																				u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
																}
														}
												}
										}

										if (logNX % 2 == 1)
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														i = m % NX;
														k = m / NX;
														j = k % NY;
														k = k / NY;
														id = i * NZ * NY + j * NZ + k;

														u2_r[m] = y1_r[id];
														u2_i[m] = y1_i[id];
												}
										}
										else
										{
#pragma acc kernels loop gang worker independent
												for (m = 0; m < NTOTAL; m++)
												{
														i = m % NX;
														k = m / NX;
														j = k % NY;
														k = k / NY;
														id = i * NZ * NY + j * NZ + k;

														u2_r[m] = y0_r[id];
														u2_i[m] = y0_i[id];
												}
										}
								}
						}
						//END_FFT//

						if (TIMERS_ENABLED == TRUE)
						{
								timer_stop (T_FFT);
						}
						if (TIMERS_ENABLED == TRUE)
						{
								timer_start (T_CHECKSUM);
						}

						//checksum (iter, u2_r, u2_i, dims[0]);
						chk_r = 0.0F;
						chk_i = 0.0F;

#pragma acc kernels loop gang worker independent
						for(m = 0; m < 1024; m++){
								j = 1+m;
								q = j%NX;
								if (q >= 0 && q < NX) {
										r = (3*j)%NY;
										if (r >= 0 && r < NY) {
												s = (5*j)%NZ;
												if (s >= 0 && s < NZ) {
														chk_r = chk_r + u2_r[s*NY*NX + r*NX + q];
														chk_i = chk_i + u2_i[s*NY*NX + r*NX + q];
												}
										}
								}
						}
						//printf("chk_r = %22.12e, chk_i =%22.12e\n", chk_r, chk_i);
						{
								sums_r[iter] += chk_r;
								sums_i[iter] += chk_i;
						}
						{
								sums_r[iter] = sums_r[iter] / (float) (NTOTAL);
								sums_i[iter] = sums_i[iter] / (float) (NTOTAL);

								printf ("T = %5d     Checksum = %22.12e %22.12e\n",
												iter, sums_r[iter], sums_i[iter]);
						}

						if (TIMERS_ENABLED == TRUE)
						{
								timer_stop (T_CHECKSUM);
						}
				}

				verify (NX, NY, NZ, niter, &verified, &classT);

#if defined(_OPENMP)
				nthreads = omp_get_num_threads ();
#endif /* _OPENMP */
		}				/* end parallel */

		timer_stop (T_TOTAL);
		total_time = timer_read (T_TOTAL);

		if (total_time != 0.0)
		{
				mflops = 1.0e-6 * (double) (NTOTAL) *
						(14.8157 + 7.19641 * log ((double) (NTOTAL))
						 + (5.23518 + 7.21113 * log ((double) (NTOTAL))) * niter)
						/ total_time;
		}
		else
		{
				mflops = 0.0;
		}
		c_print_results ("FT", classT, NX, NY, NZ, niter, nthreads,
						total_time, mflops, "          floating point", verified,
						NPBVERSION, COMPILETIME,
						CS1, CS2, CS3, CS4, CS5, CS6, CS7);
		if (TIMERS_ENABLED == TRUE)
				print_timers ();

	return 0;
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

		static void
compute_initial_conditions (float u0_r[NTOTAL], float u0_i[NTOTAL])
{

		/*--------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		/*--------------------------------------------------------------------
		  c Fill in array u0 with initial conditions from 
		  c random number generator 
		  c-------------------------------------------------------------------*/

		int k;
		float x0, start, an, dummy;
		static float tmp[NX * 2 * MAXDIM + 1];
		int i, j, t;

		start = SEED;
		/*--------------------------------------------------------------------
		  c Jump to the starting element for our first plane.
		  c-------------------------------------------------------------------*/
		ipow46 (A, (zstart[0] - 1) * 2 * NX * NY + (ystart[0] - 1) * 2 * NX, &an);
		dummy = randlc (&start, an);
		ipow46 (A, 2 * NX * NY, &an);

		/*--------------------------------------------------------------------
		  c Go through by z planes filling in one square at a time.
		  c-------------------------------------------------------------------*/
		for (k = 0; k < NZ; k++)
		{
				x0 = start;
				vranlc (2 * NX * NY, &x0, A, tmp);

				t = 1;
				for (j = 0; j < NY; j++)
						for (i = 0; i < NX; i++)
						{
								u0_r[k * NY * NX + j * NX + i] = tmp[t++];
								u0_i[k * NY * NX + j * NX + i] = tmp[t++];
						}

				if (k != NZ)
						dummy = randlc (&start, an);
		}
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

		static void
ipow46 (float a, int exponent, float *result)
{

		/*--------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		/*--------------------------------------------------------------------
		  c compute a^exponent mod 2^46
		  c-------------------------------------------------------------------*/

		float dummy, q, r;
		int n, n2;

		/*--------------------------------------------------------------------
		  c Use
		  c   a^n = a^(n/2)*a^(n/2) if n even else
		  c   a^n = a*a^(n-1)       if n odd
		  c-------------------------------------------------------------------*/
		*result = 1;
		if (exponent == 0)
				return;
		q = a;
		r = 1;
		n = exponent;

		while (n > 1)
		{
				n2 = n / 2;
				if (n2 * 2 == n)
				{
						dummy = randlc (&q, q);
						n = n2;
				}
				else
				{
						dummy = randlc (&r, q);
						n = n - 1;
				}
		}
		dummy = randlc (&r, q);
		*result = r;
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

		static void
setup (void)
{

		/*--------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		int ierr, i, j, fstatus;

		printf ("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"
						" - FT Benchmark\n\n");

		niter = NITER_DEFAULT;

		printf (" Size                : %3dx%3dx%3d\n", NX, NY, NZ);
		printf (" Iterations          :     %7d\n", niter);

		/* 1004 format(' Number of processes :     ', i7)
		   1005 format(' Processor array     :     ', i3, 'x', i3)
		   1006 format(' WARNING: compiled for ', i5, ' processes. ',
		   >       ' Will not verify. ')*/

		for (i = 0; i < 3; i++)
		{
				dims[i][0] = NX;
				dims[i][1] = NY;
				dims[i][2] = NZ;
		}


		for (i = 0; i < 3; i++)
		{
				xstart[i] = 1;
				xend[i] = NX;
				ystart[i] = 1;
				yend[i] = NY;
				zstart[i] = 1;
				zend[i] = NZ;
		}

		/*--------------------------------------------------------------------
		  c Set up info for blocking of ffts and transposes.  This improves
		  c performance on cache-based systems. Blocking involves
		  c working on a chunk of the problem at a time, taking chunks
		  c along the first, second, or third dimension. 
		  c
		  c - In cffts1 blocking is on 2nd dimension (with fft on 1st dim)
		  c - In cffts2/3 blocking is on 1st dimension (with fft on 2nd and 3rd dims)

		  c Since 1st dim is always in processor, we'll assume it's long enough 
		  c (default blocking factor is 16 so min size for 1st dim is 16)
		  c The only case we have to worry about is cffts1 in a 2d decomposition. 
		  c so the blocking factor should not be larger than the 2nd dimension. 
		  c-------------------------------------------------------------------*/

		fftblock = FFTBLOCK_DEFAULT;
		fftblockpad = FFTBLOCKPAD_DEFAULT;

		if (fftblock != FFTBLOCK_DEFAULT)
				fftblockpad = fftblock + 3;
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

		static void
print_timers (void)
{

		/*--------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		int i;
		char *tstrings[] = { "          total ",
				"          setup ",
				"            fft ",
				"         evolve ",
				"       checksum ",
				"         fftlow ",
				"        fftcopy "
		};

		for (i = 0; i < T_MAX; i++)
		{
				if (timer_read (i) != 0.0)
				{
						printf ("timer %2d(%16s( :%10.6f\n", i, tstrings[i],
										timer_read (i));
				}
		}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

		static void
fft (int dir, float x1_r[NTOTAL], float x1_i[NTOTAL],
				float x2_r[NTOTAL], float x2_i[NTOTAL])
{

		/*--------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		static float y0_r[NTOTAL];
		static float y0_i[NTOTAL];
		static float y1_r[NTOTAL];
		static float y1_i[NTOTAL];
		int logNX, logNY, logNZ;

		///////////////////////
		//Used for cffts1(). //
		///////////////////////
		int i, j, k, jj;
		int m, id;
		int is;
		///////////////////////
		// Used for cfftz(). //
		///////////////////////
		int l;

		///////////////////////
		// Used for fftz2(). //
		///////////////////////
		int k_fftz2, n1, li, lj, lk, ku, i_fftz2, i11, i12, i21, i22;
		float u1_rf, x11_r, x21_r;
		float u1_if, x11_i, x21_i;

		int idx, p, nn;
		float x11real, x11imag, x21real, x21imag;

		//START_FFT//
		logNX = ilog2(NX);	
		logNY = ilog2(NY);	
		logNZ = ilog2(NZ);	

		/*--------------------------------------------------------------------
		  c note: args x1, x2 must be different arrays
		  c note: args for cfftsx are (direction, layout, xin, xout, scratch)
		  c       xin/xout may be the same and it can be somewhat faster
		  c       if they are
		  c-------------------------------------------------------------------*/

#pragma acc data \
		copyin(u_r[0:NX], u_i[0:NX]) \
		copy(x1_r[0:NTOTAL], x1_i[0:NTOTAL]) \
		copyout(x2_r[0:NTOTAL], x2_i[0:NTOTAL]) \
		create(y0_r[0:NTOTAL], y0_i[0:NTOTAL]) \
		create(y1_r[0:NTOTAL], y1_i[0:NTOTAL])
		{
				if (dir == 1)
				{
						//cffts1 (1, logNX, x1_r, x1_i, x1_r, x1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
						is = 1;

#pragma acc kernels loop gang worker independent
						for (m = 0; m < NTOTAL; m++)
						{
								i = m % NX;
								k = m / NX;
								j = k % NY;
								k = k / NY;
								id = i * NZ * NY + j * NZ + k;

								y0_r[id] = x1_r[m];
								y0_i[id] = x1_i[m];
						}

						//cfftz (is, logNX, NX, y0_r, y0_i, y1_r, y1_i, NZ, NY);
						/*--------------------------------------------------------------------
						  c   Perform one variant of the Stockham FFT.
						  c-------------------------------------------------------------------*/
						for (l = 1; l <= logNX; l += 2)
						{
								//fftz2 (is, l, logNX, NX, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NZ, NY);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NX; idx++)
								{

										n1 = NX / 2;
										if (l - 1 == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l - 1) - 1);
										}

										if (logNX - l == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNX - l) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y0_r[i11 + p];
														x11imag = y0_i[i11 + p];
														x21real = y0_r[i12 + p];
														x21imag = y0_i[i12 + p];
														y1_r[i21 + p] = x11real + x21real;
														y1_i[i21 + p] = x11imag + x21imag;
														y1_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y1_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
								if (l == logNX)
										break;
								//fftz2 (is, l + 1, logNX, NX, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NZ, NY);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NX; idx++)
								{

										n1 = NX / 2;
										if (l == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l) - 1);
										}

										if (logNX - (l+1) == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNX - (l+1)) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y1_r[i11 + p];
														x11imag = y1_i[i11 + p];
														x21real = y1_r[i12 + p];
														x21imag = y1_i[i12 + p];
														y0_r[i21 + p] = x11real + x21real;
														y0_i[i21 + p] = x11imag + x21imag;
														y0_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y0_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
						}

						if (logNX % 2 == 1)
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = i * NZ * NY + j * NZ + k;

										x1_r[m] = y1_r[id];
										x1_i[m] = y1_i[id];
								}
						}
						else
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = i * NZ * NY + j * NZ + k;

										x1_r[m] = y0_r[id];
										x1_i[m] = y0_i[id];
								}
						}

						//cffts2 (1, logNY, x1_r, x1_i, x1_r, x1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
						is = 1;

#pragma acc kernels loop gang worker independent
						for (m = 0; m < NTOTAL; m++)
						{
								i = m % NX;
								k = m / NX;
								j = k % NY;
								k = k / NY;
								id = j * NX * NZ + k * NX + i;

								y0_r[id] = x1_r[m];
								y0_i[id] = x1_i[m];
						}

						//cfftz (is, logNY, NY, y0_r, y0_i, y1_r, y1_i, NX, NZ);
						/*--------------------------------------------------------------------
						  c   Perform one variant of the Stockham FFT.
						  c-------------------------------------------------------------------*/
						for (l = 1; l <= logNY; l += 2)
						{
								//fftz2 (is, l, logNY, NY, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NX, NZ);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NY; idx++)
								{

										n1 = NY / 2;
										if (l - 1 == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l - 1) - 1);
										}

										if (logNY - l == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNY - l) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y0_r[i11 + p];
														x11imag = y0_i[i11 + p];
														x21real = y0_r[i12 + p];
														x21imag = y0_i[i12 + p];
														y1_r[i21 + p] = x11real + x21real;
														y1_i[i21 + p] = x11imag + x21imag;
														y1_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y1_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
								if (l == logNY)
										break;
								//fftz2 (is, l + 1, logNY, NY, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NX, NZ);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NY; idx++)
								{

										n1 = NY / 2;
										if (l == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l) - 1);
										}

										if (logNY - (l+1) == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNY - (l+1)) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y1_r[i11 + p];
														x11imag = y1_i[i11 + p];
														x21real = y1_r[i12 + p];
														x21imag = y1_i[i12 + p];
														y0_r[i21 + p] = x11real + x21real;
														y0_i[i21 + p] = x11imag + x21imag;
														y0_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y0_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
						}

						if (logNY % 2 == 1)
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = j * NX * NZ + k * NX + i;

										x1_r[m] = y1_r[id];
										x1_i[m] = y1_i[id];
								}
						}
						else
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = j * NX * NZ + k * NX + i;

										x1_r[m] = y0_r[id];
										x1_i[m] = y0_i[id];
								}
						}

						//cffts3 (1, logNZ, x1_r, x1_i, x2_r, x2_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x2 */
						is = 1;

						//cfftz (is, logNZ, NZ, x1_r, x1_i, y1_r, y1_i, NX, NY);
						/*--------------------------------------------------------------------
						  c   Perform one variant of the Stockham FFT.
						  c-------------------------------------------------------------------*/
						for (l = 1; l <= logNZ; l += 2)
						{
								//fftz2 (is, l, logNZ, NZ, u_r, u_i, x1_r, x1_i, y1_r, y1_i, NX, NY);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NZ; idx++)
								{

										n1 = NZ / 2;
										if (l - 1 == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l - 1) - 1);
										}

										if (logNZ - l == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNZ - l) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = x1_r[i11 + p];
														x11imag = x1_i[i11 + p];
														x21real = x1_r[i12 + p];
														x21imag = x1_i[i12 + p];
														y1_r[i21 + p] = x11real + x21real;
														y1_i[i21 + p] = x11imag + x21imag;
														y1_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y1_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
								if (l == logNZ)
										break;
								//fftz2 (is, l + 1, logNZ, NZ, u_r, u_i, y1_r, y1_i, x1_r, x1_i, NX, NY);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NZ; idx++)
								{

										n1 = NZ / 2;
										if (l == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l) - 1);
										}

										if (logNZ - (l+1) == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNZ - (l+1)) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y1_r[i11 + p];
														x11imag = y1_i[i11 + p];
														x21real = y1_r[i12 + p];
														x21imag = y1_i[i12 + p];
														x1_r[i21 + p] = x11real + x21real;
														x1_i[i21 + p] = x11imag + x21imag;
														x1_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														x1_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
						}

						if (logNZ % 2 == 1)
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										x2_r[m] = y1_r[m];
										x2_i[m] = y1_i[m];
								}
						}
						else 
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										x2_r[m] = x1_r[m];
										x2_i[m] = x1_i[m];
								}
						}
				}
				else
				{
						//cffts3 (-1, logNZ, x1_r, x1_i, x1_r, x1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
						is = -1;

						//cfftz (is, logNZ, NZ, x1_r, x1_i, y1_r, y1_i, NX, NY);
						/*--------------------------------------------------------------------
						  c   Perform one variant of the Stockham FFT.
						  c-------------------------------------------------------------------*/
						for (l = 1; l <= logNZ; l += 2)
						{
								//fftz2 (is, l, logNZ, NZ, u_r, u_i, x1_r, x1_i, y1_r, y1_i, NX, NY);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NZ; idx++)
								{

										n1 = NZ / 2;
										if (l - 1 == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l - 1) - 1);
										}

										if (logNZ - l == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNZ - l) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = x1_r[i11 + p];
														x11imag = x1_i[i11 + p];
														x21real = x1_r[i12 + p];
														x21imag = x1_i[i12 + p];
														y1_r[i21 + p] = x11real + x21real;
														y1_i[i21 + p] = x11imag + x21imag;
														y1_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y1_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
								if (l == logNZ)
										break;
								//fftz2 (is, l + 1, logNZ, NZ, u_r, u_i, y1_r, y1_i, x1_r, x1_i, NX, NY);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NX * NY;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NZ; idx++)
								{

										n1 = NZ / 2;
										if (l == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l) - 1);
										}

										if (logNZ - (l+1) == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNZ - (l+1)) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y1_r[i11 + p];
														x11imag = y1_i[i11 + p];
														x21real = y1_r[i12 + p];
														x21imag = y1_i[i12 + p];
														x1_r[i21 + p] = x11real + x21real;
														x1_i[i21 + p] = x11imag + x21imag;
														x1_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														x1_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
						}

						if (logNZ % 2 == 1)
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										x1_r[m] = y1_r[m];
										x1_i[m] = y1_i[m];
								}
						}
						else 
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										x1_r[m] = x1_r[m];
										x1_i[m] = x1_i[m];
								}
						}

						//cffts2 (-1, logNY, x1_r, x1_i, x1_r, x1_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x1 */
						is = -1;

#pragma acc kernels loop gang worker independent
						for (m = 0; m < NTOTAL; m++)
						{
								i = m % NX;
								k = m / NX;
								j = k % NY;
								k = k / NY;
								id = j * NX * NZ + k * NX + i;

								y0_r[id] = x1_r[m];
								y0_i[id] = x1_i[m];
						}

						//cfftz (is, logNY, NY, y0_r, y0_i, y1_r, y1_i, NX, NZ);
						/*--------------------------------------------------------------------
						  c   Perform one variant of the Stockham FFT.
						  c-------------------------------------------------------------------*/
						for (l = 1; l <= logNY; l += 2)
						{
								//fftz2 (is, l, logNY, NY, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NX, NZ);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NY; idx++)
								{

										n1 = NY / 2;
										if (l - 1 == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l - 1) - 1);
										}

										if (logNY - l == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNY - l) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y0_r[i11 + p];
														x11imag = y0_i[i11 + p];
														x21real = y0_r[i12 + p];
														x21imag = y0_i[i12 + p];
														y1_r[i21 + p] = x11real + x21real;
														y1_i[i21 + p] = x11imag + x21imag;
														y1_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y1_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
								if (l == logNY)
										break;
								//fftz2 (is, l + 1, logNY, NY, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NX, NZ);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NX * NZ;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NY; idx++)
								{

										n1 = NY / 2;
										if (l == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l) - 1);
										}

										if (logNY - (l+1) == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNY - (l+1)) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y1_r[i11 + p];
														x11imag = y1_i[i11 + p];
														x21real = y1_r[i12 + p];
														x21imag = y1_i[i12 + p];
														y0_r[i21 + p] = x11real + x21real;
														y0_i[i21 + p] = x11imag + x21imag;
														y0_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y0_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
						}

						if (logNY % 2 == 1)
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = j * NX * NZ + k * NX + i;

										x1_r[m] = y1_r[id];
										x1_i[m] = y1_i[id];
								}
						}
						else
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = j * NX * NZ + k * NX + i;

										x1_r[m] = y0_r[id];
										x1_i[m] = y0_i[id];
								}
						}

						//cffts1 (-1, logNX, x1_r, x1_i, x2_r, x2_i, y0_r, y0_i, y1_r, y1_i);	/* x1 -> x2 */
						is = -1;

#pragma acc kernels loop gang worker independent
						for (m = 0; m < NTOTAL; m++)
						{
								i = m % NX;
								k = m / NX;
								j = k % NY;
								k = k / NY;
								id = i * NZ * NY + j * NZ + k;

								y0_r[id] = x1_r[m];
								y0_i[id] = x1_i[m];
						}

						//cfftz (is, logNX, NX, y0_r, y0_i, y1_r, y1_i, NZ, NY);
						/*--------------------------------------------------------------------
						  c   Perform one variant of the Stockham FFT.
						  c-------------------------------------------------------------------*/
						for (l = 1; l <= logNX; l += 2)
						{
								//fftz2 (is, l, logNX, NX, u_r, u_i, y0_r, y0_i, y1_r, y1_i, NZ, NY);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NX; idx++)
								{

										n1 = NX / 2;
										if (l - 1 == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l - 1) - 1);
										}

										if (logNX - l == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNX - l) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y0_r[i11 + p];
														x11imag = y0_i[i11 + p];
														x21real = y0_r[i12 + p];
														x21imag = y0_i[i12 + p];
														y1_r[i21 + p] = x11real + x21real;
														y1_i[i21 + p] = x11imag + x21imag;
														y1_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y1_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
								if (l == logNX)
										break;
								//fftz2 (is, l + 1, logNX, NX, u_r, u_i, y1_r, y1_i, y0_r, y0_i, NZ, NY);
								/*--------------------------------------------------------------------
								  c   Set initial parameters.
								  c-------------------------------------------------------------------*/
								nn = NZ * NY;		//number of threads

#pragma acc kernels loop gang worker independent
								for (idx = 0; idx < NTOTAL / NX; idx++)
								{

										n1 = NX / 2;
										if (l == 0)
										{
												lk = 1;
										}
										else
										{
												lk = 2 << ((l) - 1);
										}

										if (logNX - (l+1) == 0)
										{
												li = 1;
										}
										else
										{
												li = 2 << ((logNX - (l+1)) - 1);
										}

										lj = 2 * lk;
										ku = li;

										for (i_fftz2 = 0; i_fftz2 < li; i_fftz2++)
										{
												i11 = idx + i_fftz2 * lk * nn;
												i12 = i11 + n1 * nn;
												i21 = idx + i_fftz2 * lj * nn;
												i22 = i21 + lk * nn;

												if (is >= 1)
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = u_i[ku + i_fftz2];
												}
												else
												{
														u1_rf = u_r[ku + i_fftz2];
														u1_if = -u_i[ku + i_fftz2];
												}

												/*--------------------------------------------------------------------
												  c   This loop is vectorizable.
												  c-------------------------------------------------------------------*/
												for (k_fftz2 = 0; k_fftz2 < lk; k_fftz2++)
												{
														p = k_fftz2 * nn;
														x11real = y1_r[i11 + p];
														x11imag = y1_i[i11 + p];
														x21real = y1_r[i12 + p];
														x21imag = y1_i[i12 + p];
														y0_r[i21 + p] = x11real + x21real;
														y0_i[i21 + p] = x11imag + x21imag;
														y0_r[i22 + p] =
																u1_rf * (x11real - x21real) - u1_if * (x11imag - x21imag);
														y0_i[i22 + p] =
																u1_rf * (x11imag - x21imag) + u1_if * (x11real - x21real);
												}
										}
								}
						}

						if (logNX % 2 == 1)
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = i * NZ * NY + j * NZ + k;

										x2_r[m] = y1_r[id];
										x2_i[m] = y1_i[id];
								}
						}
						else
						{
#pragma acc kernels loop gang worker independent
								for (m = 0; m < NTOTAL; m++)
								{
										i = m % NX;
										k = m / NX;
										j = k % NY;
										k = k / NY;
										id = i * NZ * NY + j * NZ + k;

										x2_r[m] = y0_r[id];
										x2_i[m] = y0_i[id];
								}
						}
				}
		}
		//END_FFT//
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

		static void
fft_init (int n)
{

		/*--------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		/*--------------------------------------------------------------------
		  c compute the roots-of-unity array that will be used for subsequent FFTs. 
		  c-------------------------------------------------------------------*/

		int m, nu, ku, i, j, ln;
		float t, ti;


		/*--------------------------------------------------------------------
		  c   Initialize the U array with sines and cosines in a manner that permits
		  c   stride one access at each FFT iteration.
		  c-------------------------------------------------------------------*/
		nu = n;
		m = ilog2 (n);
		u_r[0] = (float) m;
		u_i[0] = 0.0;
		ku = 1;
		ln = 1;

		for (j = 1; j <= m; j++)
		{
				t = PI / ln;

				for (i = 0; i <= ln - 1; i++)
				{
						ti = i * t;
						u_r[i + ku] = cos (ti);
						u_i[i + ku] = sin (ti);
				}

				ku = ku + ln;
				ln = 2 * ln;
		}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

		static int
ilog2 (int n)
{

		/*--------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		int nn, lg;

		if (n == 1)
		{
				return 0;
		}
		lg = 1;
		nn = 2;
		while (nn < n)
		{
				nn = nn << 1;
				lg++;
		}

		return lg;
}



/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

		static void
verify (int d1, int d2, int d3, int nt, boolean * verified, char *classT)
{

		/*--------------------------------------------------------------------
		  c-------------------------------------------------------------------*/

		int ierr, size, i;
		double err, epsilon;

		/*--------------------------------------------------------------------
		  c   Sample size reference checksums
		  c-------------------------------------------------------------------*/

		/*--------------------------------------------------------------------
		  c   Class S size reference checksums
		  c-------------------------------------------------------------------*/
		double vdata_real_s[6 + 1] = { 0.0,
				5.546087004964e+02,
				5.546385409189e+02,
				5.546148406171e+02,
				5.545423607415e+02,
				5.544255039624e+02,
				5.542683411902e+02
		};
		double vdata_imag_s[6 + 1] = { 0.0,
				4.845363331978e+02,
				4.865304269511e+02,
				4.883910722336e+02,
				4.901273169046e+02,
				4.917475857993e+02,
				4.932597244941e+02
		};
		/*--------------------------------------------------------------------
		  c   Class W size reference checksums
		  c-------------------------------------------------------------------*/
		double vdata_real_w[6 + 1] = { 0.0,
				5.673612178944e+02,
				5.631436885271e+02,
				5.594024089970e+02,
				5.560698047020e+02,
				5.530898991250e+02,
				5.504159734538e+02
		};
		double vdata_imag_w[6 + 1] = { 0.0,
				5.293246849175e+02,
				5.282149986629e+02,
				5.270996558037e+02,
				5.260027904925e+02,
				5.249400845633e+02,
				5.239212247086e+02
		};
		/*--------------------------------------------------------------------
		  c   Class A size reference checksums
		  c-------------------------------------------------------------------*/
		double vdata_real_a[6 + 1] = { 0.0,
				5.046735008193e+02,
				5.059412319734e+02,
				5.069376896287e+02,
				5.077892868474e+02,
				5.085233095391e+02,
				5.091487099959e+02
		};
		double vdata_imag_a[6 + 1] = { 0.0,
				5.114047905510e+02,
				5.098809666433e+02,
				5.098144042213e+02,
				5.101336130759e+02,
				5.104914655194e+02,
				5.107917842803e+02
		};
		/*--------------------------------------------------------------------
		  c   Class B size reference checksums
		  c-------------------------------------------------------------------*/
		double vdata_real_b[20 + 1] = { 0.0,
				5.177643571579e+02,
				5.154521291263e+02,
				5.146409228649e+02,
				5.142378756213e+02,
				5.139626667737e+02,
				5.137423460082e+02,
				5.135547056878e+02,
				5.133910925466e+02,
				5.132470705390e+02,
				5.131197729984e+02,
				5.130070319283e+02,
				5.129070537032e+02,
				5.128182883502e+02,
				5.127393733383e+02,
				5.126691062020e+02,
				5.126064276004e+02,
				5.125504076570e+02,
				5.125002331720e+02,
				5.124551951846e+02,
				5.124146770029e+02
		};
		double vdata_imag_b[20 + 1] = { 0.0,
				5.077803458597e+02,
				5.088249431599e+02,
				5.096208912659e+02,
				5.101023387619e+02,
				5.103976610617e+02,
				5.105948019802e+02,
				5.107404165783e+02,
				5.108576573661e+02,
				5.109577278523e+02,
				5.110460304483e+02,
				5.111252433800e+02,
				5.111968077718e+02,
				5.112616233064e+02,
				5.113203605551e+02,
				5.113735928093e+02,
				5.114218460548e+02,
				5.114656139760e+02,
				5.115053595966e+02,
				5.115415130407e+02,
				5.115744692211e+02
		};
		/*--------------------------------------------------------------------
		  c   Class C size reference checksums
		  c-------------------------------------------------------------------*/
		double vdata_real_c[20 + 1] = { 0.0,
				5.195078707457e+02,
				5.155422171134e+02,
				5.144678022222e+02,
				5.140150594328e+02,
				5.137550426810e+02,
				5.135811056728e+02,
				5.134569343165e+02,
				5.133651975661e+02,
				5.132955192805e+02,
				5.132410471738e+02,
				5.131971141679e+02,
				5.131605205716e+02,
				5.131290734194e+02,
				5.131012720314e+02,
				5.130760908195e+02,
				5.130528295923e+02,
				5.130310107773e+02,
				5.130103090133e+02,
				5.129905029333e+02,
				5.129714421109e+02
		};
		double vdata_imag_c[20 + 1] = { 0.0,
				5.149019699238e+02,
				5.127578201997e+02,
				5.122251847514e+02,
				5.121090289018e+02,
				5.121143685824e+02,
				5.121496764568e+02,
				5.121870921893e+02,
				5.122193250322e+02,
				5.122454735794e+02,
				5.122663649603e+02,
				5.122830879827e+02,
				5.122965869718e+02,
				5.123075927445e+02,
				5.123166486553e+02,
				5.123241541685e+02,
				5.123304037599e+02,
				5.123356167976e+02,
				5.123399592211e+02,
				5.123435588985e+02,
				5.123465164008e+02
		};

		epsilon = 1.0e-12;
		*verified = TRUE;
		*classT = 'U';

		if (d1 == 64 && d2 == 64 && d3 == 64 && nt == 6)
		{
				*classT = 'S';
				for (i = 1; i <= nt; i++)
				{
						err = (sums_r[i] - vdata_real_s[i]) / vdata_real_s[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
						err = (sums_i[i] - vdata_imag_s[i]) / vdata_imag_s[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
				}
		}
		else if (d1 == 128 && d2 == 128 && d3 == 32 && nt == 6)
		{
				*classT = 'W';
				for (i = 1; i <= nt; i++)
				{
						err = (sums_r[i] - vdata_real_w[i]) / vdata_real_w[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
						err = (sums_i[i] - vdata_imag_w[i]) / vdata_imag_w[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
				}
		}
		else if (d1 == 256 && d2 == 256 && d3 == 128 && nt == 6)
		{
				*classT = 'A';
				for (i = 1; i <= nt; i++)
				{
						err = (sums_r[i] - vdata_real_a[i]) / vdata_real_a[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
						err = (sums_i[i] - vdata_imag_a[i]) / vdata_imag_a[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
				}
		}
		else if (d1 == 512 && d2 == 256 && d3 == 256 && nt == 20)
		{
				*classT = 'B';
				for (i = 1; i <= nt; i++)
				{
						err = (sums_r[i] - vdata_real_b[i]) / vdata_real_b[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
						err = (sums_i[i] - vdata_imag_b[i]) / vdata_imag_b[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
				}
		}
		else if (d1 == 512 && d2 == 512 && d3 == 512 && nt == 20)
		{
				*classT = 'C';
				for (i = 1; i <= nt; i++)
				{
						err = (sums_r[i] - vdata_real_c[i]) / vdata_real_c[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
						err = (sums_i[i] - vdata_imag_c[i]) / vdata_imag_c[i];
						if (fabs (err) > epsilon)
						{
								*verified = FALSE;
								break;
						}
				}
		}

		if (*classT != 'U')
		{
				printf ("Result verification successful\n");
		}
		else
		{
				printf ("Result verification failed\n");
		}
		printf ("class = %1c\n", *classT);
}
