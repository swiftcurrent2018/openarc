/**********************************************************************************************/
/* Copyright (c) 2015 Amanda Randles, John Gounley                                            */
/* All rights reserved.                                                                       */
/*                                                                                            */
/* Redistribution and use in source and binary forms, with or without modification, are       */
/*  permitted provided that the following conditions are met:                                 */
/*                                                                                            */
/* 1. Redistributions of source code must retain the above copyright notice, this list of     */
/* conditions and the following disclaimer.                                                   */
/*                                                                                            */
/* 2. Redistributions in binary form must reproduce the above copyright notice, this list of  */
/* conditions and the following disclaimer in the documentation and/or other materials        */
/* provided with the distribution.                                                            */
/*                                                                                            */
/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS*/
/* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF            */
/* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  */
/* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         */
/* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   */
/* TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         */
/* SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                              */
/**********************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>    
#include <string.h>
#include <sys/time.h>

// parameters
#define MC	19			// number of discrete velocities
#define LX_MAX	81		// number of lattice points, x-direction
#define LY_MAX	81		// number of lattice points, y-direction
#define LZ_MAX	61		// number of lattice points, z-direction
#define T (1.0 / 3.0)	// speed of sound squared in LB units
#define omega 1.0		// relaxation time for LBM

#ifdef _OPENARC_
#pragma openarc #define MC	19			
#pragma openarc #define LX_MAX	81		
#pragma openarc #define LY_MAX	81		
#pragma openarc #define LZ_MAX	61		
#pragma openarc #define T (1.0 / 3.0)	
#pragma openarc #define omega 1.0		
#endif

#ifndef NSTEPS
#define NSTEPS 100
#endif

#ifndef REORDER
#define REORDER 0
#endif

#if REORDER == 1
double	distr[LX_MAX][LY_MAX][LZ_MAX][MC];		// distribution function
double	distr_adv[LX_MAX][LY_MAX][LZ_MAX][MC];	// distribution function, to advance
#else
double	distr[MC][LX_MAX][LY_MAX][LZ_MAX];		// distribution function
double	distr_adv[MC][LX_MAX][LY_MAX][LZ_MAX];	// distribution function, to advance
#endif

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}

// main program
int main(int argc, char **argv) {

	printf("starting setup...\n");
	/* SETUP SECTION */	
	
	// variable definitions	*/
 	int		ic[MC][3], idg[MC];
	double   distr_eq[MC];

	int 	lx, ly, lz;
	double   rho, ux, uy, uz, distr0, distr1, uke;
	double	px, py, pz, delNi;
	double 	u0 = 1e-4;
	double 	cdotuM;
	
	FILE    *vtk_fp, *fp, *g_fp;	
	int 	i, ix, iy, iz, is, ik, istate, j, i1, js;

	double strt_time, end_time;
	double strt_time2, end_time2;

	// read state vectors */
	fp = fopen("states19.i", "r");
	if (fp == NULL) {
		fprintf(stderr, "Error: cannot open file states19.i.\n");
		return 1;
	}
	for(i = 0; i < MC; i++) {
		fscanf(fp, "%d %d %d %d %d", &istate, &ic[i][0], &ic[i][1], &ic[i][2], &idg[i]);
	}
	fclose(fp);
	
	// setup time
	int n_step;
	int n_step_max = NSTEPS;
	int n_step_out = NSTEPS;
	
	// setup grid dimensions
	lx = LX_MAX;
	ly = LY_MAX;	
	lz = LZ_MAX;
	double dx = 10.0/( (double) (lx-1));
	double dt = dx;		// fix
	
	// setup initial flow (shear flow)
	distr1 = 1.0 / 36.0;
	distr0 = 1.0 / 3.0;
	for(ix = 0; ix < lx; ix++) {
		for(iy = 0; iy < ly; iy++) {
			for(iz = 0; iz < lz; iz++) {
			
				rho = 1.0;
				ux = u0*dx*( iy - (ly-1.0)/2.0 );
				uy = 0.0;
				uz = 0.0;			
				uke = 0.5*(ux*ux + uy*uy + uz*uz);
				
				for(is = 0; is < MC-1; is++) {
					cdotuM = ic[is][0]*ux + ic[is][1]*uy + ic[is][2]*uz;
#if REORDER == 1
					distr[ix][iy][iz][is] = rho*distr1*idg[is]*(1.0 + 
						cdotuM/T + 0.5*(cdotuM/T)*(cdotuM/T) - uke/T);
#else
					distr[is][ix][iy][iz] = rho*distr1*idg[is]*(1.0 + 
						cdotuM/T + 0.5*(cdotuM/T)*(cdotuM/T) - uke/T);
#endif
				}
#if REORDER == 1
				distr[ix][iy][iz][MC-1] = distr0*(1.0 - uke/T);							
#else
				distr[MC-1][ix][iy][iz] = distr0*(1.0 - uke/T);							
#endif
			}
		}
	}	
	
	printf("starting loop...\n");
	strt_time = my_timer();
	//hoisted from the loop body.
	double n[3];
	double umax[3];
	double cdotn[19];
	double cdotu[19];
	double tdotu[19];
	double t[19][3];
	int exitVel[5];
	int tangVel[9];
	/* THE TIMESTEPPING LOOP */	
#if REORDER == 1
	#pragma acc data copyin(distr[0:LX_MAX][0:LY_MAX][0:LZ_MAX][0:MC], ic[0:MC][0:3], idg[0:MC]) create(distr_adv[0:LX_MAX][0:LY_MAX][0:LZ_MAX][0:MC], n, umax, cdotn, cdotu, tdotu, t, exitVel, tangVel)
#else
	#pragma acc data copyin(distr[0:MC][0:LX_MAX][0:LY_MAX][0:LZ_MAX], ic[0:MC][0:3], idg[0:MC]) create(distr_adv[0:MC][0:LX_MAX][0:LY_MAX][0:LZ_MAX], n, umax, cdotn, cdotu, tdotu, t, exitVel, tangVel)
#endif
	for(n_step = 1; n_step <= n_step_max; n_step++) {
	
		/* collision - vanilla version */
		#pragma acc kernels loop gang worker collapse(3) private(distr_eq[0:MC])
		for(ix = 0; ix < lx; ix++) {
		    for(iy = 0; iy < ly; iy++) {
		        for(iz = 0; iz < lz; iz++) {

					rho = px = py = pz = 0.0;
					for(is = 0; is < MC; is++) {
#if REORDER == 1
						rho += distr[ix][iy][iz][is];
						px += distr[ix][iy][iz][is]*ic[is][0];
						py += distr[ix][iy][iz][is]*ic[is][1];
						pz += distr[ix][iy][iz][is]*ic[is][2];
#else
						rho += distr[is][ix][iy][iz];
						px += distr[is][ix][iy][iz]*ic[is][0];
						py += distr[is][ix][iy][iz]*ic[is][1];
						pz += distr[is][ix][iy][iz]*ic[is][2];
#endif
					}
					ux = px / rho;
					uy = py / rho;
					uz = pz / rho;

					uke = 0.5*(ux*ux + uy*uy + uz*uz);

					for(is = 0; is < MC-1; is++) {
						cdotuM = ic[is][0]*ux + ic[is][1]*uy + ic[is][2]*uz;
						distr_eq[is] = rho*distr1*idg[is]*(1.0 + 
							cdotuM/T + 0.5*(cdotuM/T)*(cdotuM/T) - uke/T);
					}
					distr_eq[MC-1] = distr0*(1.0 - uke/T);

					for(is = 0; is < MC; is++) {
#if REORDER == 1
						delNi = -omega * (distr[ix][iy][iz][is] - distr_eq[is]);
						distr_adv[ix][iy][iz][is] = distr[ix][iy][iz][is] + delNi;
#else
						delNi = -omega * (distr[is][ix][iy][iz] - distr_eq[is]);
						distr_adv[is][ix][iy][iz] = distr[is][ix][iy][iz] + delNi;
#endif
					}
				}
			}
		}
		

		/* advection */
		#pragma acc kernels loop gang worker collapse(3)
		for(ix = 0; ix < lx; ix++) {
		    for(iy = 0; iy < ly; iy++) {
		        for(iz = 0; iz < lz; iz++) {
						
					int ixa, iya, iza;

					for(is = 0; is < MC; is++) {
							
						ixa = ix + ic[is][0];
						iya = iy + ic[is][1];
						iza = iz + ic[is][2];

						ixa = (ixa < 0) ? lx-1 : ixa % lx;
						iza = (iza < 0) ? lz-1 : iza % lz;
						
						if (iya >= 0 && iya < ly){
#if REORDER == 1
							distr[ixa][iya][iza][is] = distr_adv[ix][iy][iz][is];
#else
							distr[is][ixa][iya][iza] = distr_adv[is][ix][iy][iz];
#endif
						}
					}
				}
			}
		}
			
		// todo : real bcs	
		
		// bottom
		iy = 0;
		n[0] = 0.0; n[1] = -1.0; n[2] = 0.0;
		umax[0] = u0*dx*(iy - (ly-1.0)/2.0);
		umax[1] = 0.0;
		umax[2] = 0.0;
		int exitCount = 0;
		int tangCount = 0;
		
		for(is = 0; is < MC; is++) {
			cdotn[is] = ic[is][0]*n[0] + ic[is][1]*n[1] + ic[is][2]*n[2];
			cdotu[is] = ic[is][0]*umax[0] + ic[is][1]*umax[1] + ic[is][2]*umax[2];
			t[is][0] = ic[is][0] - cdotn[is]*n[0];
			t[is][1] = ic[is][1] - cdotn[is]*n[1];
			t[is][2] = ic[is][2] - cdotn[is]*n[2];
			tdotu[is] = t[is][0]*umax[0] + t[is][1]*umax[1] + t[is][2]*umax[2];
			if (cdotn[is] > 0.5)
			{
				exitVel[exitCount] = is;
				exitCount++;
			}
			else if (cdotn[is] > -0.5)
			{
				tangVel[tangCount] = is;
				tangCount++;
			}
		}

		#pragma acc update device(tangVel, cdotn, cdotu, tdotu, t, exitVel, umax, n)
		
		#pragma acc kernels loop gang 
		for(ix = 0; ix < lx; ix++) {
			#pragma acc loop worker
		    for(iz = 0; iz < lz; iz++) {
		    
		    	double tempRho = 0.0;
				unsigned in;
				for (in=0; in<9; ++in)
				{	
					is = tangVel[in];
#if REORDER == 1
					tempRho += distr[ix][iy][iz][is];
#else
					tempRho += distr[is][ix][iy][iz];
#endif
				}
					
				for (in=0; in<5; ++in)
				{
					is = exitVel[in];
#if REORDER == 1
					tempRho += 2.0*distr[ix][iy][iz][is];
#else
					tempRho += 2.0*distr[is][ix][iy][iz];
#endif
				}
				
				double rho = tempRho/(1+n[0]*umax[0]+n[1]*umax[1]+n[2]*umax[2]);
		
				// compute sum in equation 27 of Hecht, Harting (JSM, 2010)
				double temp_sum[MC];
				for(is = 0; is < MC; is++)
				{
					temp_sum[is] = 0.0;
					for(js = 0; js < MC; js++)
					{
						double cdottij = (ic[js][0]*t[is][0] + ic[js][1]*t[is][1] + ic[js][2]*t[is][2]);
#if REORDER == 1
						temp_sum[is] += distr[ix][iy][iz][js]*cdottij*(1.-fabs(cdotn[js]));
#else
						temp_sum[is] += distr[js][ix][iy][iz]*cdottij*(1.-fabs(cdotn[js]));
#endif
					}
				}
		
				// equation 27 of Hecht, Harting (JSM, 2010), with fix based on stencil weight idg
				for (in=0; in<5; ++in)
				{
				int is = exitVel[in];
				int opp = is-1;
#if REORDER == 1
				distr[ix][iy][iz][opp]  = distr[ix][iy][iz][is] 
					- cdotu[is]*rho*(double)idg[is]/6. - tdotu[is]*rho/3. + 0.5*temp_sum[is];
#else
				distr[opp][ix][iy][iz]  = distr[is][ix][iy][iz] 
					- cdotu[is]*rho*(double)idg[is]/6. - tdotu[is]*rho/3. + 0.5*temp_sum[is];
#endif
				}
		 	}
		}   
		
		// top
		iy = ly-1;
		n[0] = 0.0;	n[1] = 1.0; n[2] = 0.0;
		umax[0] = u0*dx*(iy - (ly-1.0)/2.0); umax[1] = 0.0; umax[2] = 0.0;
		exitCount = 0;
		tangCount = 0;
		
		for(is = 0; is < MC; is++) {
			cdotn[is] = ic[is][0]*n[0] + ic[is][1]*n[1] + ic[is][2]*n[2];
			cdotu[is] = ic[is][0]*umax[0] + ic[is][1]*umax[1] + ic[is][2]*umax[2];
			t[is][0] = ic[is][0] - cdotn[is]*n[0];
			t[is][1] = ic[is][1] - cdotn[is]*n[1];
			t[is][2] = ic[is][2] - cdotn[is]*n[2];
			tdotu[is] = t[is][0]*umax[0] + t[is][1]*umax[1] + t[is][2]*umax[2];
			if (cdotn[is] > 0.5)
			{
				exitVel[exitCount] = is;
				exitCount++;
			}
			else if (cdotn[is] > -0.5)
			{
				tangVel[tangCount] = is;
				tangCount++;
			}
		}

		#pragma acc update device(tangVel, cdotn, cdotu, tdotu, t, exitVel, umax, n)

		#pragma acc kernels loop gang 
		for(ix = 0; ix < lx; ix++) {
			#pragma acc loop worker
		    for(iz = 0; iz < lz; iz++) {
		    
		    	double tempRho = 0.0;
				unsigned in;
				for (in=0; in<9; ++in)
				{	
					is = tangVel[in];
#if REORDER == 1
					tempRho += distr[ix][iy][iz][is];
#else
					tempRho += distr[is][ix][iy][iz];
#endif
				}
					
				for (in=0; in<5; ++in)
				{
					is = exitVel[in];
#if REORDER == 1
					tempRho += 2.0*distr[ix][iy][iz][is];
#else
					tempRho += 2.0*distr[is][ix][iy][iz];
#endif
				}
				
				double rho = tempRho/(1+n[0]*umax[0]+n[1]*umax[1]+n[2]*umax[2]);
		
				// compute sum in equation 27 of Hecht, Harting (JSM, 2010)
				double temp_sum[MC];
				for(is = 0; is < MC; is++)
				{
					temp_sum[is] = 0.0;
					for(js = 0; js < MC; js++)
					{
						double cdottij = (ic[js][0]*t[is][0] + ic[js][1]*t[is][1] + ic[js][2]*t[is][2]);
#if REORDER == 1
						temp_sum[is] += distr[ix][iy][iz][js]*cdottij*(1.-fabs(cdotn[js]));
#else
						temp_sum[is] += distr[js][ix][iy][iz]*cdottij*(1.-fabs(cdotn[js]));
#endif
					}
				}
		
				// equation 27 of Hecht, Harting (JSM, 2010), with fix based on stencil weight idg
				for (in=0; in<5; ++in)
				{
				int is = exitVel[in];
				int opp = is+1;
#if REORDER == 1
				distr[ix][iy][iz][opp]  = distr[ix][iy][iz][is] 
					- cdotu[is]*rho*(double)idg[is]/6. - tdotu[is]*rho/3. + 0.5*temp_sum[is];
#else
				distr[opp][ix][iy][iz]  = distr[is][ix][iy][iz] 
					- cdotu[is]*rho*(double)idg[is]/6. - tdotu[is]*rho/3. + 0.5*temp_sum[is];
#endif
				}
		 	}
		}
				
		/* output */
		if (n_step % n_step_out == 0)
		{
#if REORDER == 1
			#pragma acc update host(distr[0:LX_MAX][0:LY_MAX][0:LZ_MAX][0:MC])
#else
			#pragma acc update host(distr[0:MC][0:LX_MAX][0:LY_MAX][0:LZ_MAX])
#endif
			strt_time2 = my_timer();
			printf("%d\n", n_step);
			
			char fname[15];
        	sprintf(fname,"lbe_%d.vtk",n_step);
	        vtk_fp = fopen(fname, "w");
        	if (vtk_fp == NULL) {
                	fprintf(stderr, "Error: cannot open file lbe.vtk. \n");
                	return 1;
        	}
        	fprintf(vtk_fp, "# vtk DataFile Version 3.0");
        	fprintf(vtk_fp, "\nvtk global output\n");
        	fprintf(vtk_fp, "ASCII\n");
        	fprintf(vtk_fp, "DATASET STRUCTURED_POINTS\n");
			fprintf(vtk_fp, "DIMENSIONS %i %i %i\n", lx, ly, lz);
			fprintf(vtk_fp, "ORIGIN 0. 0. 0.\n");
			fprintf(vtk_fp, "SPACING 1 1 1\n");
			fprintf(vtk_fp, "\nPOINT_DATA %d\n",lx*ly*lz);
	        fprintf(vtk_fp, "SCALARS rho double \n");
	        fprintf(vtk_fp, "LOOKUP_TABLE default\n");
	        
	        for(iz = 0; iz < lz; iz++) {
                for(iy = 0; iy < ly; iy++) {
                    for(ix = 0; ix < lx; ix++) {
                        rho = px = py = pz = 0.0;

                        for(is = 0; is < MC; is++) {
#if REORDER == 1
                        	rho += distr[ix][iy][iz][is];
#else
                        	rho += distr[is][ix][iy][iz];
#endif
                        }
                        
                    	fprintf(vtk_fp, "%f\n",rho);
                	 }
        	     }
	        }
	        
	        fprintf(vtk_fp, "VECTORS velocity double \n");
        	for(iz = 0; iz < lz; iz++) {
	        	for(iy = 0; iy < ly; iy++) {
                	for(ix = 0; ix < lx; ix++) {
                        rho = px = py = pz = 0.0;

                        for(is = 0; is < MC; is++) {
#if REORDER == 1
                            rho += distr[ix][iy][iz][is];
                            px += distr[ix][iy][iz][is]*ic[is][0];
                            py += distr[ix][iy][iz][is]*ic[is][1];
                            pz += distr[ix][iy][iz][is]*ic[is][2];
#else
                            rho += distr[is][ix][iy][iz];
                            px += distr[is][ix][iy][iz]*ic[is][0];
                            py += distr[is][ix][iy][iz]*ic[is][1];
                            pz += distr[is][ix][iy][iz]*ic[is][2];
#endif
                    	}

                	    ux = px / rho;
        	            uy = py / rho;
	                    uz = pz / rho;

                    	fprintf(vtk_fp, "%f %f %f\n",ux,uy,uz);
                    }
                }
        	}
        	
        	fprintf(vtk_fp, "VECTORS coordinates double \n");
        	for(iz = 0; iz < lz; iz++) {
	        	for(iy = 0; iy < ly; iy++) {
                	for(ix = 0; ix < lx; ix++) {

                    	fprintf(vtk_fp, "%f %f %f\n",ix*dx,iy*dx,iz*dx);
                    }
                }
        	}

	        fclose(vtk_fp);  
			end_time2 = my_timer();
		}
		
	
	}
	end_time = my_timer();
	printf("Main Computation Time = %lf sec\n", end_time - strt_time);
	printf("Output Print Time = %lf sec\n", end_time2 - strt_time2);
	printf("Net Computation Time = %lf sec\n", (end_time - strt_time)-(end_time2 - strt_time2));

	return 0;
}

