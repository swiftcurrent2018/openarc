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

// parameters
#define MC	19			// number of discrete velocities
#define LX_MAX	81		// number of lattice points, x-direction
#define LY_MAX	81		// number of lattice points, y-direction
#define LZ_MAX	61		// number of lattice points, z-direction
#define T (1.0 / 3.0)	// speed of sound squared in LB units
#define omega 1.0		// relaxation time for LBM

double	distr[MC][LX_MAX][LY_MAX][LZ_MAX];		// distribution function
double	distr_adv[MC][LX_MAX][LY_MAX][LZ_MAX];	// distribution function, to advance

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
	double 	cdotu;
	
	FILE    *vtk_fp, *fp, *g_fp;	
	int 	i, ix, iy, iz, is, ik, istate, j, i1, js;

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
	int n_step_max = 1000;
	int n_step_out = 1000;
	
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
					cdotu = ic[is][0]*ux + ic[is][1]*uy + ic[is][2]*uz;
					distr[is][ix][iy][iz] = rho*distr1*idg[is]*(1.0 + 
						cdotu/T + 0.5*(cdotu/T)*(cdotu/T) - uke/T);
				}
				distr[MC-1][ix][iy][iz] = distr0*(1.0 - uke/T);							
			}
		}
	}	
	
	printf("starting loop...\n");
	/* THE TIMESTEPPING LOOP */	
	for(n_step = 1; n_step <= n_step_max; n_step++) {
	
		/* collision - vanilla version */
		for(ix = 0; ix < lx; ix++) {
		    for(iy = 0; iy < ly; iy++) {
		        for(iz = 0; iz < lz; iz++) {

					rho = px = py = pz = 0.0;
					for(is = 0; is < MC; is++) {
						rho += distr[is][ix][iy][iz];
						px += distr[is][ix][iy][iz]*ic[is][0];
						py += distr[is][ix][iy][iz]*ic[is][1];
						pz += distr[is][ix][iy][iz]*ic[is][2];
					}
					ux = px / rho;
					uy = py / rho;
					uz = pz / rho;

					uke = 0.5*(ux*ux + uy*uy + uz*uz);

					for(is = 0; is < MC-1; is++) {
						cdotu = ic[is][0]*ux + ic[is][1]*uy + ic[is][2]*uz;
						distr_eq[is] = rho*distr1*idg[is]*(1.0 + 
							cdotu/T + 0.5*(cdotu/T)*(cdotu/T) - uke/T);
					}
					distr_eq[MC-1] = distr0*(1.0 - uke/T);

					for(is = 0; is < MC; is++) {
						delNi = -omega * (distr[is][ix][iy][iz] - distr_eq[is]);
						distr_adv[is][ix][iy][iz] = distr[is][ix][iy][iz] + delNi;
					}
				}
			}
		}
		

		/* advection */
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
							distr[is][ixa][iya][iza] = distr_adv[is][ix][iy][iz];
						}
					}
				}
			}
		}
			
		// todo : real bcs	
		
		// bottom
		iy = 0;
		double n[3] = {0.0, -1.0, 0.0};
		double umax[3] = {u0*dx*(iy - (ly-1.0)/2.0), 0.0, 0.0};
		double cdotn[19];
		double cdotu[19];
		double tdotu[19];
		double t[19][3];
		int exitVel[5];
		int tangVel[9];
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
		
		for(ix = 0; ix < lx; ix++) {
		    for(iz = 0; iz < lz; iz++) {
		    
		    	double tempRho = 0.0;
				for (unsigned in=0; in<9; ++in)
				{	
					is = tangVel[in];
					tempRho += distr[is][ix][iy][iz];
				}
					
				for (unsigned in=0; in<5; ++in)
				{
					is = exitVel[in];
					tempRho += 2.0*distr[is][ix][iy][iz];
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
						temp_sum[is] += distr[js][ix][iy][iz]*cdottij*(1.-fabs(cdotn[js]));
					}
				}
		
				// equation 27 of Hecht, Harting (JSM, 2010), with fix based on stencil weight idg
				for (unsigned in=0; in<5; ++in)
				{
				int is = exitVel[in];
				int opp = is-1;
				distr[opp][ix][iy][iz]  = distr[is][ix][iy][iz] 
					- cdotu[is]*rho*(double)idg[is]/6. - tdotu[is]*rho/3. + 0.5*temp_sum[is];
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
		for(ix = 0; ix < lx; ix++) {
		    for(iz = 0; iz < lz; iz++) {
		    
		    	double tempRho = 0.0;
				for (unsigned in=0; in<9; ++in)
				{	
					is = tangVel[in];
					tempRho += distr[is][ix][iy][iz];
				}
					
				for (unsigned in=0; in<5; ++in)
				{
					is = exitVel[in];
					tempRho += 2.0*distr[is][ix][iy][iz];
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
						temp_sum[is] += distr[js][ix][iy][iz]*cdottij*(1.-fabs(cdotn[js]));
					}
				}
		
				// equation 27 of Hecht, Harting (JSM, 2010), with fix based on stencil weight idg
				for (unsigned in=0; in<5; ++in)
				{
				int is = exitVel[in];
				int opp = is+1;
				distr[opp][ix][iy][iz]  = distr[is][ix][iy][iz] 
					- cdotu[is]*rho*(double)idg[is]/6. - tdotu[is]*rho/3. + 0.5*temp_sum[is];
				}
		 	}
		}
				
		/* output */
		if (n_step % n_step_out == 0)
		{
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
                        	rho += distr[is][ix][iy][iz];
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
                            rho += distr[is][ix][iy][iz];
                            px += distr[is][ix][iy][iz]*ic[is][0];
                            py += distr[is][ix][iy][iz]*ic[is][1];
                            pz += distr[is][ix][iy][iz]*ic[is][2];
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
		}
		
	
	}

	return 0;
}

