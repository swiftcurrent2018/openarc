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

// flow parameters
#define MC	19			// number of discrete velocities
#define LX_MAX	101		// number of lattice points, x-direction
#define LY_MAX	101		// number of lattice points, y-direction
#define LZ_MAX	101		// number of lattice points, z-direction
#define T (1.0 / 3.0)	// speed of sound squared in LB units
//#define omega 1.0		// relaxation time for LBM

// cell parameters
#define refinements	3	// number of refinements of icosahedron
#define NV 642	//2+10*(int)pow(4,refinements)			// number of vertices
#define NT 1280 //20*(int)pow(4,refinements)			// number of elements
#define NC 17			// number of cells

double	distr[MC][LX_MAX][LY_MAX][LZ_MAX];		// distribution function
double	distr_adv[MC][LX_MAX][LY_MAX][LZ_MAX];	// distribution function, to advance
double	eulf[LX_MAX][LY_MAX][LZ_MAX][3];		// force, in physical dimensions, on fluid points

double 	verts[NC][NV][3];	// vertex locations for cell, vertex, and dimension
int 	triangles[NT][3];	// triangle connectivity of vertices (ordered for outward normal)

// delta function
double delta(double r)
{
	double w = 0.0;
	if (fabs(r) <= 2.0){
		w = 0.25*(1.0+cos((3.1415926/2.0)*r));
	}	
	return w;
}

// series of utility functions to simplify finite element code
double norm(double a[3])
{
	double b = sqrt( a[0]*a[0] + a[1]*a[1] + a[2]*a[2] );
	return b;
}

double dot3(double a[3], double b[3])
{
	double c = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
	return c;
}

void dif3(double a[3], double b[3], double c[3])
{
	for(int index=0; index<3; index++)
	{
		c[index] = a[index]-b[index];
	}
	return;
}

void sum3(double a[3], double b[3], double c[3])
{
	for(int index=0; index<3; index++)
	{
		c[index] = a[index]+b[index];
	}
	return;
}

void times3(double a[3], double b, double c[3])
{
	for(int index=0; index<3; index++)
	{
		c[index] = b*a[index];
	}
	return;
}

void cross3(double a[3], double b[3], double c[3])
{
	c[0] = a[1]*b[2]-a[2]*b[1];
	c[1] = a[2]*b[0]-a[0]*b[2];
	c[2] = a[0]*b[1]-a[1]*b[0];
	return;
}

void refineIcosahedron(double undeformed[NV][3], int triangles[NT][3])
{
	int tempTriangles[NT][3];
	int ik;

	int numTriCur = 20;		
	int numVertCur = 12;
	double tolerance = 1e-6;	
		
	if (refinements > 0)
	{
		for (int n=1; n <= refinements; n++)
		{
			for (int i=0; i < numTriCur; i++)
			{
				int triCounter = i*4;
				int triI = triangles[i][0];
				int triJ = triangles[i][1];
				int triK = triangles[i][2];
				
				double bisect12[3], bisect23[3], bisect13[3];
				sum3(undeformed[triI], undeformed[triJ], bisect12);
				times3(bisect12, 0.5, bisect12);
				sum3(undeformed[triJ], undeformed[triK], bisect23);
				times3(bisect23, 0.5, bisect23);
				sum3(undeformed[triI], undeformed[triK], bisect13);
				times3(bisect13, 0.5, bisect13);
			
				int b12 = -1;
				int b23 = -1;	
				int b13 = -1;
				
				for (int j=0; j < numVertCur; j++)
				{
					double difference12[3], difference23[3], difference13[3];
					dif3(undeformed[j], bisect12, difference12);
					if( norm(difference12) < tolerance ){
 						b12 = j;
 					}
 					dif3(undeformed[j], bisect23, difference23);
 					if( norm(difference23) < tolerance ){
 						b23 = j;
 					}
 					dif3(undeformed[j], bisect13, difference13);
 					if( norm(difference13) < tolerance ){
 						b13 = j;
 					}
				}
				
				if (b12 == -1)
				{
					b12 = numVertCur;
					undeformed[numVertCur][0] = bisect12[0];
					undeformed[numVertCur][1] = bisect12[1];
					undeformed[numVertCur][2] = bisect12[2];
					numVertCur++;
				}
				
				if (b23 == -1)
				{
					b23 = numVertCur;
					undeformed[numVertCur][0] = bisect23[0];
					undeformed[numVertCur][1] = bisect23[1];
					undeformed[numVertCur][2] = bisect23[2];
					numVertCur++;
				}
				
				if (b13 == -1)
				{
					b13 = numVertCur;
					undeformed[numVertCur][0] = bisect13[0];
					undeformed[numVertCur][1] = bisect13[1];
					undeformed[numVertCur][2] = bisect13[2];
					numVertCur++;
				} 
				
				int newTri0[3] = {triI, b12, b13};
				tempTriangles[triCounter][0] = newTri0[0];
				tempTriangles[triCounter][1] = newTri0[1];
				tempTriangles[triCounter][2] = newTri0[2];
				
				int newTri1[3] = {triJ, b23, b12};
				tempTriangles[triCounter+1][0] = newTri1[0];
				tempTriangles[triCounter+1][1] = newTri1[1];
				tempTriangles[triCounter+1][2] = newTri1[2];
				
				int newTri2[3] = {triK, b13, b23};
				tempTriangles[triCounter+2][0] = newTri2[0];
				tempTriangles[triCounter+2][1] = newTri2[1];
				tempTriangles[triCounter+2][2] = newTri2[2];
				
				int newTri3[3] = {b12, b23, b13};
				tempTriangles[triCounter+3][0] = newTri3[0];
				tempTriangles[triCounter+3][1] = newTri3[1];
				tempTriangles[triCounter+3][2] = newTri3[2];
				
			}
			numTriCur = numTriCur*4;
			for (ik = 0; ik<numTriCur; ik++)
			{	
				triangles[ik][0] = tempTriangles[ik][0];
				triangles[ik][1] = tempTriangles[ik][1];
				triangles[ik][2] = tempTriangles[ik][2];
			}	    		
		}
	}
	
	// project refined icosahedron onto sphere		
	double tol = 1e-6;
	double scale = 0.8;				
	for (ik = 0; ik<NV; ik++)
	{
		double chi = atan(sqrt(pow(undeformed[ik][0],2.0)+pow(undeformed[ik][2],2.0))/undeformed[ik][1]);
		double phi = atan(undeformed[ik][2]/undeformed[ik][0]);
		if ( undeformed[ik][1] < 0.0)
		{
			chi += M_PI;
		}
		else if ( fabs(undeformed[ik][1]) < tol)
		{
			chi = M_PI/2.0;
		}
		if ( undeformed[ik][0] < 0.0)
		{
			phi += M_PI;
		}
		else if ( fabs(undeformed[ik][0]) < tol)
		{
			if (undeformed[ik][2] < 0.0)
			{
				phi = -M_PI/2.0;
			}
			else
			{
				phi = M_PI/2.0;
			}
		}	
		
		// spherical cells
// 		undeformed[ik][0] = scale*sin(chi)*cos(phi);
// 		undeformed[ik][1] = scale*cos(chi);
// 		undeformed[ik][2] = scale*sin(chi)*sin(phi);

		// biconcave cells
		double alpha = 1.38581894;
		undeformed[ik][0] = scale*alpha*sin(chi)*cos(phi);
		undeformed[ik][1] = (scale*alpha/2.0)*(0.207+2.003*pow(sin(chi),2.0)-1.123*pow(sin(chi),4.0))*cos(chi);
		undeformed[ik][2] = scale*alpha*sin(chi)*sin(phi);
		
	}

	return;
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
	
	double 	fx, fy, fz, distr_f[MC];
	double 	cdotu, fdotu, fdotc; 
	
	FILE    *vtk_fp, *fp, *g_fp;	
	int 	i, ix, iy, iz, is, ik, istate, j, i1, js;
	
	double 	centers[NC][3];
	double 	undeformed[NV][3];
	double 	lagf[NC][NV][3];
	double	lagu[NC][NV][3];
	
	// setup time
	int n_step;
	int n_step_max = 1000;
	int n_step_out = 1000;
	
	// setup grid dimensions
	lx = LX_MAX;
	ly = LY_MAX;	
	lz = LZ_MAX;
	
	double 	dx = 10.0/( (double) (lx-1));
	double 	u0 = 2e-3;
	double 	nuphys = 0.008;
	double 	nulb = 1.0/30.0;
	double 	omega = 1.0/(3.0*nulb+0.5);
	double 	dt = dx*dx*nulb/nuphys;		
	double 	es = u0*nulb/0.5;
	double 	re = u0/nulb;

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
	
	// read cell center coordinates */
	fp = fopen("centers.i", "r");
	if (fp == NULL) {
		fprintf(stderr, "Error: cannot open file centers.i.\n");
		return 1;
	}
	for(i = 0; i < NC; i++) {
		fscanf(fp, "%i %lf %lf %lf", &istate, &centers[i][0], &centers[i][1], &centers[i][2]);
	}
	fclose(fp);
	
	// read icosahedron vertices and connectivity
	fp = fopen("icosahedron.i", "r");
	if (fp == NULL) {
		fprintf(stderr, "Error: cannot open file icosahedron.i.\n");
		return 1;
	}
	for(i = 0; i < 12; i++) {
		fscanf(fp, "%i %lf %lf %lf", &istate, &undeformed[i][0], &undeformed[i][1], &undeformed[i][2]);
	}
	for(i = 0; i < 20; i++) {
		fscanf(fp, "%i %i %i %i", &istate, &triangles[i][0], &triangles[i][1], &triangles[i][2]);
	}
	fclose(fp);
	
	refineIcosahedron(undeformed, triangles);
	
	for (i1 = 0; i1 < NC; i1++){
		for (i = 0; i<NV; i++){
			verts[i1][i][0] = undeformed[i][0] + centers[i1][0];
			verts[i1][i][1] = undeformed[i][1] + centers[i1][1];
			verts[i1][i][2] = undeformed[i][2] + centers[i1][2];
		}
	}	
	
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
						cdotu/T + 0.5*(cdotu/T)*(cdotu/T) - uke/T +
						4.5*cdotu*cdotu*cdotu - 9.0*uke*cdotu);
				}
				distr[MC-1][ix][iy][iz] = distr0*(1.0 - uke/T);							
			}
		}
	}	
	
	double err = 0.0;
	
	printf("starting loop...\n");
	/* THE TIMESTEPPING LOOP */	
	for(n_step = 1; n_step <= n_step_max; n_step++) {
	
		printf("%i \n", n_step);
	
		// rest lagu to zeros
 		memset(lagu, 0.0, sizeof(lagu[0][0][0])*NC*NV*3);
 		
 		// interpolate //
		for (i1 = 0; i1 < NC; i1++){
			for (ik = 0; ik < NV; ik++){
				int cornerx = verts[i1][ik][0]/dx - 1;
				int cornery = verts[i1][ik][1]/dx - 1;
				int cornerz = verts[i1][ik][2]/dx - 1;
							
				for (ix = cornerx; ix < cornerx+4; ix++){
					for (iy = cornery; iy < cornery+4; iy++){
						for (iz = cornerz; iz < cornerz+4; iz++){
							double delta3 = delta(ix-verts[i1][ik][0]/dx)*
								delta(iy-verts[i1][ik][1]/dx)*delta(iz-verts[i1][ik][2]/dx);

							rho = 0.0;
							px = 0.0;
							py = 0.0;
							pz = 0.0;
							for (is=0; is<MC; is++)
							{
								double b = distr[is][ix][iy][iz];
								rho += b;
								px += b * ic[is][0];
								py += b * ic[is][1];
								pz += b * ic[is][2];
							}

							if (rho != 0){ 		
								lagu[i1][ik][0] += delta3*px/rho;
								lagu[i1][ik][1] += delta3*py/rho;
								lagu[i1][ik][2] += delta3*pz/rho;
							}
					
						}	
					}
				}
				verts[i1][ik][0] += dt*lagu[i1][ik][0];
				verts[i1][ik][1] += dt*lagu[i1][ik][1];
				verts[i1][ik][2] += dt*lagu[i1][ik][2];
			}
		}
				
		// rest lagf to zeros
 		memset(lagf, 0.0, sizeof(lagf[0][0][0])*NC*NV*3);	
 		
 		// finite element model
		for (i1 = 0; i1 < NC; i1++){
			for (int ik=0; ik<NT; ik++)
			{
				int triI = triangles[ik][0];
				int triJ = triangles[ik][1];
				int triK = triangles[ik][2];
	
				// 1.  deformed configuration
		
				// compute edge vectors
				double bij[3], bik[3], bjk[3];
				dif3( verts[i1][triJ], verts[i1][triI], bij);
				dif3( verts[i1][triK], verts[i1][triI], bik);
				dif3( verts[i1][triK], verts[i1][triJ], bjk);
					
				// compute rotation matrix				
				double be1[3], be2[3], be3[3], bn[3];
			 
				double be1norm = norm(bij);
				for (j=0; j<3; j++){
					be1[j] = bij[j]/be1norm;
				}
			
				cross3(bij, bik, bn);
				double bnnorm = norm(bn);		
				for (j=0; j<3; j++){
					be3[j] = bn[j]/bnnorm;
				}
			
				cross3(be3, be1, be2);
				double be2norm = norm(be2);
				for (j=0; j<3; j++){
					be2[j] /= be2norm;
				}

				double bArea = 0.5*bnnorm;
			
				// map into new coordinates 
				double bxi[3] = { 0.0, 0.0, 0.0};
				double bxj[3] = { dot3(be1,bij), dot3(be2,bij), dot3(be3,bij) };
				double bxk[3] = { dot3(be1,bik), dot3(be2,bik), dot3(be3,bik) };
							
				// 2.  undeformed configuration 
									
				// compute edge vectors
				double lij[3], lik[3], ljk[3];
				dif3( undeformed[triJ], undeformed[triI], lij);
				dif3( undeformed[triK], undeformed[triI], lik);
				dif3( undeformed[triK], undeformed[triJ], ljk);	
							
				// compute rotation matrix	
				double le1[3], le2[3], le3[3], ln[3];
			 
				double le1norm = norm(lij);
				for (j=0; j<3; j++){
					le1[j] = lij[j]/le1norm;
				}
			
				cross3(lij, lik, ln);
				double lnnorm = norm(ln);		
				for (j=0; j<3; j++){
					le3[j] = ln[j]/lnnorm;
				}
			
				cross3(le3, le1, le2);
				double le2norm = norm(le2);
				for (j=0; j<3; j++){
					le2[j] /= le2norm;
				}

				double lArea = 0.5*lnnorm;			
			
				// map into new coordinates 
				double lxi[3] = { 0.0, 0.0, 0.0};
				double lxj[3] = { dot3(le1,lij), dot3(le2,lij), dot3(le3,lij) };
				double lxk[3] = { dot3(le1,lik), dot3(le2,lik), dot3(le3,lik) };
							
				double li = (lxj[1]-lxk[1])*lxi[0] + (lxk[0]-lxj[0])*lxi[1] + lxj[0]*lxk[1] - lxk[0]*lxj[1];
				double lj = (lxk[1]-lxi[1])*lxj[0] + (lxi[0]-lxk[0])*lxj[1] + lxk[0]*lxi[1] - lxi[0]*lxk[1];
				double lk = (lxi[1]-lxj[1])*lxk[0] + (lxj[0]-lxi[0])*lxk[1] + lxi[0]*lxj[1] - lxj[0]*lxi[1];
			
				double avec[3] = {	(lxj[1]-lxk[1])/li, (lxk[1]-lxi[1])/lj, (lxi[1]-lxj[1])/lk };
				double bvec[3] = {	(lxk[0]-lxj[0])/li, (lxi[0]-lxk[0])/lj, (lxj[0]-lxi[0])/lk };
				double uvec[3] = {	bxi[0]-lxi[0], bxj[0]-lxj[0], bxk[0]-lxk[0] };
				double vvec[3] = {	bxi[1]-lxi[1], bxj[1]-lxj[1], bxk[1]-lxk[1] };
			
				double uta = dot3(uvec,avec);
				double vta = dot3(vvec,avec);
				double utb = dot3(uvec,bvec);
				double vtb = dot3(vvec,bvec);
			
				double g11 = 1.0 + 2.0*uta + pow(uta,2.0) + pow(vta,2.0);
				double g22 = 1.0 + 2.0*vtb + pow(utb,2.0) + pow(vtb,2.0);
				double g12 = utb + utb*uta + vta + vtb*vta;
			
				double g11du[3], g11dv[3], g22du[3], g22dv[3], g12du[3], g12dv[3];
				times3(avec, 2.0*(1.0+uta), g11du);
				times3(avec, 2.0*vta, g11dv);
				times3(bvec, 2.0*utb, g22du);
				times3(bvec, 2.0*(1.0+vtb), g22dv);				
				for (j=0; j<3; j++){
					g12du[j] = (1.0+uta)*bvec[j] + utb*avec[j];
					g12dv[j] = vta*bvec[j] + (1.0+vtb)*avec[j];
				}
			
				double kernel0 = pow(g11-g22,2.0)+ 4.0*pow(g12,2.0);
				double kernel1 = 0.5*(g11+g22+sqrt( kernel0 ));
				double kernel2 = 0.5*(g11+g22-sqrt( kernel0 ));
				double lambda1 = sqrt(kernel1);
				double lambda2 = sqrt(kernel2);
			
				double endu[3], endv[3], du1[3], du2[3], dv1[3], dv2[3];
				for (j=0; j<3; j++){
					endu[j] = 2.0*g11*g11du[j] + 2.0*g22*g22du[j] - 2.0*(g11*g22du[j]+g22*g11du[j]) + 8.0*g12*g12du[j];	
					endv[j] = 2.0*g11*g11dv[j] + 2.0*g22*g22dv[j] - 2.0*(g11*g22dv[j]+g22*g11dv[j]) + 8.0*g12*g12dv[j];
					du1[j] = (0.25/lambda1)*( g11du[j]+g22du[j] + 0.5*pow(kernel0,-0.5)*endu[j]);
					du2[j] = (0.25/lambda2)*( g11du[j]+g22du[j] - 0.5*pow(kernel0,-0.5)*endu[j]);
					dv1[j] = (0.25/lambda1)*( g11dv[j]+g22dv[j] + 0.5*pow(kernel0,-0.5)*endv[j]);
					dv2[j] = (0.25/lambda2)*( g11dv[j]+g22dv[j] - 0.5*pow(kernel0,-0.5)*endv[j]);
				}
			
				double wdlam1 = (es/3.0)*( lambda1 - pow(lambda1,-3.0)*pow(lambda2,-2.0) );
				double wdlam2 = (es/3.0)*( lambda2 - pow(lambda1,-2.0)*pow(lambda2,-3.0) );

				double fxl[3], fyl[3];
				for (j=0; j<3; j++){
					fxl[j] = lArea * ( wdlam1*du1[j] + wdlam2*du2[j]);
					fyl[j] = lArea * ( wdlam1*dv1[j] + wdlam2*dv2[j]); 
				}			

				double fi[3], fj[3], fk[3];
				for (j=0; j<3; j++){
					fi[j] = be1[j]*fxl[0] + be2[j]*fyl[0];
					fj[j] = be1[j]*fxl[1] + be2[j]*fyl[1];
					fk[j] = be1[j]*fxl[2] + be2[j]*fyl[2];
				}
			
				for (j=0; j<3; j++){
					lagf[i1][triI][j] -= fi[j];
					lagf[i1][triJ][j] -= fj[j];
					lagf[i1][triK][j] -= fk[j];
				}
			}
		}
		
		// rest distr_f to zeros
 		memset(eulf, 0.0, sizeof(eulf[0][0][0][0])*3*LX_MAX*LY_MAX*LZ_MAX);
 		
 		// spread //
		for (i1 = 0; i1 < NC; i1++){
			for (ik = 0; ik < NV; ik++){
				int cornerx = verts[i1][ik][0]/dx - 1;
				int cornery = verts[i1][ik][1]/dx - 1;
				int cornerz = verts[i1][ik][2]/dx - 1;
				double fx =  lagf[i1][ik][0];
				double fy =  lagf[i1][ik][1];
				double fz =  lagf[i1][ik][2];
			
				for (ix = cornerx; ix < cornerx+4; ix++){
					for (iy = cornery; iy < cornery+4; iy++){
						for (iz = cornerz; iz < cornerz+4; iz++){
							double delta3 = delta(ix-verts[i1][ik][0]/dx)*
								delta(iy-verts[i1][ik][1]/dx)*delta(iz-verts[i1][ik][2]/dx);
						
							eulf[ix][iy][iz][0] += delta3*fx/(dx*dx*dx);	
							eulf[ix][iy][iz][1] += delta3*fy/(dx*dx*dx);
							eulf[ix][iy][iz][2] += delta3*fz/(dx*dx*dx);
						
						}
					}
				}
			}
		}
	
		/* collision - ibm version */
		for(ix = 0; ix < lx; ix++) {
		    for(iy = 0; iy < ly; iy++) {
		        for(iz = 0; iz < lz; iz++) {
		        
		        	fx = eulf[ix][iy][iz][0];
		        	fy = eulf[ix][iy][iz][1];
		        	fz = eulf[ix][iy][iz][2];

					// compute u* ala Guo, Zheng, Shi (PRE, 2002)
					rho = px = py = pz = 0.0;
					for(is = 0; is < MC; is++) {
						rho += distr[is][ix][iy][iz];
						px += distr[is][ix][iy][iz]*ic[is][0];
						py += distr[is][ix][iy][iz]*ic[is][1];
						pz += distr[is][ix][iy][iz]*ic[is][2];
					}
					ux = px / rho + 0.5*dt*fx;
					uy = py / rho + 0.5*dt*fy;
					uz = pz / rho + 0.5*dt*fz;

					uke = 0.5*(ux*ux + uy*uy + uz*uz);
					
					// compute distr_eq with u*
					for(is = 0; is < MC; is++) {
						cdotu = ic[is][0]*ux + ic[is][1]*uy + ic[is][2]*uz;
						distr_eq[is] = distr1*rho*idg[is]*(1.0 + 
							cdotu/T + 0.5*(cdotu/T)*(cdotu/T) - uke/T);
					}
					
					// compute distr_f with eulf
					double fdotu = fx*ux +fy*uy + fz*uz;
					for(is = 0; is < MC-1; is++) {
						fdotc = ic[is][0]*fx + ic[is][1]*fy + ic[is][2]*fz;
						cdotu = ic[is][0]*ux + ic[is][1]*uy + ic[is][2]*uz;
						distr_f[is] = (1.0-0.5*omega)*distr1*     
							(fdotc/T - fdotu/T + cdotu*fdotc/(T*T) );
					}
					distr_f[MC-1] = (1.0-0.5*omega)*distr0*(-fdotu/T);

					for(is = 0; is < MC; is++) {
						delNi = -omega * (distr[is][ix][iy][iz] - distr_eq[is]) + dt*distr_f[is];
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
                        
                    	fprintf(vtk_fp, "%lf\n",rho);
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

                    	fprintf(vtk_fp, "%lf %lf %lf\n",ux,uy,uz);
                    }
                }
        	}
        	
        	fprintf(vtk_fp, "VECTORS coordinates double \n");
        	for(iz = 0; iz < lz; iz++) {
	        	for(iy = 0; iy < ly; iy++) {
                	for(ix = 0; ix < lx; ix++) {

                    	fprintf(vtk_fp, "%lf %lf %lf\n",ix*dx,iy*dx,iz*dx);
                    }
                }
        	}
        	
        	fprintf(vtk_fp, "VECTORS force double \n");
        	for(iz = 0; iz < lz; iz++) {
	        	for(iy = 0; iy < ly; iy++) {
                	for(ix = 0; ix < lx; ix++) {

                    	fprintf(vtk_fp, "%1.15lf %1.15lf %1.15lf\n",eulf[ix][iy][iz][0],eulf[ix][iy][iz][1],eulf[ix][iy][iz][2]);
                    }
                }
        	}

	        fclose(vtk_fp);  
	        
	        /////////////////////////////////////////////
	        
	        for (i1 = 0; i1 < NC; i1++){
				sprintf(fname,"cell_%d_%i.vtk",n_step,i1);
				vtk_fp = fopen(fname, "w");
				if (vtk_fp == NULL) {
						fprintf(stderr, "Error: cannot open file cell.vtk. \n");
						return 1;
				}
			
				fprintf(vtk_fp, "# vtk DataFile Version 2.0\n");
				fprintf(vtk_fp, "Essai\n");
				fprintf(vtk_fp, "ASCII\n");
				fprintf(vtk_fp, "DATASET UNSTRUCTURED_GRID\n");
				fprintf(vtk_fp, "POINTS %i double \n", NV);

				for (int ik = 0; ik < NV; ik++){
					fprintf(vtk_fp, "%lf %lf %lf\n",verts[i1][ik][0]/dx,verts[i1][ik][1]/dx,verts[i1][ik][2]/dx);
				}

				fprintf(vtk_fp, "CELLS %i %i \n", NT, 4*NT);
				for (int ik = 0; ik < NT; ik++)
				{
					fprintf(vtk_fp, "3 %i %i %i \n",triangles[ik][0],triangles[ik][1],triangles[ik][2]);
				}
			  
				fprintf(vtk_fp, "CELL_TYPES %i \n", NT);
				for (int ik = 0; ik < NT; ik++)
				{
					fprintf(vtk_fp, "5\n");
				}  
			  
				fprintf(vtk_fp, "POINT_DATA %i \n", NV);  
				fprintf(vtk_fp, "VECTORS force double\n"); 
				for (int ik = 0; ik < NV; ik++)
				{
					fprintf(vtk_fp, "%1.12lf %1.12lf %1.12lf \n",lagf[i1][ik][0],lagf[i1][ik][1],lagf[i1][ik][2]);
				}
				fprintf(vtk_fp, "VECTORS velocity double\n"); 
				for (int ik = 0; ik < NV; ik++)
				{
					fprintf(vtk_fp, "%1.8lf %1.8lf %1.8lf \n",lagu[i1][ik][0],lagu[i1][ik][1],lagu[i1][ik][2]);
				}
				fclose(vtk_fp); 
			}
			
		}
		
	
	}

	return 0;
}

