/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifndef _NPOINTS
#define _NPOINTS 819200
#endif

#ifndef _UNROLLFAC_
#define _UNROLLFAC_ 100
#endif

#define _NTHREADS	(_NPOINTS/_UNROLLFAC_)

#ifdef _OPENARC_

#if _NPOINTS == 204800
#pragma openarc #define _NPOINTS 204800
#elif _NPOINTS == 494020
#pragma openarc #define _NPOINTS 494020
#elif _NPOINTS == 819200
#pragma openarc #define _NPOINTS 819200
#endif

#if _UNROLLFAC_ == 1
#pragma openarc #define _UNROLLFAC_ 1
#elif _UNROLLFAC_ == 2
#pragma openarc #define _UNROLLFAC_ 2
#elif _UNROLLFAC_ == 4
#pragma openarc #define _UNROLLFAC_ 4
#elif _UNROLLFAC_ == 5
#pragma openarc #define _UNROLLFAC_ 5
#elif _UNROLLFAC_ == 800
#pragma openarc #define _UNROLLFAC_ 800
#elif _UNROLLFAC_ == 10
#pragma openarc #define _UNROLLFAC_ 10
#elif _UNROLLFAC_ == 100
#pragma openarc #define _UNROLLFAC_ 100
#endif

#pragma openarc #define _NATTRIBUTES 34

#pragma openarc #define _NCLUSTERS 5

#pragma openarc #define _NTHREADS (_NPOINTS/_UNROLLFAC_)

#endif

extern double wtime(void);


/*----< kmeans_clustering() >---------------------------------------------*/
PAType kmeans_clustering(float feature[_NPOINTS][_NATTRIBUTES],    /* in: [npoints][nfeatures] */
				int     nfeatures,
				int     npoints,
				int     nclusters,
				float   threshold,
				int    membership[_NPOINTS]) /* out: [npoints] */
{

		int      i, j, k, n=0, index, loop=0;
		int     *new_centers_len;			/* [nclusters]: no. of points in each cluster */
		float  (*new_centers)[_NATTRIBUTES];				/* [nclusters][nfeatures] */
		float  (*clusters)[_NATTRIBUTES];					/* out: [nclusters][nfeatures] */
		float    delta;

		double   timing;

		int      nthreads;
		int    (*partial_new_centers_len)[_NCLUSTERS];
		float (*partial_new_centers)[_NCLUSTERS][_NATTRIBUTES];

		/////////////////////////////////////////////
		// Added for inlining find_nearest_point() //
		/////////////////////////////////////////////
		int index_fnp, i_fnp;
		float max_dist=FLT_MAX;
		int i_ed;

		///////////////////////////////////////////////
		// Added for unrolling of the parallel loop. //
		///////////////////////////////////////////////
		int tid, ii;

		nthreads = npoints/_UNROLLFAC_;

		/* allocate space for returning variable clusters[] */
		clusters = (float (*)[_NATTRIBUTES])  malloc(nclusters * nfeatures * sizeof(float));

		/* randomly pick cluster centers */
		for (i=0; i<nclusters; i++) {
				//n = (int)rand() % npoints;
				for (j=0; j<nfeatures; j++)
						clusters[i][j] = feature[n][j];
				n++;
		}

		for (i=0; i<npoints; i++)
				membership[i] = -1;

		/* need to initialize new_centers_len and new_centers[0] to all 0 */
		new_centers_len = (int*) calloc(nclusters, sizeof(int));

		new_centers = (float (*)[_NATTRIBUTES])  calloc(nclusters * nfeatures, sizeof(float));

		partial_new_centers_len = (int (*)[_NCLUSTERS])  calloc(nthreads*nclusters, sizeof(int));

		partial_new_centers =(float (*)[_NCLUSTERS][_NATTRIBUTES]) calloc(nthreads*nclusters*nfeatures, sizeof(float));

		printf("num of threads = %d\n", nthreads);
#pragma acc data copyin (feature[0:_NPOINTS][0:_NATTRIBUTES], membership[0:_NPOINTS]) create(clusters[0:_NCLUSTERS][0:_NATTRIBUTES])
		do {
				delta = 0.0F;
#pragma acc update device(clusters)
#pragma acc kernels loop gang worker independent \
				private(i, index, index_fnp, max_dist) \
				reduction(+:new_centers[0:_NCLUSTERS][0:_NATTRIBUTES],new_centers_len[0:_NCLUSTERS]) 
#pragma openarc cuda sharedRW(new_centers_len)
				for(tid=0; tid<nthreads; tid++) {
#pragma acc loop seq
						for (ii=0; ii<_UNROLLFAC_; ii++) {
								i = tid + ii*nthreads;
								/* find the index of nestest cluster centers */					
								//index = find_nearest_point(feature[i],
								//        nfeatures,
								//        clusters,
								//        nclusters);				

								max_dist = FLT_MAX;
								/* find the cluster center id with min distance to pt */
								for (i_fnp=0; i_fnp<nclusters; i_fnp++) {
										float dist;
										//dist = euclid_dist_2(feature[i_fnp], clusters[i_fnp], nfeatures);  /* no need square root */
										dist = 0.0F;
										for (i_ed=0; i_ed<nfeatures; i_ed++)
												dist += (feature[i][i_ed]-clusters[i_fnp][i_ed]) * (feature[i][i_ed]-clusters[i_fnp][i_ed]);
										if (dist < max_dist) {
												max_dist = dist;
												index_fnp    = i_fnp;
										}
								}
								index = index_fnp;

								/* if membership changes, increase delta by 1 */
								if (membership[i] != index) delta += 1.0F;

								/* assign the membership to object i */
								membership[i] = index;

								/* update new cluster centers : sum of all objects located
								   within */
								new_centers_len[index]++;				
								for (j=0; j<nfeatures; j++)
										new_centers[index][j] += feature[i][j];
						}
				} /* end of #pragma omp parallel for */

				/* replace old cluster centers with new_centers */
				for (i=0; i<nclusters; i++) {
						for (j=0; j<nfeatures; j++) {
								if (new_centers_len[i] > 0)
										clusters[i][j] = new_centers[i][j] / new_centers_len[i];
								new_centers[i][j] = 0.0F;   /* set back to 0 */
						}
						new_centers_len[i] = 0;   /* set back to 0 */
				}

		} while (delta > threshold && loop++ < 500);
		printf("loop count: %d\n", loop);

		free(new_centers);
		free(new_centers_len);

		return clusters;
}

