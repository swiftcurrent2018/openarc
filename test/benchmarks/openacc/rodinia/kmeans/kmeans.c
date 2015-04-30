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
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
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
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include "getopt.h"

#include "kmeans.h"
#include "openacc.h"

#define NULLZ 0

extern double wtime(void);

double gettime() {
  struct timeval t;
  gettimeofday(&t,0);
  return t.tv_sec+t.tv_usec*1e-6;
}


int num_omp_threads = 1;

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "Usage: %s [switches] -i filename\n"
        "       -i filename     		: file containing data to be clustered\n"
        "       -b                 	: input file is in binary format\n"
		"       -k                 	: number of clusters (default is 5) \n"
        "       -t threshold		: threshold value\n"
		"       -n no. of threads	: number of threads";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     nclusters=5;
           char   *filename = 0;           
           float  *buf;
           float (*attributes)[_NATTRIBUTES];
           float (*cluster_centres)[_NATTRIBUTES]=NULLZ;
           int     i, j;
                
           int     numAttributes;
           int     numObjects;        
           char    line[1024];           
           int     isBinaryFile = 0;
           int     nloops = 1;
           float   threshold = 0.001;
		   double  start_time1, end_time1;		   
			FILE *fp;

	while ( (opt=getopt(argc,argv,"i:k:t:b:n:?"))!= EOF) {
		switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'k': nclusters = atoi(optarg);
                      break;			
			case 'n': num_omp_threads = atoi(optarg);
					  break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }


    if (filename == 0) usage(argv[0]);

    numAttributes = numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/
   
    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &numObjects,    sizeof(int));
        read(infile, &numAttributes, sizeof(int));
   

        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
        //attributes = (float (*)[_NATTRIBUTES]) malloc(numObjects*numAttributes*sizeof(float));
        attributes = (float (*)[_NATTRIBUTES]) acc_create_unified(NULL, numObjects*numAttributes*sizeof(float));

        read(infile, buf, numObjects*numAttributes*sizeof(float));

        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULLZ) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        while (fgets(line, 1024, infile) != NULLZ)
            if (strtok(line, " \t\n") != 0)
                numObjects++;
        rewind(infile);
        while (fgets(line, 1024, infile) != NULLZ) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): numAttributes = 1; */
                while (strtok(NULLZ, " ,\t\n") != NULLZ) numAttributes++;
                break;
            }
        }
     

        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
        //attributes = (float (*)[_NATTRIBUTES]) malloc(numObjects*numAttributes*sizeof(float));
        attributes = (float (*)[_NATTRIBUTES]) acc_create_unified(NULL, numObjects*numAttributes*sizeof(float));
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULLZ) {
            if (strtok(line, " \t\n") == NULLZ) continue; 
            for (j=0; j<numAttributes; j++) {
                buf[i] = atof(strtok(NULLZ, " ,\t\n"));
                i++;
            }
        }
        fclose(infile);
    }     
	printf("I/O completed\n");	

	memcpy(attributes, buf, numObjects*numAttributes*sizeof(float));

	start_time1 = gettime();
    for (i=0; i<nloops; i++) {
        
        cluster_centres = NULLZ;
        cluster(numObjects,
                numAttributes,
                attributes,           /* [numObjects][numAttributes] */                
                nclusters,
                threshold,
                &cluster_centres   
               );
     
    }
    end_time1 = gettime();
	

	printf("number of Objects %d\n",numObjects); 
	printf("number of Clusters %d\n",nclusters); 
	printf("number of Attributes %d\n\n",numAttributes); 
	printf("Accelerator Elapsed Time = %f sec\n", (end_time1-start_time1));
#ifdef DEBUG
  if( (fp = fopen("kmeans.out", "w")) == NULLZ ) { 
    printf("Can not open %s\n", "kmeans.out");
    exit(1);
  }
  fprintf(fp, "\n================= Centroid Coordinates =================\n");

  for (i=0; i< nclusters; i++) {
        fprintf(fp, "%d: ", i);
        for (j=0; j<numAttributes; j++)
            fprintf(fp, "%f ", cluster_centres[i][j]);
        fprintf(fp, "\n\n");
  }
    fclose(fp);
#endif


    //free(attributes);
    acc_delete_unified(attributes, 0);
    //free(cluster_centres);
    acc_delete_unified(cluster_centres, 0);
    free(buf);
    return(0);
}

