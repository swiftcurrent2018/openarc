#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sys/time.h>
#include <math.h>
#include "openacc.h"
//using namespace std;
#define STR_SIZE	256

#ifndef VERIFICATION
#define VERIFICATION 0
#endif

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001F
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5F
//#define VERBOSE
#define DEBUG

#ifndef ROW_SIZE
#define ROW_SIZE 64
#endif

#ifndef COL_SIZE
#define COL_SIZE 64
#endif

#define TEMP_SIZE   (ROW_SIZE*COL_SIZE)
#define POWER_SIZE  (ROW_SIZE*COL_SIZE)
#define RESULT_SIZE (ROW_SIZE*COL_SIZE)

#ifdef _OPENARC_
#if ROW_SIZE == 64
#pragma openarc #define ROW_SIZE 64
#elif ROW_SIZE == 512
#pragma openarc #define ROW_SIZE 512
#elif ROW_SIZE == 1024
#pragma openarc #define ROW_SIZE 1024
#elif ROW_SIZE == 4096
#pragma openarc #define ROW_SIZE 4096
#endif


#if COL_SIZE == 64
#pragma openarc #define COL_SIZE 64
#elif COL_SIZE == 512
#pragma openarc #define COL_SIZE 512
#elif COL_SIZE == 1024
#pragma openarc #define COL_SIZE 1024
#elif COL_SIZE == 4096
#pragma openarc #define COL_SIZE 4096
#endif

#pragma openarc #define TEMP_SIZE   (ROW_SIZE*COL_SIZE)
#pragma openarc #define POWER_SIZE  (ROW_SIZE*COL_SIZE)
#pragma openarc #define RESULT_SIZE (ROW_SIZE*COL_SIZE)

#endif

/* chip parameters	*/
double t_chip = 0.0005F;
double chip_height = 0.016F;
double chip_width = 0.016F;
/* ambient temperature, assuming no package at all	*/
double amb_temp = 80.0F;

int num_omp_threads;

double gettime() {
		struct timeval t;
		gettimeofday(&t,0);
		return t.tv_sec+t.tv_usec*1e-6;
}


/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(double *result, int num_iterations, double *temp, double *power, int row, int col) 
{
#ifdef VERBOSE
		int i = 0;
#else
		int i = 0;
#endif

#ifdef DEBUG
		double start_time, end_time;
#endif

		double grid_height = chip_height / row;
		double grid_width = chip_width / col;

		double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
		double Rx = grid_width / (2.0F * K_SI * t_chip * grid_height);
		double Ry = grid_height / (2.0F * K_SI * t_chip * grid_width);
		double Rz = t_chip / (K_SI * grid_height * grid_width);

		double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
		double step = PRECISION / max_slope;

		//////////////////////////////////////////////
		// Added for inlining of single_iteration() //
		//////////////////////////////////////////////
		double delta;
		int r, c;
		int rc;
		double tSum;

#ifdef VERBOSE
		fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
		fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
#endif

#ifdef DEBUG
		start_time = gettime();
#endif

#pragma acc data copy(temp[0:TEMP_SIZE]) create(result[0:RESULT_SIZE]) copyin(power[0:POWER_SIZE])
		for (i = 0; i < num_iterations ; i++)
		{
				//single_iteration(result, temp, power, row, col, Cap, Rx, Ry, Rz, step);
#pragma acc parallel 
{
				#pragma acc loop gang
				//#pragma openarc transform permute(c,r)
				for (r = 0; r < ROW_SIZE; r++) {
				#pragma acc loop worker
				for (c = 0; c < COL_SIZE; c++) {
						/*	Corner 1	*/
						if ( (r == 0) && (c == 0) ) {
								delta = (step / Cap) * (power[0] +
												(temp[1] - temp[0]) / Rx +
												(temp[col] - temp[0]) / Ry +
												(amb_temp - temp[0]) / Rz);
						}	/*	Corner 2	*/
						else if ((r == 0) && (c == col-1)) {
								delta = (step / Cap) * (power[c] +
												(temp[c-1] - temp[c]) / Rx +
												(temp[c+col] - temp[c]) / Ry +
												(amb_temp - temp[c]) / Rz);
						}	/*	Corner 3	*/
						else if ((r == row-1) && (c == col-1)) {
								delta = (step / Cap) * (power[r*col+c] + 
												(temp[r*col+c-1] - temp[r*col+c]) / Rx + 
												(temp[(r-1)*col+c] - temp[r*col+c]) / Ry + 
												(amb_temp - temp[r*col+c]) / Rz);					
						}	/*	Corner 4	*/
						else if ((r == row-1) && (c == 0)) {
								delta = (step / Cap) * (power[r*col] + 
												(temp[r*col+1] - temp[r*col]) / Rx + 
												(temp[(r-1)*col] - temp[r*col]) / Ry + 
												(amb_temp - temp[r*col]) / Rz);
						}	/*	Edge 1	*/
						else if (r == 0) {
								delta = (step / Cap) * (power[c] + 
												(temp[c+1] + temp[c-1] - 2.0F*temp[c]) / Rx + 
												(temp[col+c] - temp[c]) / Ry + 
												(amb_temp - temp[c]) / Rz);
						}	/*	Edge 2	*/
						else if (c == col-1) {
								delta = (step / Cap) * (power[r*col+c] + 
												(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0F*temp[r*col+c]) / Ry + 
												(temp[r*col+c-1] - temp[r*col+c]) / Rx + 
												(amb_temp - temp[r*col+c]) / Rz);
						}	/*	Edge 3	*/
						else if (r == row-1) {
								delta = (step / Cap) * (power[r*col+c] + 
												(temp[r*col+c+1] + temp[r*col+c-1] - 2.0F*temp[r*col+c]) / Rx + 
												(temp[(r-1)*col+c] - temp[r*col+c]) / Ry + 
												(amb_temp - temp[r*col+c]) / Rz);
						}	/*	Edge 4	*/
						else if (c == 0) {
								delta = (step / Cap) * (power[r*col] + 
												(temp[(r+1)*col] + temp[(r-1)*col] - 2.0F*temp[r*col]) / Ry + 
												(temp[r*col+1] - temp[r*col]) / Rx + 
												(amb_temp - temp[r*col]) / Rz);
						}	/*	Inside the chip	*/
						else {
								delta = (step / Cap) * (power[r*col+c] + 
												(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0F*temp[r*col+c]) / Ry + 
												(temp[r*col+c+1] + temp[r*col+c-1] - 2.0F*temp[r*col+c]) / Rx + 
												(amb_temp - temp[r*col+c]) / Rz);
						}

						/*	Update Temperatures	*/
						result[r*col+c] =temp[r*col+c]+ delta;
				}
				}
			}
			
		#pragma acc parallel 
		{
				#pragma acc loop gang
				//#pragma openarc transform permute(c,r)
				for (r = 0; r < ROW_SIZE; r++) {
				#pragma acc loop worker
				for (c = 0; c < COL_SIZE; c++) {
						temp[r*col+c]=result[r*col+c];
				}
				}
		}	
}		

//Fake computation to measure timing of unified memory version
		tSum = 0.0;
		for(r=0; r<TEMP_SIZE; r++) {
			tSum += temp[r];
		}
		printf("Sum of temp: %lf\n", tSum);
#ifdef DEBUG
		end_time = gettime();
		printf("Accelerator Elapsed Time = %lf sec.\n", end_time - start_time);
#endif

#ifdef VERBOSE
		//fprintf(stdout, "iteration %d\n", i++);
#endif
}


void compute_tran_temp_CPU(double *result, int num_iterations, double *temp, double *power, int row, int col) 
{
#ifdef VERBOSE
		int i = 0;
#else
		int i = 0;
#endif

#ifdef DEBUG
		double start_time, end_time;
#endif

		double grid_height = chip_height / row;
		double grid_width = chip_width / col;

		double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
		double Rx = grid_width / (2.0F * K_SI * t_chip * grid_height);
		double Ry = grid_height / (2.0F * K_SI * t_chip * grid_width);
		double Rz = t_chip / (K_SI * grid_height * grid_width);

		double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
		double step = PRECISION / max_slope;

		//////////////////////////////////////////////
		// Added for inlining of single_iteration() //
		//////////////////////////////////////////////
		double delta;
		int r, c;
		int rc;

#ifdef VERBOSE
		fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
		fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
#endif

#ifdef DEBUG
		start_time = gettime();
#endif

		for (i = 0; i < num_iterations ; i++)
		{
				//single_iteration(result, temp, power, row, col, Cap, Rx, Ry, Rz, step);
				for (rc = 0; rc < ROW_SIZE*COL_SIZE; rc++) {
						c = rc%COL_SIZE;
						r = rc/COL_SIZE;
						/*	Corner 1	*/
						if ( (r == 0) && (c == 0) ) {
								delta = (step / Cap) * (power[0] +
												(temp[1] - temp[0]) / Rx +
												(temp[col] - temp[0]) / Ry +
												(amb_temp - temp[0]) / Rz);
						}	/*	Corner 2	*/
						else if ((r == 0) && (c == col-1)) {
								delta = (step / Cap) * (power[c] +
												(temp[c-1] - temp[c]) / Rx +
												(temp[c+col] - temp[c]) / Ry +
												(amb_temp - temp[c]) / Rz);
						}	/*	Corner 3	*/
						else if ((r == row-1) && (c == col-1)) {
								delta = (step / Cap) * (power[r*col+c] + 
												(temp[r*col+c-1] - temp[r*col+c]) / Rx + 
												(temp[(r-1)*col+c] - temp[r*col+c]) / Ry + 
												(amb_temp - temp[r*col+c]) / Rz);					
						}	/*	Corner 4	*/
						else if ((r == row-1) && (c == 0)) {
								delta = (step / Cap) * (power[r*col] + 
												(temp[r*col+1] - temp[r*col]) / Rx + 
												(temp[(r-1)*col] - temp[r*col]) / Ry + 
												(amb_temp - temp[r*col]) / Rz);
						}	/*	Edge 1	*/
						else if (r == 0) {
								delta = (step / Cap) * (power[c] + 
												(temp[c+1] + temp[c-1] - 2.0F*temp[c]) / Rx + 
												(temp[col+c] - temp[c]) / Ry + 
												(amb_temp - temp[c]) / Rz);
						}	/*	Edge 2	*/
						else if (c == col-1) {
								delta = (step / Cap) * (power[r*col+c] + 
												(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0F*temp[r*col+c]) / Ry + 
												(temp[r*col+c-1] - temp[r*col+c]) / Rx + 
												(amb_temp - temp[r*col+c]) / Rz);
						}	/*	Edge 3	*/
						else if (r == row-1) {
								delta = (step / Cap) * (power[r*col+c] + 
												(temp[r*col+c+1] + temp[r*col+c-1] - 2.0F*temp[r*col+c]) / Rx + 
												(temp[(r-1)*col+c] - temp[r*col+c]) / Ry + 
												(amb_temp - temp[r*col+c]) / Rz);
						}	/*	Edge 4	*/
						else if (c == 0) {
								delta = (step / Cap) * (power[r*col] + 
												(temp[(r+1)*col] + temp[(r-1)*col] - 2.0F*temp[r*col]) / Ry + 
												(temp[r*col+1] - temp[r*col]) / Rx + 
												(amb_temp - temp[r*col]) / Rz);
						}	/*	Inside the chip	*/
						else {
								delta = (step / Cap) * (power[r*col+c] + 
												(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0F*temp[r*col+c]) / Ry + 
												(temp[r*col+c+1] + temp[r*col+c-1] - 2.0F*temp[r*col+c]) / Rx + 
												(amb_temp - temp[r*col+c]) / Rz);
						}

						/*	Update Temperatures	*/
						result[r*col+c] =temp[r*col+c]+ delta;
				}

				for (rc = 0; rc < ROW_SIZE*COL_SIZE; rc++) {
						c = rc%COL_SIZE;
						r = rc/COL_SIZE;
						temp[r*col+c]=result[r*col+c];
				}
		}	
#ifdef DEBUG
		end_time = gettime();
		printf("Main Comp. Time CPU = %lf sec.\n", end_time - start_time);
#endif

#ifdef VERBOSE
		//fprintf(stdout, "iteration %d\n", i++);
#endif
}

void fatal(char *s)
{
		fprintf(stderr, "error: %s\n", s);
		exit(1);
}

void writeoutput(double *vect, int grid_rows, int grid_cols, char *file){

		int i,j, index=0;
		FILE *fp;
		char str[STR_SIZE];

		if( (fp = fopen(file, "w" )) == 0 ) 
				printf( "The file was not opened\n" );


		for (i=0; i < grid_rows; i++) 
				for (j=0; j < grid_cols; j++)
				{   

						sprintf(str, "%d\t%lf\n", index, vect[i*grid_cols+j]);
						fputs(str,fp);
						index++;
				}   

		fclose(fp); 
}


void read_input(double *vect, int grid_rows, int grid_cols, char *file)
{
		int i, index;
		FILE *fp;
		char str[STR_SIZE];
		double val;

		fp = fopen (file, "r");
		if (!fp)
				fatal ("file could not be opened for reading");

		for (i=0; i < grid_rows * grid_cols; i++) {
				fgets(str, STR_SIZE, fp);
				if (feof(fp))
						fatal("not enough lines in file");
				if ((sscanf(str, "%lf", &val) != 1) )
				//if ((sscanf(str, "%f", &val) != 1) )
						fatal("invalid file format");
				vect[i] = val;
		}

		fclose(fp);	
}

void usage(int argc, char **argv)
{
		fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
		fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
		fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
		fprintf(stderr, "\t<sim_time>   - number of iterations\n");
		fprintf(stderr, "\t<no. of threads>   - number of threads\n");
		fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
		fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
		fprintf(stderr, "\t<output_file> - name of the output file\n");
		exit(1);
}

int main(int argc, char **argv)
{
		int grid_rows, grid_cols, sim_time, i;
		double *temp, *power, *result;
		char *tfile, *pfile, *ofile;

		double start_time, end_time;

		/* check validity of inputs	*/
		if (argc != 8)
				usage(argc, argv);
		if ((grid_rows = atoi(argv[1])) <= 0 ||
						(grid_cols = atoi(argv[2])) <= 0 ||
						(sim_time = atoi(argv[3])) <= 0 || 
						(num_omp_threads = atoi(argv[4])) <= 0
		   )
				usage(argc, argv);

		start_time = gettime();

		/* allocate memory for the temperature and power arrays	*/
/*
		temp = (double *) calloc (grid_rows * grid_cols, sizeof(double));
		power = (double *) calloc (grid_rows * grid_cols, sizeof(double));
		result = (double *) calloc (grid_rows * grid_cols, sizeof(double));
*/
		temp = (double *) acc_create_unified (NULL, grid_rows * grid_cols * sizeof(double));
		power = (double *) acc_create_unified (NULL, grid_rows * grid_cols * sizeof(double));
		result = (double *) acc_create_unified (NULL, grid_rows * grid_cols * sizeof(double));
		if(!temp || !power)
				fatal("unable to allocate memory");

		/* read initial temperatures and input power	*/
		tfile = argv[5];
		pfile = argv[6];
		ofile = argv[7];
		read_input(temp, grid_rows, grid_cols, tfile);
		read_input(power, grid_rows, grid_cols, pfile);

		printf("Start computing the transient temperature\n");
		compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols);
		printf("Ending simulation\n");
		/* output results	*/
#ifdef VERBOSE
		fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
		for(i=0; i < grid_rows * grid_cols; i++)
				fprintf(stdout, "%d\t%g\n", i, temp[i]);
#endif

		writeoutput(temp, grid_rows, grid_cols, ofile);

		if(VERIFICATION) {
			double *tempCPU, *powerCPU, *resultCPU;
			double deltaL2Norm = 0;
		  double nonAccL2Norm = 0;
			double L2Norm;

			tempCPU = (double *) calloc (grid_rows * grid_cols, sizeof(double));
			powerCPU = (double *) calloc (grid_rows * grid_cols, sizeof(double));
			resultCPU = (double *) calloc (grid_rows * grid_cols, sizeof(double));
		
			read_input(tempCPU, grid_rows, grid_cols, tfile);
			read_input(powerCPU, grid_rows, grid_cols, pfile);

			printf("Start computing the transient temperature\n");
			compute_tran_temp_CPU(resultCPU,sim_time, tempCPU, powerCPU, grid_rows, grid_cols);
			printf("Ending simulation\n");

			
		  for (i = 0; i < grid_rows * grid_cols; ++i) {
		      double d = tempCPU[i] - temp[i];
		      deltaL2Norm += d * d;
		      nonAccL2Norm += tempCPU[i] * tempCPU[i];
		  }

		  L2Norm = sqrt(deltaL2Norm / nonAccL2Norm);
			
			if (L2Norm < 1e-6)
        printf("Verification: Successful\n");
		  else
		      printf("Verification: Failed\n");

			free(tempCPU);
			free(powerCPU);
			free(resultCPU);
		}




		/* cleanup	*/
/*
		free(temp);
		free(power);
*/
		acc_delete_unified(temp, 0);
		acc_delete_unified(power, 0);

		end_time = gettime();
		printf("Total Execution Time = %lf sec.\n", end_time - start_time);

		return 0;
}

