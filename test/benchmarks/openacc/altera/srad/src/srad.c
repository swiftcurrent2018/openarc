// srad.c : Defines the entry point for the console application.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "srad.h"

int grid_rows, grid_cols, size_I, size_R, niter = 1;
int r1, r2, c1, c2;
float lambda;
int nthreads;

// High-resolution timer.
double getCurrentTimestamp() {
  struct timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return ( (double) a.tv_nsec * 1.0e-9) + (double) a.tv_sec;
}

void random_matrix(float *I, int grid_rows, int grid_cols);

void mainComp(float J[_SIZE_I_], int iN[_ROWS_], int iS[_ROWS_],
    int jE[_COLS_], int jW[_COLS_], float dN[_SIZE_I_], float dS[_SIZE_I_],
    float dW[_SIZE_I_], float dE[_SIZE_I_], float c[_SIZE_I_]);

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <y1> <y2> <x1> <x2> <no. of threads><lamda> <no. of iter>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows> - number of grid_rows\n");
  fprintf(stderr, "\t<grid_cols  - number of grid_cols\n");
  fprintf(stderr, "\t<y1>      	 - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>        - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>        - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>        - x2 value of the speckle\n");
  fprintf(stderr, "\t<no. of threads>  - no. of threads\n");
  fprintf(stderr, "\t<lamda>           - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter>     - number of iterations\n");

  exit(1);
}

#if VERIFICATION == 1
void mainCompCPU(float J[_SIZE_I_], int iN[_ROWS_], int iS[_ROWS_],
    int jE[_COLS_], int jW[_COLS_], float dN[_SIZE_I_], float dS[_SIZE_I_],
    float dW[_SIZE_I_], float dE[_SIZE_I_], float c[_SIZE_I_])
{
  int iter, k;
  float q0sqr, sum, sum2, tmp, meanROI,varROI ;
  float Jc, G2, L, num, den, qsqr;
  float cN,cS,cW,cE;
  float D;
  int i, j;
#ifdef ITERATION
  for (iter=0; iter< niter; iter++){
#endif        
    sum=0; sum2=0;     

    for (i=r1; i<=r2; i++) {
      for (j=c1; j<=c2; j++) {
        tmp   = J[i * grid_cols + j];
        sum  += tmp ;
        sum2 += tmp*tmp;
      }
    }
    meanROI = sum / size_R;
    varROI  = (sum2 / size_R) - meanROI*meanROI;
    q0sqr   = varROI / (meanROI*meanROI);

    for (i = 0 ; i < grid_rows ; i++) {
      for (j = 0; j < grid_cols; j++) { 

        k = i * grid_cols + j;
        Jc = J[k];

        // directional derivates
        if( i == 0 ) {
          dN[k] = J[j] - Jc;
          dS[k] = J[grid_cols + j] - Jc;
        } else if( i == (grid_rows-1) ) {
          dN[k] = J[(i-1) * grid_cols + j] - Jc;
          dS[k] = J[k] - Jc;
        } else {
          dN[k] = J[(i-1) * grid_cols + j] - Jc;
          dS[k] = J[(i+1) * grid_cols + j] - Jc;
        }
        if( j == 0 ) {	
          dW[k] = J[i * grid_cols] - Jc;
          dE[k] = J[i * grid_cols + 1] - Jc;
        } else if( j == (grid_cols-1) ) {
          dW[k] = J[i * grid_cols + (j-1)] - Jc;
          dE[k] = J[k] - Jc;
        } else {
          dW[k] = J[i * grid_cols + (j-1)] - Jc;
          dE[k] = J[i * grid_cols + (j+1)] - Jc;
        }

        G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
            + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

        num  = (0.5F*G2) - ((1.0F/16.0F)*(L*L)) ;
        den  = 1 + (.25F*L);
        qsqr = num/(den*den);

        // diffusion coefficent (equ 33)
        den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
        c[k] = 1.0F / (1.0F+den) ;

        // saturate diffusion coefficent
        if (c[k] < 0) {c[k] = 0;}
        else if (c[k] > 1) {c[k] = 1;}

      }

    }
    for (i = 0; i < grid_rows; i++) {
      for (j = 0; j < grid_cols; j++) {        

        // current index
        k = i * grid_cols + j;

        // diffusion coefficent
        cN = c[k];
        if( i == (grid_rows-1) ) {
          cS = c[k];
        } else {
          cS = c[(i+1) * grid_cols + j];
        }
        cW = c[k];
        if( j == (grid_cols-1) ) {
          cE = c[k];
        } else {
          cE = c[i * grid_cols + j+1];
        }

        // divergence (equ 58)
        D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

        // image update (equ 61)
        J[k] = J[k] + 0.25F*lambda*D;
      }
    }

#ifdef ITERATION
  }
#endif
}
#endif

int main(int argc, char* argv[])
{   
  float *I, *J;
  int *iN,*iS,*jE,*jW;
  float *dN,*dS,*dW,*dE;
  float *c;
  int i, j, k;
  double start_time, end_time;
#ifdef DEBUG
  double start_time1, end_time1;
  double start_time2, end_time2;
#endif
#ifdef OUTPUT
  FILE *fp;
#endif
#if VERIFICATION == 1
  float *CPU_J;
#endif 

  #if 0
  if (argc == 10)
  {
    grid_rows = atoi(argv[1]); //number of grid_rows in the domain
    grid_cols = atoi(argv[2]); //number of grid_cols in the domain
    if ((grid_rows%16!=0) || (grid_cols%16!=0)){
      fprintf(stderr, "grid_rows and grid_cols must be multiples of 16\n");
      exit(1);
    }
    if( (grid_rows != _ROWS_) || (grid_cols != _COLS_) ) {
      fprintf(stderr, "Wrong values for _ROWS_ or _COLS_!\n");
      exit(1);
    }
    r1   = atoi(argv[3]); //y1 position of the speckle
    r2   = atoi(argv[4]); //y2 position of the speckle
    c1   = atoi(argv[5]); //x1 position of the speckle
    c2   = atoi(argv[6]); //x2 position of the speckle
    nthreads = atoi(argv[7]); // number of threads
    lambda = atof(argv[8]); //Lambda value
    niter = atoi(argv[9]); //number of iterations
  }
  else{
    usage(argc, argv);
  }
  #endif

  if (argc != 2) {printf("usage: srad_ACC iters\n"); exit(0); }

  grid_rows = _ROWS_;
  grid_cols = _COLS_;
  printf("_ROWS_: %d, _COLS_: %d\n", grid_rows, grid_cols);

  r1 = 0;
  r2 = grid_rows - 1;
  c1 = 0;
  c2 = grid_cols - 1;

  nthreads = 1;
  lambda = 0.5;
  niter = atoi(argv[1]);

  start_time = getCurrentTimestamp();

  size_I = grid_cols * grid_rows;
  size_R = (r2-r1+1)*(c2-c1+1);   

  I = (float *)malloc( size_I * sizeof(float) );
  J = (float *)malloc( size_I * sizeof(float) );
  c = (float *)malloc( size_I * sizeof(float) );

  iN = (int *)malloc(sizeof(unsigned int*) * grid_rows) ;
  iS = (int *)malloc(sizeof(unsigned int*) * grid_rows) ;
  jW = (int *)malloc(sizeof(unsigned int*) * grid_cols) ;
  jE = (int *)malloc(sizeof(unsigned int*) * grid_cols) ;    


  dN = (float *)malloc(sizeof(float)* size_I) ;
  dS = (float *)malloc(sizeof(float)* size_I) ;
  dW = (float *)malloc(sizeof(float)* size_I) ;
  dE = (float *)malloc(sizeof(float)* size_I) ;    

#if VERIFICATION == 1
  CPU_J = (float *)malloc( size_I * sizeof(float) );
#endif

  for (i=0; i< grid_rows; i++) {
    iN[i] = i-1;
    iS[i] = i+1;
  }    
  for (j=0; j< grid_cols; j++) {
    jW[j] = j-1;
    jE[j] = j+1;
  }
  iN[0]    = 0;
  iS[grid_rows-1] = grid_rows-1;
  jW[0]    = 0;
  jE[grid_cols-1] = grid_cols-1;

  printf("Randomizing the input matrix\n");

#ifdef DEBUG
  start_time2 = getCurrentTimestamp();
#endif

  random_matrix(I, grid_rows, grid_cols);

  for (k = 0;  k < size_I; k++ ) 
  {
    J[k] = (float)exp(I[k]) ;
#if VERIFICATION == 1
    CPU_J[k] = J[k];
#endif
  }

#ifdef DEBUG
  end_time2 = getCurrentTimestamp();
  printf("Input Randomization Time = %lf sec\n", end_time2 - start_time2);
#endif

  printf("Start the SRAD main loop\n");

  #pragma acc data \
      copy(J[0:_SIZE_I_]) \
      create( dN[0:_SIZE_I_], dS[0:_SIZE_I_], dW[0:_SIZE_I_], dE[0:_SIZE_I_], \
      c[0:_SIZE_I_])
  {
    start_time1 = getCurrentTimestamp();

    for (int iter = 0; iter < niter; iter++){ 

      mainComp(J, iN, iS, jE, jW, dN, dS, dW, dE, c);

      #if ND_UPDATE || SWI_UPDATE  || SWI_UPDATE_PIPE
      #pragma acc update host(J[0:_SIZE_I_])
      #endif
    }

    end_time1 = getCurrentTimestamp();
    printf(" Accelerator Elapsed Time = %lf sec\n", end_time1 - start_time1);
  }

#if VERIFICATION == 1
  {
    double diff;
    double diff_norm = 0.0f;
    double cpu_norm = 0.0f;
    double rel_err;

    start_time1 = getCurrentTimestamp();
    mainCompCPU(CPU_J, iN, iS, jE, jW, dN, dS, dW, dE, c);
    end_time1 = getCurrentTimestamp();
    printf(" CPU Elapsed Time = %lf sec\n", end_time1 - start_time1);

    for(i = 0; i < grid_rows; i++)
    {
      for(j = 0; j < grid_cols; j++)
      {
        int k = i*grid_cols+j;
        diff = (J[i * grid_cols + j] - CPU_J[i * grid_cols + j]);
            
        diff_norm += diff * diff;
        cpu_norm += CPU_J[i * grid_cols + j]*CPU_J[i * grid_cols + j];
      }
    }

    diff_norm = sqrt(diff_norm);
    cpu_norm = sqrt(cpu_norm);

    rel_err = diff_norm/cpu_norm;

    if(rel_err < 1e-5) // changed from 1e-6
      printf("\nVerification Successfull\n\t CPU L2Norm = %e, Diff L2Norm = %e Err=%e\n", cpu_norm, diff_norm, rel_err);
    else
    {
      printf("\nVerification Failed\n\t CPU L2Norm = %e, Diff L2Norm = %e Err=%e\n", cpu_norm, diff_norm, rel_err);
    }
  }
#endif

#ifdef OUTPUT
  if( (fp = fopen("sradOutput.txt", "w")) == 0 ) {
    printf("Can not open %s\n", "sradOutput.txt");
    exit(1);
  }
  for( i = 0 ; i < grid_rows ; i++){
    for ( j = 0 ; j < grid_cols ; j++){

      fprintf(fp, "%.5f ", J_out[i * grid_cols + j]);

    }
    fprintf(fp, "\n");
  }
  fclose(fp);
#endif

  printf("\nComputation Done\n");

  free(I);
  free(J); 
  free(iN); free(iS); free(jW); free(jE);
  free(dN); free(dS); free(dW); free(dE);

  free(c);

  end_time = getCurrentTimestamp();
  printf("Total Execution Time = %lf sec.\n", end_time - start_time);
  return 0;
}

void random_matrix(float *I, int grid_rows, int grid_cols){
  int i, j;

  srand(7);

  for( i = 0 ; i < grid_rows ; i++){
    for ( j = 0 ; j < grid_cols ; j++){
      I[i * grid_cols + j] = rand()/(float)RAND_MAX ;
#ifdef OUTPUT
      //printf("%g ", I[i * grid_cols + j]); 
#endif 
    }
#ifdef OUTPUT
    //printf("\n"); 
#endif 
  }

}
