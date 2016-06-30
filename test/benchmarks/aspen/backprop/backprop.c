/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
//#define _OPENMP

#define ABS(x)          (((x) > 0.0F) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

#define ROWS_TO_PRINT	5
//#define ROWS_TO_PRINT	6553600

/*** Return random number between 0.0 and 1.0 ***/
float drnd()
{
  return ((float) rand() / (float) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
float dpn1()
{
  return ((drnd() * 2.0F) - 1.0F);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

float squash(x)
float x;
{
  //float m;
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
  return (1.0F / (1.0F + expf(-x)));
}


/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(n)
int n;
{
  float *newA;

  newA = (float *) malloc ((unsigned) (n * sizeof (float)));
  if (newA == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (float *)0;
  }
  return (newA);
}


/*** Allocate 2d array of floats ***/

float *alloc_2d_dbl(m, n)
int m, n;
{
  float *newA;

  newA = (float *) malloc ((unsigned) (m * n * sizeof (float)));
  if (newA == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (float *)0;
  }

  return (newA);
}


void bpnn_randomize_weights1(w, m, n)
float w[H_SIZE][I_SIZE];
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[j][i] = (float) rand()/RAND_MAX;
    //  w[i][j] = dpn1();
    }
  }
}

void bpnn_randomize_weights2(w, m, n)
float w[O_SIZE][H_SIZE];
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[j][i] = (float) rand()/RAND_MAX;
    //  w[i][j] = dpn1();
    }
  }
}

void bpnn_randomize_row(w, m)
float *w;
int m;
{
	int i;
	for (i = 0; i <= m; i++) {
     //w[i] = (float) rand()/RAND_MAX;
	 w[i] = 0.1;
    }
}


void bpnn_zero_weights1(w, m, n)
float w[H_SIZE][I_SIZE];
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0F;
    }
  }
}

void bpnn_zero_weights2(w, m, n)
float w[O_SIZE][H_SIZE];
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0F;
    }
  }
}


void bpnn_initialize(int seed)
{
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}


void bpnn_internal_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{
  input_n = n_in;
  hidden_n = n_hidden;
  output_n = n_out;
  input_units = alloc_1d_dbl(n_in + 1);
  hidden_units = alloc_1d_dbl(n_hidden + 1);
  output_units = alloc_1d_dbl(n_out + 1);

  hidden_delta = alloc_1d_dbl(n_hidden + 1);
  output_delta = alloc_1d_dbl(n_out + 1);
  target = alloc_1d_dbl(n_out + 1);

  input_weights = (float (*)[I_SIZE])alloc_2d_dbl(n_in + 1, n_hidden + 1);
  hidden_weights = (float (*)[H_SIZE])alloc_2d_dbl(n_hidden + 1, n_out + 1);

  input_prev_weights = (float (*)[I_SIZE])alloc_2d_dbl(n_in + 1, n_hidden + 1);
  hidden_prev_weights = (float (*)[H_SIZE])alloc_2d_dbl(n_hidden + 1, n_out + 1);

	if(VERIFICATION) {
		//input_units_CPU = alloc_1d_dbl(n_in + 1);
		hidden_units_CPU = alloc_1d_dbl(n_hidden + 1);
		output_units_CPU = alloc_1d_dbl(n_out + 1);

		hidden_delta_CPU = alloc_1d_dbl(n_hidden + 1);
		output_delta_CPU = alloc_1d_dbl(n_out + 1);


		input_weights_CPU = (float (*)[I_SIZE])alloc_2d_dbl(n_in + 1, n_hidden + 1);
		hidden_weights_CPU = (float (*)[H_SIZE])alloc_2d_dbl(n_hidden + 1, n_out + 1);

		input_prev_weights_CPU = (float (*)[I_SIZE])alloc_2d_dbl(n_in + 1, n_hidden + 1);
		hidden_prev_weights_CPU = (float (*)[H_SIZE])alloc_2d_dbl(n_hidden + 1, n_out + 1);
	  
	}
}


void bpnn_free()
{
  int n1, n2;

  n1 = input_n;
  n2 = hidden_n;

  free((char *) input_units);
  free((char *) hidden_units);
  free((char *) output_units);

  free((char *) hidden_delta);
  free((char *) output_delta);
  free((char *) target);

  free((char *) input_weights);
  free((char *) input_prev_weights);

  free((char *) hidden_weights);
  free((char *) hidden_prev_weights);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

void bpnn_create(n_in, n_hidden, n_out)
int n_in, n_hidden, n_out;
{

  bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights1(input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights1(input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights2(hidden_weights,  n_hidden, n_out);
  bpnn_zero_weights1(input_prev_weights, n_hidden, n_in);
  bpnn_zero_weights2(hidden_prev_weights, n_out, n_hidden);
  bpnn_randomize_row(target, n_out);
}


void bpnn_layerforward1(float l1[I_SIZE], float l2[H_SIZE], float conn[H_SIZE][I_SIZE], int n1, int n2)
{
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0F;
#if defined(_OPENMP)
  omp_set_num_threads(NUM_THREAD);
	//printf("OpenMP in bpnn_layerforward1()\n");
#endif 
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0F;
    for (k = 0; k <= n1; k++) {	
      sum += conn[j][k] * l1[k]; 
    }
    //l2[j] = squash(sum);
  	l2[j] = (1.0F / (1.0F + expf(-sum)));
  }
}

void bpnn_layerforward2(float l1[H_SIZE], float l2[O_SIZE], float conn[O_SIZE][H_SIZE], int n1, int n2)
{
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0F;
#if defined(_OPENMP)
  omp_set_num_threads(NUM_THREAD);
	//printf("OpenMP in bpnn_layerforward2()\n");
#endif 
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0F;
    for (k = 0; k <= n1; k++) {	
      sum += conn[j][k] * l1[k]; 
    }
    //l2[j] = squash(sum);
  	l2[j] = (1.0F / (1.0F + expf(-sum)));
  }
}

//extern "C"
void bpnn_output_error(delta, target, output, nj, err)  
float *delta, *target, *output, *err;
int nj;
{
//[DEBUG] below will not be necessary if interprocedural constant
//propagation works.
#pragma aspen declare param(nj:O_SIZE-1)
  int j;
  float o, t, errsum;
  errsum = 0.0F;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0F - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}


void bpnn_hidden_error(delta_h,   
					   nh, 
					   delta_o, 
					   no, 
					   who, 
					   hidden, 
					   err)
float *delta_h, *delta_o, *hidden, who[O_SIZE][H_SIZE], *err;
int nh, no;
{
//[DEBUG] below will not be necessary if interprocedural constant
//propagation works.
#pragma aspen declare param(nh:H_SIZE-1, no:O_SIZE-1)
  int j, k;
  float h, sum, errsum;

  errsum = 0.0F;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0F;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[k][j];
    }
    delta_h[j] = h * (1.0F - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}


void bpnn_adjust_weights1(float delta[O_SIZE], int ndelta, float ly[H_SIZE], 
int nly, float w[O_SIZE][H_SIZE], float oldw[O_SIZE][H_SIZE])
{
  float new_dw;
  int k, j;
  ly[0] = 1.0F;
  //eta = 0.3;
  //momentum = 0.3;

#if defined(_OPENMP)
  omp_set_num_threads(NUM_THREAD);
	//printf("OpenMP in bpnn_adjust_weights1()\n");
#endif 
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[j][k]));
	  w[j][k] += new_dw;
	  oldw[j][k] = new_dw;
    }
  }
}

void bpnn_adjust_weights2(float delta[H_SIZE], int ndelta, float ly[I_SIZE], 
int nly, float w[H_SIZE][I_SIZE], float oldw[H_SIZE][I_SIZE])
{
  float new_dw;
  int k, j;
  ly[0] = 1.0F;
  //eta = 0.3;
  //momentum = 0.3;

#if defined(_OPENMP)
  omp_set_num_threads(NUM_THREAD);
	//printf("OpenMP in bpnn_adjust_weights2()\n");
#endif 
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[j][k]));
	  w[j][k] += new_dw;
	  oldw[j][k] = new_dw;
    }
  }
}


void bpnn_feedforward()
{
  int in, hid, out;

  in = input_n;
  hid = hidden_n;
  out = output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward1(input_units, hidden_units,
      input_weights, in, hid);
  bpnn_layerforward2(hidden_units, output_units,
      hidden_weights, hid, out);

}


void bpnn_train(eo, eh)
float *eo, *eh;
{
  int in, hid, out;
  float out_err, hid_err;

  in = input_n;
  hid = hidden_n;
  out = output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward1(input_units, hidden_units,
      input_weights, in, hid);
  bpnn_layerforward2(hidden_units, output_units,
      hidden_weights, hid, out);

  /*** Compute error on output and hidden units. ***/
  bpnn_output_error(output_delta, target, output_units,
      out, &out_err);
  bpnn_hidden_error(hidden_delta, hid, output_delta, out,
      hidden_weights, hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights1(output_delta, out, hidden_units, hid,
      hidden_weights, hidden_prev_weights);
  bpnn_adjust_weights2(hidden_delta, hid, input_units, in,
      input_weights, input_prev_weights);

}




void bpnn_save(filename)
char *filename;
{
  int n1, n2, n3, i, j, memcnt;
  float dvalue, (*w_i)[I_SIZE], (*w_h)[H_SIZE];
  char *mem;
  ///add//
  FILE *pFile;
  pFile = fopen( filename, "w+" );
  ///////
  /*
  if ((fd = creat(filename, 0644)) == -1) {
    printf("BPNN_SAVE: Cannot create '%s'\n", filename);
    return;
  }
  */

  n1 = input_n;  n2 = hidden_n;  n3 = output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  //fflush(stdout);

  //write(fd, (char *) &n1, sizeof(int));
  //write(fd, (char *) &n2, sizeof(int));
  //write(fd, (char *) &n3, sizeof(int));

  fwrite( (char *) &n1 , sizeof(char), sizeof(char), pFile);
  fwrite( (char *) &n2 , sizeof(char), sizeof(char), pFile);
  fwrite( (char *) &n3 , sizeof(char), sizeof(char), pFile);

  

  memcnt = 0;
  w_i = input_weights;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w_i[j][i];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  fwrite( mem , (unsigned)(sizeof(float)), (unsigned) ((n1+1) * (n2+1) * sizeof(float)) , pFile);
  free(mem);

  memcnt = 0;
  w_h = hidden_weights;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w_h[j][i];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  //write(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  fwrite( mem , sizeof(float), (unsigned) ((n2+1) * (n3+1) * sizeof(float)) , pFile);
  free(mem);

  fclose(pFile);
  return;
}

void bpnn_save2(filename)
char *filename;
{
  int n1, n2, n3, i, j;
  float (*w_i)[I_SIZE], (*w_h)[H_SIZE];
  ///add//
  FILE *pFile; 
  int UB1, UB2;
  pFile = fopen( filename, "w+" ); 
  /////// 

  n1 = input_n;  n2 = hidden_n;  n3 = output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  //fflush(stdout);

  fprintf(pFile, "%d %d %d\n", n1, n2, n3);
  
  if( n1 < ROWS_TO_PRINT ) { UB1 = n1; }
  else { UB1 = ROWS_TO_PRINT; }

  if( n2 < ROWS_TO_PRINT ) { UB2 = n2; }
  else { UB2 = ROWS_TO_PRINT; }

  w_i = input_weights;
  for (i = 0; i < UB1; i++) {
    for (j = 0; j <= n2; j++) {
      fprintf(pFile, "%lf ", w_i[j][i]);
    }
    fprintf(pFile, "\n");
  }

  w_h = hidden_weights;
  for (i = 0; i < UB2; i++) {
    for (j = 0; j <= n3; j++) {
      fprintf(pFile, "%lf ", w_h[j][i]);
    }
    fprintf(pFile, "\n");
  }


  fclose(pFile);
  return;
}


void bpnn_read(filename)
char *filename;
{
  char *mem;
  int fd, n1, n2, n3, i, j, memcnt;

  if ((fd = open(filename, 0, 0644)) == -1) {
    return;
  }

  printf("Reading '%s'\n", filename);  //fflush(stdout);

  read(fd, (char *) &n1, sizeof(int));
  read(fd, (char *) &n2, sizeof(int));
  read(fd, (char *) &n3, sizeof(int));
  bpnn_internal_create(n1, n2, n3);

  printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  printf("Reading input weights...");  //fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
  read(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(input_weights[j][i]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);

  printf("Done\nReading hidden weights...");  //fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
  read(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(hidden_weights[j][i]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);
  close(fd);

  printf("Done\n");  //fflush(stdout);

  bpnn_zero_weights1(input_prev_weights, n2, n1);
  bpnn_zero_weights2(hidden_prev_weights, n3, n2);
}
