#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#define BIGRND 0x7fffffff



#ifndef VERIFICATION
#define VERIFICATION 0
#endif


#define ETA 0.3F       //eta value
#define MOMENTUM 0.3F  //momentum value
#define NUM_THREAD 4 //OpenMP threads

//I_SIZE should be the same as commandline input (layer_size in facetrain.c)
#ifndef I_SIZE
#define I_SIZE  655361
#endif
#define H_SIZE  (16 + 1)
#define O_SIZE  (1 + 1) 
#ifdef _OPENARC_
#if I_SIZE == 655361
	#pragma openarc #define I_SIZE 65537
#elif I_SIZE == 655361
	#pragma openarc #define I_SIZE 655361
#elif I_SIZE == 6553601
	#pragma openarc #define I_SIZE 6553601
#endif
#pragma openarc #define H_SIZE  (16 + 1)
#pragma openarc #define O_SIZE  (1 + 1) 
#endif


extern  int input_n;                  /* number of input units */
extern  int hidden_n;                 /* number of hidden units */
extern  int output_n;                 /* number of output units */

extern  float *input_units;          /* the input units */
extern  float *hidden_units;         /* the hidden units */
//[DEBUG] below data are not used in the OpenACC data clauses and allocated 
//indirectedly through a function call; should be handled manually.
#pragma aspen declare data(output_units:traits(Array(2)))
extern  float *output_units;         /* the output units */

extern  float *hidden_delta;         /* storage for hidden unit error */
extern  float *output_delta;         /* storage for output unit error */

//[DEBUG] below data are not used in the OpenACC data clauses and allocated 
//indirectedly through a function call; should be handled manually.
#pragma aspen declare data(target:traits(Array(2)))
extern  float *target;               /* storage for target vector */

extern  float (*input_weights)[I_SIZE];       /* weights from input to hidden layer */
extern  float (*hidden_weights)[H_SIZE];      /* weights from hidden to output layer */

                                /*** The next two are for momentum ***/
extern  float (*input_prev_weights)[I_SIZE];  /* previous change on input to hidden wgt */
extern  float (*hidden_prev_weights)[H_SIZE]; /* previous change on hidden to output wgt */

//Verification related variables
extern float *hidden_units_CPU;         /* the hidden units */
extern float *output_units_CPU;         /* the output units */

extern float *hidden_delta_CPU;         
extern float *output_delta_CPU;        

extern float (*input_weights_CPU)[I_SIZE];       /* weights from input to hidden layer */
extern float (*hidden_weights_CPU)[H_SIZE];      /* weights from hidden to output layer */
	                              /*** The next two are for momentum ***/
extern float (*input_prev_weights_CPU)[I_SIZE];  /* previous change on input to hidden wgt */
extern float (*hidden_prev_weights_CPU)[H_SIZE]; /* previous change on hidden to output wgt */




/*** User-level functions ***/

void bpnn_initialize(int seed);

void bpnn_create(int n_in, int n_hidden, int n_out);
void bpnn_free();

void bpnn_train();
void bpnn_feedforward();

void bpnn_save(char *filename);
void bpnn_save2(char *filename);
void bpnn_read();


#endif
