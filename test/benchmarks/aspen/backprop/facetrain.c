#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();
extern void load();
extern void bpnn_train_kernel(float *eo, float *eh);

int layer_size = 0;

void backprop_face()
{
  //int i;
  float out_err, hid_err;
#if SAVE_OUTPUT == 1
  char filename[32];
#endif
  bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load();
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_kernel(&out_err, &hid_err);
#if SAVE_OUTPUT == 1
  sprintf(filename, "bp_out.%d", layer_size);
  bpnn_save2(filename);
#endif
  bpnn_free();
  printf("Training done\n");
}

void setup(argc, argv)
int argc;
char *argv[];
{
  int seed;
  if(argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }

  layer_size = atoi(argv[1]);
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  //exit(0);
}
