
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

extern int layer_size;

void load( )
{
  float *units;
  //int nc, imgsize, j;
	int nr, i, k;

  nr = layer_size;
  
  //imgsize = nr * nc;
  units = input_units;

  k = 1;
  for (i = 0; i < nr; i++) {
	  units[k] = (float) rand()/RAND_MAX ;
	  k++;
  }
}
