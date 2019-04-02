// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>

// This is the minimum alignment requirement to ensure DMA can be used.
const unsigned AOCL_ALIGNMENT = 64; 

void kernels(unsigned int *data_in, unsigned int *output_lib, unsigned int *output_builtin);

// Runtime constants
static unsigned int BUFSIZE = 4*1024*1024;

// Entry point.
int main(int argc, char** argv) {

   bool emulator_run = false;
   if ( (argc > 1) && (strcmp (argv[1], "-emulator") == 0) ) {
      emulator_run = true;
   }
   if (emulator_run) {
      BUFSIZE /= 1024;
      printf ("Shortened test size to %d for emulator run.\n", BUFSIZE);
   }

  // Generate random input data
  void *p;
  unsigned int *data_in;
  posix_memalign(&p, AOCL_ALIGNMENT, BUFSIZE*sizeof(unsigned int));
  data_in = (unsigned int *)p;
  printf("Generate random data for conversion...\n");
  srand( 1 );   
  for ( int i = 0; i < BUFSIZE ; i++ ) {
    data_in[i] = rand();
  }
  // Allocate output data
  unsigned int *output_lib;
  posix_memalign(&p, AOCL_ALIGNMENT, BUFSIZE*sizeof(unsigned int));
  output_lib = (unsigned int *)p;
  unsigned int *output_builtin;
  posix_memalign(&p, AOCL_ALIGNMENT, BUFSIZE*sizeof(unsigned int));
  output_builtin = (unsigned int *)p;

  kernels(data_in, output_lib, output_builtin);

  printf("Checking results...\n");

  int num_printed, num_errs = 0;
  for (int i = 0; i < BUFSIZE; i++ )
  {
    if (output_lib[i] != output_builtin[i] ||  
        output_lib[i] == 0 && output_builtin[i] == 0 && data_in[i] != 0 ) { 
      num_errs++;
      if (num_printed < 10) {
        printf ("ERROR at i=%d, library = %08x, builtin = %08x, datain = %08x\n", i, output_lib[i], output_builtin[i], data_in[i]);
        num_printed++;
      }   
    }   
  }


  if ( num_errs > 0 ) { 
    printf("FAILED with %i errors.\n", num_errs);
  } else {
    printf("PASSED\n");
  }

  free(data_in);
  free(output_lib);
  free(output_builtin);

  return 0;
}
