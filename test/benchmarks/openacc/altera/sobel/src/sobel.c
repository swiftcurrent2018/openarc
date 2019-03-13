// Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "parse_ppm.h"
#include "defines.h"

bool useFilter = false;
unsigned int thresh = 128;

unsigned int *input = NULL;
unsigned int *acc_output = NULL;
unsigned int *cpu_output = NULL;

char const *imageFilename;
bool testMode = false;
unsigned testThresholds[] = {32, 96, 128, 192, 225};
unsigned testFrameIndex = 0;

void sobel_ACC(unsigned int * frame_in, unsigned int * frame_out, int iterations, int threshold);
void sobel_CPU(unsigned int * frame_in, unsigned int * frame_out, int iterations, int threshold);

void teardown(int exit_status);
void dumpFrame(unsigned frameIndex, unsigned *frameData);
void filter();
double getCurrentTimestamp();

int main(int argc, char **argv)
{
  imageFilename = "../butterflies.ppm";

  useFilter = true;
  testMode = true;

  double time_0, time_s;
  time_0 = getCurrentTimestamp();

  posix_memalign((void  **)(&input), AOCL_ALIGNMENT, sizeof(unsigned int) * ROWS * COLS);
  posix_memalign((void  **)(&acc_output), AOCL_ALIGNMENT, sizeof(unsigned int) * ROWS * COLS);
  posix_memalign((void  **)(&cpu_output), AOCL_ALIGNMENT, sizeof(unsigned int) * ROWS * COLS);

  // Read the image
  if (!parse_ppm(imageFilename, COLS, ROWS, (unsigned char *)input)) {
    fprintf(stderr, "Error: could not load %s\n", argv[1]);
    teardown(1);
  }

  filter();


  teardown(0);

  time_s = (getCurrentTimestamp() - time_0);
  printf("Total %d time:\t%.6f ms\n", (time_s) / 1000 );
}

/// Helper Functions
void filter()
{
  size_t sobelSize = 1;
  double time_0, time_s;

  int pixels = COLS * ROWS;

  #pragma acc data present_or_copyin(input[0:ROWS*COLS]) copyout(acc_output[0:ROWS*COLS])
  {
    time_0 = getCurrentTimestamp();

    sobel_ACC(input, acc_output, pixels, thresh);

    time_s = (getCurrentTimestamp() - time_0);
    printf("Accelerator %d time:\t%.6fms\n", thresh, time_s * 1000);
  }

  // CPU 
  {
    time_0 = getCurrentTimestamp();

    sobel_CPU(input, cpu_output, pixels, thresh);

    time_s = (getCurrentTimestamp() - time_0);
    printf("CPU %d time:\t%.6fms\n", thresh, time_s * 1000);
  }


  // Verification
  double diff_norm = 0;
  double cpu_norm = 0;

  for (int i = 0; i < ROWS*COLS; ++i) {
    int diff = acc_output[i] - cpu_output[i];
    diff_norm += diff * diff;
    cpu_norm += cpu_output[i] * cpu_output[i];
  }

  diff_norm = sqrt(diff_norm);
  cpu_norm = sqrt(cpu_norm);

  double rel_err = diff_norm / cpu_norm;

    
  if(rel_err < 1e-6)
    printf("\nVerification Successfull\n\t CPU L2Norm = %e, Diff L2Norm = %e Err=%e\n", cpu_norm, diff_norm, rel_err);
  else
    printf("\nVerification Failed\n\t CPU L2Norm = %e, Diff L2Norm = %e Err=%e\n", cpu_norm, diff_norm, rel_err);


  //dumpFrame(testFrameIndex, output);

}

// Dump frame data in PPM format.
void dumpFrame(unsigned frameIndex, unsigned *frameData) {
  unsigned y; 
  unsigned x; 
 
  char fname[256];
  sprintf(fname, "frame%d.ppm", frameIndex);

  printf("Dumping %s\n", fname);

  FILE *f = fopen(fname, "wb");
  fprintf(f, "P6\n%d %d\n%d\n", COLS, ROWS, 255);
  for(y = 0; y < ROWS; ++y) {
    for(x = 0; x < COLS; ++x) {
      // This assumes byte-order is little-endian.
      unsigned int pixel = frameData[y * COLS + x];
      fwrite(&pixel, 1, 3, f);
    }
  }
  fclose(f);
}

void teardown(int exit_status)
{
  if (input) free(input);
  if (acc_output) free(acc_output);
  if (cpu_output) free(cpu_output);

  exit(exit_status);
}

// High-resolution timer.
double getCurrentTimestamp() {
  struct timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return ( (double) a.tv_nsec * 1.0e-9) + (double) a.tv_sec;
}

