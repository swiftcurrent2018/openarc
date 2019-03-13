#include "defines.h"
#include "stdio.h"


#if ND_RANGE

void sobel_ACC(unsigned int * frame_in, 
               unsigned int * frame_out, 
               int iterations, int threshold)
{
  int Gx[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
  int Gy[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};

  #pragma acc parallel copyin(threshold) \
           present(frame_in[0:ROWS*COLS], frame_out[0:ROWS*COLS]) 
  #pragma openarc opencl num_simd_work_items(16)
  //#pragma openarc opencl num_compute_units(16)
  {
    #pragma acc loop gang worker
    for (int count = 0; count < ROWS*COLS; count++) {
      int x_dir = 0;
      int y_dir = 0;

      #pragma unroll
      for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {

          unsigned int pixel;
          if (count - (i * COLS) - j >= 0) {
            pixel = frame_in[count - (i * COLS)  - j];
          } else {
            pixel = 0;
          }

          unsigned int b = pixel & 0xff;
          unsigned int g = (pixel >> 8) & 0xff;
          unsigned int r = (pixel >> 16) & 0xff;

          // RGB -> Luma conversion approximation
          // Avoiding floating point math operators greatly reduces
          // resource usage.
          unsigned int luma = r * 66 + g * 129 + b * 25;
          luma = (luma + 128) >> 8;
          luma += 16;

          x_dir += luma * Gx[i][j];
          y_dir += luma * Gy[i][j];
        }
      }

      int temp = abs(x_dir) + abs(y_dir);
      unsigned int clamped;
      if (temp > threshold) {
        clamped = 0xffffff;
      } else {
        clamped = 0;
      }

      frame_out[count] = clamped;
    }
  }
}
#endif
