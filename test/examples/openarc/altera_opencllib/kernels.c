#include <stdio.h>
#include <math.h>
#include <sys/time.h>

double my_timer ()
{
    struct timeval time;
    gettimeofday (&time, 0); 
    return time.tv_sec + time.tv_usec / 1000000.0;
}

unsigned int my_byteswap(unsigned int x);

#pragma acc routine seq bind(my_byteswap)
unsigned int byteswap(unsigned int x) {
  unsigned int result = x << 16 | x >> 16;
  return result;
}

// Runtime constants
static unsigned int BUFSIZE = 4*1024*1024;
static int N = 1;

void kernels(unsigned int *data_in, unsigned int *output_lib, unsigned int *output_builtin) {
  int i, k, m;
  int NUM_ITERATIONS = 4;
  
  // Configure work set over which the kernel will execute
  size_t gSize = BUFSIZE/N;
  size_t lSize = 1; 

  #pragma acc data copyin(data_in[0:BUFSIZE]) copyout(output_lib[0:BUFSIZE], output_builtin[0:BUFSIZE])
  {
    // Launch the kernels
    printf("Enqueueing both library and builtin in kernels %d times with global size %d\n", NUM_ITERATIONS, (int) gSize);
  
    // Launch kernel_lib
    double lib_start = my_timer();
    for (m = 0; m < NUM_ITERATIONS; m++) {
      #pragma acc kernels loop gang(BUFSIZE/N) present(data_in[0:BUFSIZE]) present(output_lib[0:BUFSIZE]) copyin(N)
      for (i=0; i<BUFSIZE/N; i++) {
        #pragma acc loop worker(1)
        for (k =0; k < N; k++) {
          unsigned int x = data_in[i*N + k]; 
          output_lib[i*N + k] = byteswap(x);
        }
      }
    }
    double lib_stop = my_timer();
    printf ("Kernel computation using library function took %g seconds\n", lib_stop - lib_start);

    // Launch kernel_builtin
    double builtin_start = my_timer();
    for (int m = 0; m < NUM_ITERATIONS; m++) {
      #pragma acc kernels loop gang(BUFSIZE/N) present(data_in[0:BUFSIZE]) present(output_builtin[0:BUFSIZE]) copyin(N)
      for (i=0; i<BUFSIZE/N; i++) {
        #pragma acc loop worker(1)
        for (k =0; k < N; k++) {
          unsigned int x = data_in[i*N + k]; 
          output_builtin[i*N + k] = x << 16 | x >> 16;
        }
      }
    }
    double builtin_stop = my_timer();
    printf ("Kernel computation using builtin function took %g seconds\n", builtin_stop - builtin_start);
    double time_ratio = (lib_stop - lib_start) / (builtin_stop - builtin_start);
  }
}
