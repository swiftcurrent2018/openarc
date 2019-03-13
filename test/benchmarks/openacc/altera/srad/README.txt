To build:
  ./O2GBuild.script;
  make;
  
To execute:
  cd ./bin;
  ./srad_ACC 100;

To modify input size:
  adjust O2GBuild.Script
  adjust inc/srad.h

Kernel Versions:
  nd_update.c - 
      ND-Range Kernels, with reduction done on the host and results transferred to 
      the device via the OpenACC update directive. 
  nd_reduce.c - 
      ND-Range Kernels, with the reduction done on the device via the OpenACC reduction clause 
      (invoking the multi-threaded tree-based reduction transformation). 
  swi_reduce.c - 
      Single Work Item Kernels, with reduction done on the device using the OpenACC reduction clause
      (invoking the FPGA-specific shift-register-based reduction clause). The OpenACC collapse(2) clause 
      is applied to the main computation loops (invoking the FPGA-specific collapse optimization).
  window_update.c - 
      Single Work Item Kernels, with reduction done on the host and results transferred to
      the device via the OpenACC update directive. The window directive is also applied to the 
      two main computation loops, and loop unrolling can be optionally applied. 
  window_update_pipe.c - 
      Single Work Item Kernels, with reduction done on the host and results transferred to
      the device via the OpenACC update directive. The window directive is also applied to the 
      two main computation loops, and loop unrolling can be optionally applied. The pipe clause is also
      applied to transfer data between the two main loops.
