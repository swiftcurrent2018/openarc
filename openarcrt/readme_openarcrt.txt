-------------------------------------------------------------------------------
RELEASE
-------------------------------------------------------------------------------
OpenARC Runtime V0.4 (June 01, 2015)

OpenARC Runtime implements APIs used by the output program translated
by OpenARC.


-------------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------------
* GCC
* NVCC to run on the CUDA target
* GCC or other OpenCL compiler to run on the OpenCL target

 
-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
* Build
  - Go to the parent directory, which is a main entry directory of OpenARC.
  - Copy "make.header.sample" to "make.header", and modify environment 
  variables in the "make.header" file according to user's environment.
  - Go back to the openarcrt directory 
  - Run "batchmake.bash"
    $ ./batchmake.bash
	//which will create either CUDA or OpenCL libraries of the OpenARC runtime, 
	//depending on the target architecture. 
	//For example on a CUDA device:
	//libopenaccrt_cuda.a -- normal mode OpenARC CUDA library
	//libopenaccrt_cudapf.a -- profile mode OpenARC CUDA library
	//libopenaccrtomp_cuda.a -- normal mode OpenARC CUDA library with OpenMP support
	//libopenaccrtomp_cudapf.a -- profile mode OpenARC CUDA library with OpenMP support


-------------------------------------------------------------------------------
FEATURES/UPDATES
-------------------------------------------------------------------------------
* New features

* Updates

* Bug fixes and improvements


-------------------------------------------------------------------------------
LIMITATIONS
-------------------------------------------------------------------------------


The OpenARC Team

URL: http://ft.ornl.gov/research/openarc
EMAIL: lees2@ornl.gov
