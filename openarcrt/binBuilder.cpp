#include <stdio.h>
#include <stdlib.h>
#include <string>

#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
#include <cuda.h>
#include <cuda_runtime.h>
#else
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifndef PRINT_LOG
#define PRINT_LOG 0
#endif

#include <sstream>

#define MAX_SOURCE_SIZE (0x100000)

char * deblank(char *str) {
	char *out = str, *put = str; 
	for(; *str != '\0'; ++str) {
		if((*str != ' ') && (*str != ':') && (*str != '(') && (*str != ')') && (*str != '[') && (*str != ']') && (*str != '<') && (*str != '>')) {
			*put++ = *str; 
		}    
	}    
	*put = '\0';
	return out; 
}   

int main (){
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
	//Generate ptx files for .cu, only if nvcc is found on the system
	if (system("which nvcc")==0){
		CUresult err;
		int major, minor;
		int max_threads_per_block;
		CUdevice cuDevice;
		CUcontext cuContext;
		CUmodule cuModule;
		int numDevices = 0;
		if( cudaGetDeviceCount(&numDevices) != cudaSuccess ) {
			fprintf(stderr, "[ERROR in CUDA binary creation] no available NVIDIA GPU found!; exit!\n");
			exit(1);
		}
		
		for(int i=0 ; i < numDevices; i++) {
			cuDeviceGet(&cuDevice, i);
			#if CUDA_VERSION >= 5000
			cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
			cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
			#else
				cuDeviceComputeCapability(&major, &minor, cuDevice);
			#endif
			cuDeviceGetAttribute (&max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuDevice);

			std::stringstream ss;
			ss << major;
			ss << minor;
			std::string version = ss.str();
			std::string ptxName = std::string("openarc_kernel_") + version + std::string(".ptx");
			fprintf(stderr, "[INFO] Create %s for device %d\n", ptxName.c_str(), i);
			fprintf(stderr, "[INFO] Max # of threads per thread block for device %d: %d\n", i, max_threads_per_block);
			std::string command = std::string("nvcc $OPENARC_JITOPTION -arch=sm_") + version + std::string(" openarc_kernel.cu -ptx -o ") + ptxName;
			system(command.c_str());
		}
	} else {
		fprintf(stderr, "[ERROR in CUDA binary creation] cannot find the NVIDIA CUDA compiler (nvcc) \n");
	}

#else

	cl_platform_id clPlatform;
	cl_device_id clDevice;
	cl_context clContext;
	cl_command_queue clQueue;
	cl_program clProgram;
	int isMic=0;
	cl_uint numDevices;
	cl_platform_id platform;
	clGetPlatformIDs(1, &platform, NULL);
	cl_int err;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	//Check for MIC if GPU is not found
	if (err != CL_SUCCESS) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices);
		isMic = 1;
	}
	if (err != CL_SUCCESS) {
		fprintf(stderr, "[ERROR in clGetDeviceIDs()] Failed to get device IDs  for type \n");
	}
	
	
	cl_device_id devices[numDevices];
	clGetPlatformIDs(1, &clPlatform, NULL);
	if(isMic)
		clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, NULL);
	else
		clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	
	for(int i=0; i< numDevices; i++) {
		clDevice = devices[i];
		
		FILE *fp;
		char *source_str;
		size_t source_size;
		char filename[] = "openarc_kernel.cl";
		fp = fopen(filename, "r");
		if (!fp) {
			fprintf(stderr, "[INFO: in OpenCL binary creation] Failed to read the kernel file %s, so skipping binary generation for OpenCL devices %d\n", filename, i);
			exit(1);
		}
		source_str = (char*)malloc(MAX_SOURCE_SIZE);
		source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
		fclose( fp );

		cl_int err;
		clContext = clCreateContext( NULL, 1, &clDevice, NULL, NULL, &err);
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to create OPENCL context with error %d (OPENCL GPU)\n", err);
			exit(1);
		}

		clQueue = clCreateCommandQueue(clContext, clDevice, 0, &err);
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to create OPENCL queue with error %d (OPENCL GPU)\n", err);
			exit(1);
		}

		size_t max_work_group_size;
		clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
		
		char cBuffer[1024];
		char *cBufferN;
		clGetDeviceInfo(clDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
		cBufferN = deblank(cBuffer);
		
		std::string binaryName = std::string("openarc_kernel_") + cBufferN + std::string(".ptx");
		
		fprintf(stderr, "[INFO] Create %s for device %d\n", binaryName.c_str(), i);
		fprintf(stderr, "[INFO] Max # of work-items in a work-group for device %d: %lu\n", i, max_work_group_size);
		clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to create OPENCL program with error %d (OPENCL GPU)\n", err);
			exit(1);
		}
		
		char *envVar;
		envVar = getenv("OPENARC_JITOPTION");
		err = clBuildProgram(clProgram, 1, &clDevice, envVar, NULL, NULL);
#if PRINT_LOG == 0
		if(err != CL_SUCCESS)
		{
				printf("[ERROR in OpenCL binary creation] Error in clBuildProgram, Line %u in file %s : %d!!!\n\n", __LINE__, __FILE__, err);
				if (err == CL_BUILD_PROGRAM_FAILURE)
				{
						// Determine the size of the log
						size_t log_size;
						clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

						// Allocate memory for the log
						char *log = (char *) malloc(log_size);

						// Get the log
						clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

						// Print the log
						printf("%s\n", log);
				}
				exit(1);
		}
#else
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
#endif
		size_t size;
		err = clGetProgramInfo( clProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL );
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to get OPENCL program info error %d (OPENCL GPU)\n", err);
			exit(1);
		}

		unsigned char * binary = new unsigned char [size];
		
		//#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
		//err = clGetProgramInfo( clProgram, CL_PROGRAM_BINARIES, size, &binary, NULL );
		//#else
		err = clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL);
		//#endif
		
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to dump OPENCL program binary error %d (OPENCL GPU)\n", err);
			exit(1);
		}
		
		FILE * fpbin = fopen(binaryName.c_str(), "wb" );
		fwrite(binary, 1 , size, fpbin);
		fclose(fpbin);
		delete[] binary;
	}	

#endif

}
