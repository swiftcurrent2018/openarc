#include "openacc.h"
#include "openaccrt.h"
#include "openaccrt_ext.h"
#include <unistd.h>

#define MAX_SOURCE_SIZE (0x100000)
//[DEBUG] commented out since it is no more static.
//std::vector<std::string> OpenCLDriver::kernelNameList;

cl_context OpenCLDriver_t::clContext;

char * deblank(char *str)
{
  char *out = str, *put = str;

  for(; *str != '\0'; ++str)
  {
    if(*str != ' ')
      *put++ = *str;
  }
  *put = '\0';

  return out;
}

///////////////////////////
// Device Initialization //
///////////////////////////
OpenCLDriver::OpenCLDriver(acc_device_t devType, int devNum, std::vector<std::string>kernelNames, HostConf_t *conf, int numDevices) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::OpenCLDriver()\n");
	}	
#endif
    dev = devType;
    device_num = devNum;
	num_devices = numDevices;
//Moved to init();
/*
    cl_device_id devices[numDevices];
    clGetPlatformIDs(1, &clPlatform, NULL);
    
    if(devType == acc_device_xeonphi)
		clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, NULL);
    else 
		clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		
    clDevice = devices[devNum];
    char cBuffer[1024];
    clGetDeviceInfo(clDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
    int thread_id = get_thread_id();
    fprintf(stderr, "OpenCL : Host Thread %d initializes device %d: %s\n", thread_id, device_num, cBuffer);
*/

    for (std::vector<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
        //const char *kernelName = (*it).c_str();
		//fprintf(stderr, "[OpenCLDriver()] Kernel name = %s\n", kernelName);
        kernelNameList.push_back(*it);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::OpenCLDriver()\n");
	}
#endif
}

HI_error_t OpenCLDriver::init() {
    FILE *fp;
    char *source_str;
    size_t source_size;
    char filename[] = "openarc_kernel.cl";
	char kernel_keyword[] = "__kernel";
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::init()\n");
	}
#endif
    cl_device_id devices[num_devices];
    clGetPlatformIDs(1, &clPlatform, NULL);
    
    if(dev == acc_device_xeonphi)
		clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ACCELERATOR, num_devices, devices, NULL);
    else 
		clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    HostConf_t * tconf = getHostConf();
	if( tconf->use_unifiedmemory > 0 ) {
		//[FIXME] Need to check whether unified memory is supported or not.
		unifiedMemSupported = 0;
	} else {	
		unifiedMemSupported = 0;
	}
		
    clDevice = devices[device_num];
    char cBuffer1[1024];
    clGetDeviceInfo(clDevice, CL_DEVICE_NAME, sizeof(cBuffer1), &cBuffer1, NULL);
    int thread_id = get_thread_id();
#ifdef _OPENARC_PROFILE_
    fprintf(stderr, "OpenCL : Host Thread %d initializes device %d: %s\n", thread_id, device_num, cBuffer1);
#endif
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR in OpenCLDriver::init()] Failed to read the kernel file %s.\n", filename);
		exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    cl_int err;
#ifdef _OPENMP
#pragma omp critical(clContext_critical)
#endif
    {
        if (clContext == NULL) {
            clContext = clCreateContext( NULL, num_devices, devices, NULL, NULL, &err);
            if(err != CL_SUCCESS) {
                fprintf(stderr, "[ERROR in OpenCLDriver::init()] failed to create OPENCL context with error %d (OPENCL Device)\n", err);
                exit(1);
            }
        }
    }

//[DEBUG by Seyong] below is commented out since it does not have any effect.
/*
    clQueue = clCreateCommandQueue(clContext, clDevice, 0, &err);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::init()] failed to create OPENCL queue with error %d (OPENCL Device)\n", err);
    }
*/

    char cBuffer[1024];
    char* cBufferN;
    clGetDeviceInfo(clDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
	cBufferN = deblank(cBuffer); //Remove spaces.
    std::string binaryName = std::string("openarc_kernel_") + cBufferN + std::string(".ptx");

    //Build the program from source if the binary file is not found
    if( access( binaryName.c_str(), F_OK ) == -1 ) {

		if( strstr(source_str, kernel_keyword) != NULL ) { 
			//Compile the kernel file only if a kernel exists.
        	clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
        	if(err != CL_SUCCESS) {
            	fprintf(stderr, "[ERROR in OpenCLDriver::init()] failed to create OPENCL program with error %d (OPENCL Device)\n", err);
				exit(1);
        	}

			char *envVar;
			envVar = getenv("OPENARC_JITOPTION");
       		err = clBuildProgram(clProgram, 1, &clDevice, envVar, NULL, NULL);
        	if(err != CL_SUCCESS)
        	{
            	printf("[ERROR in OpenCLDriver::init()] Error in clBuildProgram, Line %u in file %s : %d!!!\n\n", __LINE__, __FILE__, err);
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

#if 0 /* disabled by JK */
        	size_t size;
        	err = clGetProgramInfo( clProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL );
        	if(err != CL_SUCCESS) {
            	fprintf(stderr, "[ERROR in OpenCLDriver::init()] failed to get OPENCL program info error %d (OPENCL Device)\n", err);
				exit(1);
        	}

        	unsigned char * binary = new unsigned char [size];

#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
        	err = clGetProgramInfo( clProgram, CL_PROGRAM_BINARIES, size, &binary, NULL );
#else
        	err = clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL);
#endif

        	if(err != CL_SUCCESS) {
            	fprintf(stderr, "[ERROR in OpenCLDriver::init()] failed to dump OPENCL program binary error %d (OPENCL Device)\n", err);
				exit(1);
        	}

        	FILE * fpbin = fopen(binaryName.c_str(), "wb" );
        	fwrite(binary, 1 , size, fpbin);
        	fclose(fpbin);
        	delete[] binary;
#endif
		} else {
			clProgram = NULL;
		}
    } // Binary file is found; build the program from it
    else {
        size_t binarySize;
        FILE *fp = fopen(binaryName.c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        binarySize = ftell(fp);
        rewind(fp);
        unsigned char *programBinary = new unsigned char[binarySize];
        fread(programBinary, 1, binarySize, fp);
        fclose(fp);

        cl_int binaryStatus;
        clProgram = clCreateProgramWithBinary(clContext, 1, &clDevice, &binarySize, (const unsigned char**)&programBinary,
                                              &binaryStatus, &err);

        if(err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR in OpenCLDriver::init()] failed to read OPENCL program binary error %d (OPENCL Device)\n", err);
			exit(1);
        }

        if(binaryStatus != CL_SUCCESS) {
            fprintf(stderr, "[ERROR in OpenCLDriver::init()] Invalid binary found for the device (OPENCL Device)\n");
			exit(1);
        }

        err = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
        if(err != CL_SUCCESS)
        {
            printf("[ERROR in OpenCLDriver::init()] Error in clBuildProgram, Line %u in file %s : %d!!!\n\n", __LINE__, __FILE__, err);
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

    }

    cl_command_queue s0, s1;
    s0 = clCreateCommandQueue(clContext, clDevice, 0, &err);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::init()] failed to create OPENCL queue with error %d (OPENCL Device)\n", err);
		exit(1);
    }
    s1 = clCreateCommandQueue(clContext, clDevice, 0, &err);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::init()] failed to create OPENCL queue with error %d (OPENCL Device)\n", err);
		exit(1);
    }
    queueMap[0] = s0;
    queueMap[1] = s1;

    //make s0 the default queue
    clQueue = s0;

    cl_event e0, e1;
    std::map<int, cl_event> eventMap;
    e0 = clCreateUserEvent(clContext, &err);
    if(err != CL_SUCCESS) {
        printf("[ERROR in OpenCLDriver::init()] Error in clCreateUserEvent, Line %u in file %s : %d!!!\n\n", __LINE__, __FILE__, err);
		exit(1);
    }

    clSetUserEventStatus(e0, CL_COMPLETE);

    e1 = clCreateUserEvent(clContext, &err);
    if(err != CL_SUCCESS) {
        printf("[ERROR in OpenCLDriver::init()] Error in clCreateUserEvent, Line %u in file %s : %d!!!\n\n", __LINE__, __FILE__, err);
		exit(1);
    }

    clSetUserEventStatus(e1, CL_COMPLETE);

    eventMap[0]= e0;
    eventMap[1]= e1;
    threadQueueEventMap[get_thread_id()] = eventMap;


    createKernelArgMap();
    init_done = 1;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::init()\n");
	}
#endif
    return HI_success;
}

HI_error_t OpenCLDriver::HI_pin_host_memory(const void* hostPtr, size_t size) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_pin_host_memory()\n");
	}
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_pin_host_memory()\n");
	}
#endif
	return HI_success;
}

void OpenCLDriver::HI_unpin_host_memory(const void* hostPtr) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_unpin_host_memory()\n");
	}
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_unpin_host_memory()\n");
	}
#endif
	return;
}

HI_error_t OpenCLDriver::createKernelArgMap() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::createKernelArgMap()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    cl_int err;
    std::map<std::string, cl_kernel> kernelMap;
	std::map<std::string, kernelParams_t*> kernelArgs;
    //find all kernels and insert them in the map for the corresponding device
    for(std::vector<std::string>::iterator it=kernelNameList.begin(); it!=kernelNameList.end(); ++it) {
        cl_kernel clFunc;
        const char *kernelName = (*it).c_str();
        clFunc = clCreateKernel(clProgram, kernelName, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[%d] [ERROR in OpenCLDriver::init()] Function Load FAIL on %s, %d\n", __LINE__, kernelName, err);
			exit(1);
        }
        kernelMap[*it] = clFunc;

        // Create argument mapping for the kernel
		// Below is not used for now.
		kernelParams_t *kernelParams = new kernelParams_t;
		kernelParams->num_args = 0;
		kernelArgs.insert(std::pair<std::string, kernelParams_t*>(std::string(kernelName), kernelParams));
    }
	tconf->kernelArgsMap[this] = kernelArgs;
    tconf->kernelsMap[this]=kernelMap;

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::createKernelArgMap()\n");
	}
#endif
    return HI_success;
}

HI_error_t OpenCLDriver::HI_register_kernels(std::vector<std::string> kernelNames) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_register_kernels()\n");
	}
#endif
    for (std::vector<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
        //const char *kernelName = (*it).c_str();
		//fprintf(stderr, "[OpenCLDriver()] Kernel name = %s\n", kernelName);
        kernelNameList.push_back(*it);
    }
    HostConf_t * tconf = getHostConf();
    cl_int err;
	std::map<std::string, kernelParams_t*> kernelArgs;
    std::map<std::string, cl_kernel> kernelMap;
    //find all kernels and insert them in the map for the corresponding device
    for(std::vector<std::string>::iterator it=kernelNameList.begin(); it!=kernelNameList.end(); ++it) {
        cl_kernel clFunc;
        const char *kernelName = (*it).c_str();
        clFunc = clCreateKernel(clProgram, kernelName, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[%d] [ERROR in OpenCLDriver::init()] Function Load FAIL on %s, %d\n", __LINE__, kernelName, err);
			exit(1);
        }
        kernelMap[*it] = clFunc;

        // Create argument mapping for the kernel
		// Below is not used for now.
		kernelParams_t *kernelParams = new kernelParams_t;
		kernelParams->num_args = 0;
		kernelArgs.insert(std::pair<std::string, kernelParams_t*>(std::string(kernelName), kernelParams));
    }
	tconf->kernelArgsMap[this] = kernelArgs;
    tconf->kernelsMap[this]=kernelMap;

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_register_kernels()\n");
	}
#endif
    return HI_success;
}

int OpenCLDriver::HI_get_num_devices(acc_device_t devType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_get_num_devices()\n");
	}
#endif
    cl_uint numDevices = 0;
    if(devType == acc_device_gpu || devType == acc_device_xeonphi) {
        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, NULL);
        cl_int err;
        if(devType == acc_device_xeonphi)
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices);
		else
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
			
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR in OpenCLDriver::HI_get_num_devices()] Failed to get device IDs  for type %d\n", devType);
			exit(1);
        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_get_num_devices()\n");
	}
#endif
    return (int) numDevices;
}


HI_error_t OpenCLDriver::destroy() {

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::destroy()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    cl_int err;

    err = clFlush(clQueue);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::destroy()] failed to flush OPENCL queue with error %d (OPENCL Device)\n", err);
		exit(1);
        return HI_error;
    }
    err = clFinish(clQueue);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::destroy()] failed to finish OPENCL queue with error %d (OPENCL Device)\n", err);
		exit(1);
        return HI_error;
    }

    std::map<std::string, cl_kernel> kernels = tconf->kernelsMap.at(this);
    for(std::map<std::string, cl_kernel>::iterator it=kernels.begin(); it!=kernels.end(); ++it) {
        err = clReleaseKernel(it->second);
        if(err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR in OpenCLDriver::destroy()] failed to release OPENCL kernel with error %d (OPENCL Device)\n", err);
			exit(1);
            return HI_error;
        }
    }
	if( clProgram != NULL ) {
    	err = clReleaseProgram(clProgram);
    	if(err != CL_SUCCESS) {
        	fprintf(stderr, "[ERROR in OpenCLDriver::destroy()] failed to release OPENCL program with error %d (OPENCL Device)\n", err);
			exit(1);
        	return HI_error;
    	}
	}
    err = clReleaseCommandQueue(clQueue);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::destroy()] failed to release OPENCL queue with error %d (OPENCL Device)\n", err);
		exit(1);
        return HI_error;
    }
#ifdef _OPENMP
#pragma omp critical(clContext_critical)
#endif
    {
        if (clContext != NULL) {
            err = clReleaseContext(clContext);
            if(err != CL_SUCCESS) {
                fprintf(stderr, "[ERROR in OpenCLDriver::destroy()] failed to release OPENCL context with error %d (OPENCL Device)\n", err);
                exit(1);
            }
            clContext = NULL;
        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::destroy()\n");
	}
#endif
    return HI_success;
}


HI_error_t  OpenCLDriver::HI_malloc1D(const void *hostPtr, void **devPtr, int count, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_malloc1D(%d, %u)\n", asyncID, count);
	}
#endif
    HostConf_t * tconf = getHostConf();

    if( tconf->device->init_done == 0 ) {
        //fprintf(stderr, "[in HI_malloc1D()] : initing!\n");
        tconf->HI_init(DEVICE_NUM_UNDEFINED);
    }
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HI_error_t result = HI_error;
    cl_int  err = CL_SUCCESS;

    if(HI_get_device_address(hostPtr, devPtr, asyncID) == HI_success ) {
        result = HI_success;
        //fprintf(stderr, "[in HI_malloc1D()] : address found!\n");
    } else {
		std::multimap<size_t, void *>::iterator it = memPool.find((size_t) count);
		if( it != memPool.end()) {
			*devPtr = it->second;
			memPool.erase(it);
		} else {
        	(*devPtr) = (void*) clCreateBuffer(clContext, CL_MEM_READ_WRITE, (size_t) count, NULL, &err);
        	if(err != CL_SUCCESS) {
            	//fprintf(stderr, "[ERROR in OpenCLDriver::HI_malloc1D()] : Malloc failed\n");
				for (it = memPool.begin(); it != memPool.end(); ++it) {
					*devPtr = it->second;
        			err = clReleaseMemObject((cl_mem)(*devPtr));
        			if(err != CL_SUCCESS) {
            			fprintf(stderr, "[ERROR in OpenCLDriver::HI_malloc1D()] : failed to free on OpenCL\n");
					}
				}
				memPool.clear();
        		(*devPtr) = (void*) clCreateBuffer(clContext, CL_MEM_READ_WRITE, (size_t) count, NULL, &err);
			}
		}
        if(err == CL_SUCCESS) {
            HI_set_device_address(hostPtr, *devPtr, (size_t) count, asyncID);
            //fprintf(stderr, "[in HI_malloc1D()] : address not found, so malloced \n");
#ifdef _OPENARC_PROFILE_
            tconf->DMallocCnt++;
#endif
            result = HI_success;
        } else {
            fprintf(stderr, "[ERROR in OpenCLDriver::HI_malloc1D()] : Malloc failed\n");
			exit(1);
        }
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_malloc1D(%d, %u)\n", asyncID, count);
	}
#endif
    return result;

}

//[FIXME] Implement this!
HI_error_t  OpenCLDriver::HI_malloc1D_unified(const void *hostPtr, void **devPtr, int count, int asyncID) {
	fprintf(stderr, "[OPENARCRT-ERROR]OpenCLDriver::HI_malloc1D_unified() is not yet implemented!\n");
	//exit(1);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_malloc1D_unified(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();

    if( tconf->device->init_done == 0 ) {
        //fprintf(stderr, "[in HI_malloc1D_unified()] : initing!\n");
        tconf->HI_init(DEVICE_NUM_UNDEFINED);
    }
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HI_error_t result = HI_error;
    cl_int  err;

    if(HI_get_device_address(hostPtr, devPtr, asyncID) == HI_success ) {
        result = HI_success;
        //fprintf(stderr, "[in HI_malloc1D_unified()] : address found!\n");
    } else {
		if( unifiedMemSupported == 0 ) {
			result = HI_success;
            fprintf(stderr, "[OPENARCRT-WARNING in OpenCLDriver::HI_malloc1D_unified(%d)] unified memory is either not supported or disabled in the current device; device memory should be explicitly managed either through data clauses or though runtime APIs.\n", asyncID);
			if( hostPtr == NULL ) {
            	*devPtr = malloc((size_t)count);
			} else {
				*devPtr = (void *)hostPtr;
			}
		} else {
        	(*devPtr) = (void*) clCreateBuffer(clContext, CL_MEM_READ_WRITE, (size_t) count, NULL, &err);
        	if(err == CL_SUCCESS) {
            	HI_set_device_address(*devPtr, *devPtr, (size_t) count, asyncID);
            	//fprintf(stderr, "[in HI_malloc1D_unified()] : address not found, so malloced \n");
#ifdef _OPENARC_PROFILE_
            	tconf->DMallocCnt++;
#endif
            	result = HI_success;
        	} else {
            	fprintf(stderr, "[ERROR in OpenCLDriver::HI_malloc1D_unified()] : Malloc failed\n");
				exit(1);
        	}
		}
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_malloc1D_unified(%d)\n", asyncID);
	}
#endif
    return result;

}

//the ElementSizeBytes in cuMemAllocPitch is currently set to 16.
HI_error_t OpenCLDriver::HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_malloc2D(%d)\n", asyncID);
	}
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_malloc2D(%d)\n", asyncID);
	}
#endif

    return HI_success;
}


HI_error_t OpenCLDriver::HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_malloc3D(%d)\n", asyncID);
	}
#endif
    /*if( tconf == NULL ) {
    #ifdef _OPENMP
    	int thread_id = omp_get_thread_num();
    #else
    	int thread_id = 0;
    #endif
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_malloc3D()] No host configuration exists for the current host thread (thread ID: %d); please set an environment variable, OMP_NUM_THREADS, to the maximum number of OpenMP threads used for your application; exit!\n", thread_id);
        exit(1);
    }
    if( tconf->HI_init_done == 0 ) {
    	tconf->HI_init(DEVICE_NUM_UNDEFINED);
    }
    #ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
    #endif
    //TODO
    HI_error_t result;
    result = HI_error;
    #ifdef _OPENARC_PROFILE_
    tconf->DMallocCnt++;
    tconf->totalMallocTime += HI_get_localtime() - ltime;
    #endif
    return result; */
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_malloc3D(%d)\n", asyncID);
	}
#endif
    return HI_success;
}



HI_error_t OpenCLDriver::HI_free( const void *hostPtr, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_free(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();

    if( tconf->device->init_done == 0 ) {
        tconf->HI_init(DEVICE_NUM_UNDEFINED);
    }
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HI_error_t result = HI_success;
    void *devPtr;
	size_t size;
    //Check if the mapping exists. Free only if a mapping is found
    if( HI_get_device_address(hostPtr, &devPtr, NULL, &size, asyncID) != HI_error) {
       //If this method is called for unified memory, memory deallocation
       //is skipped; unified memory will be deallocatedd only by 
       //HI_free_unified().
		if( hostPtr != devPtr ) {
			//We do not free the device memory; instead put it in the memory pool
			//and remove host-pointer-to-device-pointer mapping
			memPool.insert(std::pair<size_t, void *>(size, devPtr));
			HI_remove_device_address(hostPtr, asyncID);
/*
        	cl_int  err = clReleaseMemObject((cl_mem)(devPtr));
        	if( err == CL_SUCCESS ) {
            	HI_remove_device_address(hostPtr, asyncID);

        	} else {
            	fprintf(stderr, "[ERROR in OpenCLDriver::HI_free()] OpenCL memory free failed with error %d\n", err);
				exit(1);
            	result = HI_error;
        	}
*/
#ifdef _OPENARC_PROFILE_
			tconf->DFreeCnt++;
#endif
		}
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_free(%d)\n", asyncID);
	}
#endif
    return result;

}

//[FIXME] Implement this!
HI_error_t OpenCLDriver::HI_free_unified( const void *hostPtr, int asyncID) {
	fprintf(stderr, "[OPENARCRT-ERROR]OpenCLDriver::HI_free_unified() is not yet implemented!\n");
	exit(1);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_free_unified(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();

    if( tconf->device->init_done == 0 ) {
        tconf->HI_init(DEVICE_NUM_UNDEFINED);
    }
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HI_error_t result = HI_success;
    void *devPtr;
    //Check if the mapping exists. Free only if a mapping is found
    if( HI_get_device_address(hostPtr, &devPtr, asyncID) != HI_error) {
		if( unifiedMemSupported == 0 ) {
			free(devPtr);
		} else {
        	cl_int  err = clReleaseMemObject((cl_mem)(devPtr));
        	if( err == CL_SUCCESS ) {
            	HI_remove_device_address(hostPtr, asyncID);

#ifdef _OPENARC_PROFILE_
            	tconf->DFreeCnt++;
#endif
        	} else {
            	fprintf(stderr, "[ERROR in OpenCLDriver::HI_free_unified()] OpenCL memory free failed with error %d\n", err);
				exit(1);
            	result = HI_error;
        	}
		}
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_free_unified(%d)\n", asyncID);
	}
#endif
    return result;

}



//malloc used for allocating temporary data.
//If the method is called for a pointer to existing memory, the existing memory
//will be freed before allocating new memory.
void OpenCLDriver::HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_tempMalloc1D()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    if(  devType == acc_device_gpu || devType == acc_device_nvidia || 
    devType == acc_device_radeon || devType == acc_device_xeonphi || devType == acc_device_current) {
		if( tempMallocSet.count(*tempPtr) > 0 ) {
			tempMallocSet.erase(*tempPtr);
            clReleaseMemObject((cl_mem)(*tempPtr));
#ifdef _OPENARC_PROFILE_
            tconf->DFreeCnt++;
#endif
        }
        cl_int err;
        (*tempPtr) = (void*) clCreateBuffer(clContext, CL_MEM_READ_WRITE, (size_t) count, NULL, &err);
		tempMallocSet.insert(*tempPtr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR in OpenCLDriver::HI_tempMalloc1D()] : Malloc failed\n");
			exit(1);
        }
#ifdef _OPENARC_PROFILE_
        tconf->DMallocCnt++;
#endif
    } else {
		if( tempMallocSet.count(*tempPtr) > 0 ) {
			tempMallocSet.erase(*tempPtr);
            free(*tempPtr);
#ifdef _OPENARC_PROFILE_
            tconf->HFreeCnt++;
#endif
        }
        *tempPtr = malloc(count);
		tempMallocSet.insert(*tempPtr);
#ifdef _OPENARC_PROFILE_
        tconf->HMallocCnt++;
#endif
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_tempMalloc1D()\n");
	}
#endif

}

//Used for de-allocating temporary data.
void OpenCLDriver::HI_tempFree( void** tempPtr, acc_device_t devType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_tempFree()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    if(  devType == acc_device_gpu || devType == acc_device_nvidia || 
    devType == acc_device_radeon || devType == acc_device_xeonphi || devType == acc_device_current) {
        if( *tempPtr != 0 ) {
			tempMallocSet.erase(*tempPtr);
            clReleaseMemObject((cl_mem)(*tempPtr));
#ifdef _OPENARC_PROFILE_
            tconf->DFreeCnt++;
#endif
        }
    } else {
        if( *tempPtr != 0 ) {
			tempMallocSet.erase(*tempPtr);
            free(*tempPtr);
#ifdef _OPENARC_PROFILE_
            tconf->HFreeCnt++;
#endif
        }
    }
    *tempPtr = 0;
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_tempFree()\n");
	}
#endif

}


//////////////////////
// Kernel Execution //
//////////////////////


//In the driver API, copying into a constant memory (symbol) does not require a different API call
HI_error_t  OpenCLDriver::HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_memcpy(%lu)\n", count);
	}
#endif
    HostConf_t * tconf = getHostConf();

    cl_int  err = CL_SUCCESS;
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    //err = cudaMemcpy(dst, src, count, toCudaMemcpyKind(kind));
    cl_command_queue queue = getQueue(DEFAULT_QUEUE);
    //cl_command_queue queue = queueMap.at(0);
    if( dst != src ) {
    	switch( kind ) {
    	case HI_MemcpyHostToHost: {
        	fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy()] Host to Host transfers not supported\n");
			exit(1);
        	break;
    	}
    	case HI_MemcpyHostToDevice: {
        	err = clEnqueueWriteBuffer(queue, (cl_mem) dst, CL_TRUE, 0, count, src, 0, NULL, NULL);
        	break;
    	}
    	case HI_MemcpyDeviceToHost: {
        	err = clEnqueueReadBuffer(queue, (cl_mem)src, CL_TRUE, 0, count, dst, 0, NULL, NULL);
        	break;
    	}
    	case HI_MemcpyDeviceToDevice: {
        	fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy()] Device to Device transfers not supported\n");
			exit(1);
        	break;
    	}
    	}
	}
#ifdef _OPENARC_PROFILE_
    if( dst != src ) {
    	if( kind == HI_MemcpyHostToDevice ) {
        	tconf->H2DMemTrCnt++;
        	tconf->H2DMemTrSize += count;
    	} else if( kind == HI_MemcpyDeviceToHost ) {
        	tconf->D2HMemTrCnt++;
        	tconf->D2HMemTrSize += count;
    	} else if( kind == HI_MemcpyDeviceToDevice ) {
        	tconf->D2DMemTrCnt++;
        	tconf->D2DMemTrSize += count;
    	} else {
        	tconf->H2HMemTrCnt++;
        	tconf->H2HMemTrSize += count;
    	}
	}
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( err == CL_SUCCESS ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy(%lu)\n", count);
	}
#endif
        return HI_success;
    } else {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy(%lu)\n", count);
	}
#endif
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy()] Memcpy failed with error %d\n", err);
		exit(1);
        return HI_error;
    }
}

//[FIXME] Implement this!
HI_error_t  OpenCLDriver::HI_memcpy_unified(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType) {
	fprintf(stderr, "[OPENARCRT-ERROR]OpenCLDriver::HI_memcpy_unified() is not yet implemented!\n");
	exit(1);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_memcpy_unified(%lu)\n", count);
	}
#endif
    HostConf_t * tconf = getHostConf();

    cl_int  err = CL_SUCCESS;
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    //err = cudaMemcpy(dst, src, count, toCudaMemcpyKind(kind));
    cl_command_queue queue = getQueue(DEFAULT_QUEUE);
    //cl_command_queue queue = queueMap.at(0);
    if( dst != src ) {
    	switch( kind ) {
    	case HI_MemcpyHostToHost: {
        	fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_unified()] Host to Host transfers not supported\n");
			exit(1);
        	break;
    	}
    	case HI_MemcpyHostToDevice: {
        	err = clEnqueueWriteBuffer(queue, (cl_mem) dst, CL_TRUE, 0, count, src, 0, NULL, NULL);
        	break;
    	}
    	case HI_MemcpyDeviceToHost: {
        	err = clEnqueueReadBuffer(queue, (cl_mem)src, CL_TRUE, 0, count, dst, 0, NULL, NULL);
        	break;
    	}
    	case HI_MemcpyDeviceToDevice: {
        	fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_unified()] Device to Device transfers not supported\n");
			exit(1);
        	break;
    	}
    	}
	}
#ifdef _OPENARC_PROFILE_
    if( dst != src ) {
    	if( kind == HI_MemcpyHostToDevice ) {
        	tconf->H2DMemTrCnt++;
        	tconf->H2DMemTrSize += count;
    	} else if( kind == HI_MemcpyDeviceToHost ) {
        	tconf->D2HMemTrCnt++;
        	tconf->D2HMemTrSize += count;
    	} else if( kind == HI_MemcpyDeviceToDevice ) {
        	tconf->D2DMemTrCnt++;
        	tconf->D2DMemTrSize += count;
    	} else {
        	tconf->H2HMemTrCnt++;
        	tconf->H2HMemTrSize += count;
    	}
	}
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( err == CL_SUCCESS ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy_unified(%lu)\n", count);
	}
#endif
        return HI_success;
    } else {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_unified()] Memcpy failed with error %d\n", err);
		exit(1);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy_unified(%lu)\n", count);
	}
#endif
        return HI_error;
    }
}


HI_error_t OpenCLDriver::HI_memcpy_async(void *dst, const void *src, size_t count,
        HI_MemcpyKind_t kind, int trType, int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_memcpy_async(%d, %lu)\n", async, count);
	}
#endif
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    //err = cudaMemcpy(dst, src, count, toCudaMemcpyKind(kind));
    cl_int  err = CL_SUCCESS;
    cl_command_queue queue = getQueue(async);
    cl_event *event = getEvent(async);
	if( dst != src ) {
    	switch( kind ) {
    	case HI_MemcpyHostToHost: {
        	fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_async()] Host to Host transfers not supported\n");
			exit(1);
        	break;
    	}
    	case HI_MemcpyHostToDevice: {
        	err = clEnqueueWriteBuffer(queue, (cl_mem) dst, CL_FALSE, 0, count, src, 0, NULL, event);
        	break;
    	}
    	case HI_MemcpyDeviceToHost: {
        	err = clEnqueueReadBuffer(queue, (cl_mem)src, CL_FALSE, 0, count, dst, 0, NULL, event);
        	break;
    	}
    	case HI_MemcpyDeviceToDevice: {
        	fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_async()] Device to Device transfers not supported\n");
			exit(1);
        	break;
    	}
    	}
	}
#ifdef _OPENARC_PROFILE_
	if( dst != src ) {
    	if( kind == HI_MemcpyHostToDevice ) {
        	tconf->H2DMemTrCnt++;
        	tconf->H2DMemTrSize += count;
    	} else if( kind == HI_MemcpyDeviceToHost ) {
        	tconf->D2HMemTrCnt++;
        	tconf->D2HMemTrSize += count;
    	} else if( kind == HI_MemcpyDeviceToDevice ) {
        	tconf->D2DMemTrCnt++;
        	tconf->D2DMemTrSize += count;
    	} else {
        	tconf->H2HMemTrCnt++;
        	tconf->H2HMemTrSize += count;
    	}
	}
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( err == CL_SUCCESS ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy_async(%d, %lu)\n", async, count);
	}
#endif
        return HI_success;
    } else {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_async()] Memcpy failed with error %d\n", err);
		exit(1);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy_async(%d, %lu)\n", async, count);
	}
#endif
        return HI_error;
    }
}

HI_error_t OpenCLDriver::HI_memcpy_asyncS(void *dst, const void *src, size_t count,
        HI_MemcpyKind_t kind, int trType, int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_memcpy_asyncS(%d)\n", async);
	}
#endif
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    //err = cudaMemcpy(dst, src, count, toCudaMemcpyKind(kind));
    cl_int  err;
    cl_command_queue queue = getQueue(async);
    cl_event *event = getEvent(async);
    switch( kind ) {
    case HI_MemcpyHostToHost: {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_asyncS()] Host to Host transfers not supported\n");
		exit(1);
        break;
    }
    case HI_MemcpyHostToDevice: {
        err = clEnqueueWriteBuffer(queue, (cl_mem) dst, CL_FALSE, 0, count, src, 0, NULL, event);
        break;
    }
    case HI_MemcpyDeviceToHost: {
		void *tDst;
		HI_tempMalloc1D(&tDst, count, acc_device_host);
		HI_set_temphost_address(dst, tDst, async);
        err = clEnqueueReadBuffer(queue, (cl_mem)src, CL_FALSE, 0, count, tDst, 0, NULL, event);
        break;
    }
    case HI_MemcpyDeviceToDevice: {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_asyncS()] Device to Device transfers not supported\n");
		exit(1);
        break;
    }
    }
#ifdef _OPENARC_PROFILE_
    if( kind == HI_MemcpyHostToDevice ) {
        tconf->H2DMemTrCnt++;
        tconf->H2DMemTrSize += count;
    } else if( kind == HI_MemcpyDeviceToHost ) {
        tconf->D2HMemTrCnt++;
        tconf->D2HMemTrSize += count;
    } else if( kind == HI_MemcpyDeviceToDevice ) {
        tconf->D2DMemTrCnt++;
        tconf->D2DMemTrSize += count;
    } else {
        tconf->H2HMemTrCnt++;
        tconf->H2HMemTrSize += count;
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( err == CL_SUCCESS ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy_asyncS(%d)\n", async);
	}
#endif
        return HI_success;
    } else {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy_asyncS()] Memcpy failed with error %d\n", err);
		exit(1);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy_asyncS(%d)\n", async);
	}
#endif
        return HI_error;
    }
}


HI_error_t OpenCLDriver::HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
        size_t widthInBytes, size_t height, HI_MemcpyKind_t kind) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_memcpy2D()\n");
	}
#endif
    /*if( tconf == NULL ) {
    #ifdef _OPENMP
    	int thread_id = omp_get_thread_num();
    #else
    	int thread_id = 0;
    #endif
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_memcpy2D()] No host configuration exists for the current host thread (thread ID: %d); please set an environment variable, OMP_NUM_THREADS, to the maximum number of OpenMP threads used for your application; exit!\n", thread_id);
        exit(1);
    }
    if( tconf->HI_init_done == 0 ) {
    	tconf->HI_init(DEVICE_NUM_UNDEFINED);
    }
    #ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
    #endif
    cl_int  err;
    CUDA_MEMCPY2D pcopy;
    switch( kind ) {
    	case HI_MemcpyHostToHost: {pcopy.srcMemoryType =  CU_MEMORYTYPE_HOST;
    								  pcopy.dstMemoryType =  CU_MEMORYTYPE_HOST;
    								  pcopy.srcHost = src;
    								  pcopy.dstHost = dst;}
    	case HI_MemcpyHostToDevice: {pcopy.srcMemoryType =  CU_MEMORYTYPE_HOST;
    									pcopy.dstMemoryType =  CU_MEMORYTYPE_DEVICE;
    									pcopy.srcHost = src;
    									pcopy.dstDevice = (CUdeviceptr) dst;}
    	case HI_MemcpyDeviceToHost: {pcopy.srcMemoryType =  CU_MEMORYTYPE_DEVICE;
    									pcopy.dstMemoryType =  CU_MEMORYTYPE_HOST;
    									pcopy.srcDevice = (CUdeviceptr) src;
    									pcopy.dstHost = dst;}
    	case HI_MemcpyDeviceToDevice: {pcopy.srcMemoryType =  CU_MEMORYTYPE_DEVICE;
    									  pcopy.dstMemoryType =  CU_MEMORYTYPE_DEVICE;
    									  pcopy.srcDevice = (CUdeviceptr) src;
    									  pcopy.dstDevice = (CUdeviceptr) dst;}
    }

    pcopy.srcXInBytes = 0;
    pcopy.srcY = 0;
    pcopy.dstXInBytes = 0;
    pcopy.dstY = 0;
    pcopy.srcPitch = spitch;
    pcopy.dstPitch = dpitch;
    pcopy.WidthInBytes = widthInBytes;
    pcopy.Height = height;

    err = cuMemcpy2D(&pcopy);
    #ifdef _OPENARC_PROFILE_
    if( kind == HI_MemcpyHostToDevice ) {
    	tconf->H2DMemTrCnt++;
    	tconf->H2DMemTrSize += widthInBytes*height;
    } else if( kind == HI_MemcpyDeviceToHost ) {
    	tconf->D2HMemTrCnt++;
    	tconf->D2HMemTrSize += widthInBytes*height;
    } else if( kind == HI_MemcpyDeviceToDevice ) {
    	tconf->D2DMemTrCnt++;
    	tconf->D2DMemTrSize += widthInBytes*height;
    } else {
    	tconf->H2HMemTrCnt++;
    	tconf->H2HMemTrSize += widthInBytes*height;
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
    #endif
    if( err == CL_SUCCESS ) {
    	return HI_success;
    } else {
    	return HI_error;
    }*/
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy2D()\n");
	}
#endif
    return HI_success;
}

HI_error_t OpenCLDriver::HI_memcpy2D_async(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async) {

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_memcpy2D_async(%d)\n", async);
	}
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_memcpy2D_async(%d)\n", async);
	}
#endif
    return HI_success;
}

HI_error_t OpenCLDriver::HI_register_kernel_numargs(std::string kernel_name, int num_args)
{
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_register_kernel_numargs()\n");
	}
#endif
	//[DEBUG] below code is not used now.
    HostConf_t *tconf = getHostConf();
    //fprintf(stderr, "find kernelargs map for the current device\n");
    kernelParams_t *kernelParams = tconf->kernelArgsMap.at(this).at(kernel_name);
    if( kernelParams->num_args == 0 ) { 
        if( num_args > 0 ) { 
            kernelParams->num_args = num_args;
            //kernelParams->kernelParams = (void**)malloc(sizeof(void*) * num_args);
        } else { 
            fprintf(stderr, "[ERROR in OpenCLDriver::HI_register_kernel_numargs(%s, %d)] num_args should be greater than zero.\n",kernel_name.c_str(), num_args);
            exit(1);
        }        
    }        
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_register_kernel_numargs()\n");
	}
#endif
    return HI_success;
}


HI_error_t OpenCLDriver::HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value)
{
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_register_kernel_arg()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    cl_int err = clSetKernelArg((cl_kernel)(tconf->kernelsMap.at(this).at(kernel_name)), arg_index, arg_size, arg_value);
    if(err != CL_SUCCESS)
    {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_register_kernel_arg()] failed to add argument %d to kernel %s with error %d (OPENCL Device)\n", arg_index, kernel_name.c_str(), err);
		exit(1);
        return HI_error;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_register_kernel_arg()\n");
	}
#endif
    return HI_success;
}

HI_error_t OpenCLDriver::HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async)
{
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_kernel_call(%d)\n", async);
	}
#endif
    HostConf_t * tconf = getHostConf();
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    size_t globalSize[3];
    globalSize[0] = gridSize[0]*blockSize[0];
    globalSize[1] = gridSize[1]*blockSize[1];
    globalSize[2] = gridSize[2]*blockSize[2];

    size_t localSize[3];
    localSize[0] = blockSize[0];
    localSize[1] = blockSize[1];
    localSize[2] = blockSize[2];

    cl_int err;
    //fprintf(stderr, "[HI_kernel_call()] GRIDSIZE %d %d %d\n", globalSize[2], globalSize[1], globalSize[0]);
    if(async != DEFAULT_QUEUE) {
        cl_command_queue queue = getQueue(async);
        cl_event *event = getEvent(async);
        err = clEnqueueNDRangeKernel(queue, (cl_kernel)(tconf->kernelsMap.at(this).at(kernel_name)), 3, NULL, globalSize, localSize, 0, NULL, event);
    } else {
        cl_command_queue queue = getQueue(async);
        err = clEnqueueNDRangeKernel(queue, (cl_kernel)(tconf->kernelsMap.at(this).at(kernel_name)), 3, NULL, globalSize, localSize, 0, NULL, NULL);
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_kernel_call()] Kernel Launch FAIL\n");
		exit(1);
        return HI_error;
    }
#ifdef _OPENARC_PROFILE_
    if(tconf->KernelCNTMap.count(kernel_name) == 0) {
        tconf->KernelCNTMap[kernel_name] = 0.0;
    }        
    tconf->KernelCNTMap[kernel_name] += 1;
    if(tconf->KernelTimingMap.count(kernel_name) == 0) {
        tconf->KernelTimingMap[kernel_name] = 0.0;
    }        
    tconf->KernelTimingMap[kernel_name] += HI_get_localtime() - ltime;
#endif   
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_kernel_call(%d)\n", async);
	}
#endif
    return HI_success;
}

HI_error_t OpenCLDriver::HI_synchronize()
{
    cl_int ciErr1;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_synchronize()\n");
	}
#endif
	if( unifiedMemSupported == 1 ) {
		//If unified memory is not used, the default queue will handle necessary
		//synchronization, and thus no need for this explicit synchronization.
/*
    	ciErr1 = clFlush(clQueue);
    	if(ciErr1 != CL_SUCCESS) {
        	printf("Error in clFlush, Line %u in file %s : %d\n\n", __LINE__, __FILE__, ciErr1);
        	return HI_error;
    	}
*/
    	//For OpenCL 1.2
    	//cl_int ciErr1 = clEnqueueBarrierWithWaitList(clQueue, 0, NULL, NULL);
    	//cl_int ciErr1 = clEnqueueBarrier(clQueue);
    	ciErr1 = clFinish(clQueue);
    	if (ciErr1 != CL_SUCCESS)
    	{
        	printf("Error in clFinish, Line %u in file %s : %d\n\n", __LINE__, __FILE__, ciErr1);
#ifdef _OPENARC_PROFILE_
			if( HI_openarcrt_verbosity > 2 ) {
				fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_synchronize()\n");
			}
#endif
        	return HI_error;
		}
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_synchronize()\n");
	}
#endif
    return HI_success;
}


void OpenCLDriver::HI_set_async(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_set_async(%d)\n", asyncId);
	}
#endif
#ifdef _OPENMP
    #pragma omp critical (HI_set_async_critical)
#endif
    {
        cl_int err;
        asyncId += 2;
        std::map<int, cl_command_queue >::iterator it= queueMap.find(asyncId);

        if(it == queueMap.end()) {
            cl_command_queue queue;
            queue = clCreateCommandQueue(clContext, clDevice, 0, &err);
            if(err != CL_SUCCESS) {
                fprintf(stderr, "[ERROR in OpenCLDriver::HI_set_async()] failed to create OPENCL queue with error %d (OPENCL Device)\n", err);
				exit(1);
            }
            queueMap[asyncId] = queue;
        }

        int thread_id = get_thread_id();
        std::map<int, std::map<int, cl_event> >::iterator threadIt;
        threadIt = threadQueueEventMap.find(thread_id);

        //threadQueueEventMap is empty for this thread
        if(threadIt == threadQueueEventMap.end()) {
            std::map<int, cl_event> newMap;
            cl_event ev;
            ev = clCreateUserEvent(clContext, &err);
            if(err != CL_SUCCESS) {
                printf("[ERROR in OpenCLDriver::HI_set_async()] Error in clCreateUserEvent, Line %u in file %s : %d!!!\n\n", __LINE__, __FILE__, err);
				exit(1);
            }
            clSetUserEventStatus(ev, CL_COMPLETE);
            newMap[asyncId] = ev;
            threadQueueEventMap[thread_id] = newMap;
        } else {
            //threadQueueEventMap does not have an entry for this stream
            //std::map<int, cl_event> evMap = threadIt->second;
            if(threadIt->second.find(asyncId) == threadIt->second.end()) {
                cl_event ev;
                ev = clCreateUserEvent(clContext, &err);
                if(err != CL_SUCCESS) {
                    printf("[ERROR in OpenCLDriver::HI_set_async()] Error in clCreateUserEvent, Line %u in file %s : %d!!!\n\n", __LINE__, __FILE__, err);
					exit(1);
                }
                clSetUserEventStatus(ev, CL_COMPLETE);
                threadIt->second[asyncId] = ev;
                //threadIt->second = evMap;
            }
        }
    }
	if( unifiedMemSupported == 0 ) {
		//We need explicit synchronization here since HI_synchronize() does not
		//explicitly synchronize if unified memory is not used.
    	cl_int ciErr1 = clFinish(clQueue);
    	if (ciErr1 != CL_SUCCESS)
    	{
        	printf("Error in clFinish, Line %u in file %s : %d\n\n", __LINE__, __FILE__, ciErr1);
			exit(1);
		}
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_set_async(%d)\n", asyncId-2);
	}
#endif
}

void OpenCLDriver::HI_wait(int arg) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_wait(%d)\n", arg);
	}
#endif
    cl_event *event = getEvent(arg);
    cl_int err ;
    HostConf_t * tconf = getHostConf();
    //clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(err), &err, NULL);
    //fprintf(stderr, "[OpenCLDriver::HI_wait()] status is %d (NVIDIA CUDA GPU)\n", err);

    err = clWaitForEvents(1, event);

    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_wait()] failed wait on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", arg, err);
		exit(1);
    }

	HI_postponed_free(arg);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_wait(%d)\n", arg);
	}
#endif
}

void OpenCLDriver::HI_wait_async(int arg, int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_wait_async(%d, %d)\n", arg, async);
	}
#endif
    cl_event *event = getEvent(arg);
    cl_event *event2 = getEvent(async);
    cl_int err ;
    HostConf_t * tconf = getHostConf();
    //clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(err), &err, NULL);
    //fprintf(stderr, "[OpenCLDriver::HI_wait_async()] status is %d (NVIDIA CUDA GPU)\n", err);

    err = clWaitForEvents(1, event);

    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_wait_async()] failed wait on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", arg, err);
		exit(1);
    }

	HI_postponed_free(arg);

    err = clWaitForEvents(1, event2);

    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_wait_async()] failed wait on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", async, err);
		exit(1);
    }

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_wait_async(%d, %d)\n", arg, async);
	}
#endif
}

void OpenCLDriver::HI_waitS1(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_waitS1(%d)\n", asyncId);
	}
#endif
    cl_event *event = getEvent(asyncId);
    cl_int err ;
    HostConf_t * tconf = getHostConf();
    //clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(err), &err, NULL);
    //fprintf(stderr, "[OpenCLDriver::HI_wait()] status is %d (NVIDIA CUDA GPU)\n", err);

    err = clWaitForEvents(1, event);

    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_wait()] failed wait on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", asyncId, err);
		exit(1);
    }

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_waitS1(%d)\n", asyncId);
	}
#endif
}

void OpenCLDriver::HI_waitS2(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_waitS2(%d)\n", asyncId);
	}
#endif
	HI_free_temphosts(asyncId);
	HI_postponed_free(asyncId);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_waitS2(%d)\n", asyncId);
	}
#endif
}

void OpenCLDriver::HI_wait_all() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_wait_all()\n");
	}
#endif
    eventmap_opencl_t *eventMap = &threadQueueEventMap.at(get_thread_id());
    cl_int err;
    HostConf_t * tconf = getHostConf();

    for(eventmap_opencl_t::iterator it = eventMap->begin(); it != eventMap->end(); ++it) {
        //clGetEventInfo((it->second), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(err), &err, NULL);
        //fprintf(stderr, "[OpenCLDriver::HI_wait_all()] status is %d on queue %d (NVIDIA CUDA GPU)\n", err, it->first);
        err = clWaitForEvents(1, &(it->second));
        if(err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR in OpenCLDriver::HI_wait_all()] failed wait on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", it->first, err);
			exit(1);
        }
    }

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_wait_all()\n");
	}
#endif
}

void OpenCLDriver::HI_wait_all_async(int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_wait_all_async(%d)\n", async);
	}
#endif
    eventmap_opencl_t *eventMap = &threadQueueEventMap.at(get_thread_id());
    cl_int err;
    HostConf_t * tconf = getHostConf();

    for(eventmap_opencl_t::iterator it = eventMap->begin(); it != eventMap->end(); ++it) {
        //clGetEventInfo((it->second), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(err), &err, NULL);
        //fprintf(stderr, "[OpenCLDriver::HI_wait_all_async()] status is %d on queue %d (NVIDIA CUDA GPU)\n", err, it->first);
        err = clWaitForEvents(1, &(it->second));
        if(err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR in OpenCLDriver::HI_wait_all_async()] failed wait on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", it->first, err);
			exit(1);
        }
    }

    cl_event *event2 = getEvent(async);
    err = clWaitForEvents(1, event2);

    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_wait_all_async()] failed wait on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", async, err);
		exit(1);
    }

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_wait_all_async(%d)\n", async);
	}
#endif
}


int OpenCLDriver::HI_async_test(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_async_test(%d)\n", asyncId);
	}
#endif
    cl_event *event = getEvent(asyncId);
    cl_int err, status ;
    HostConf_t * tconf = getHostConf();
    //fprintf(stderr, "[OpenCLDriver::HI_wait()] status is %d (NVIDIA CUDA GPU)\n", err);

    err = clGetEventInfo(*event,  CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
    if(err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_async_test()] failed test on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", asyncId, err);
		exit(1);
    }

    if(status != CL_COMPLETE) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_async_test(%d)\n", asyncId);
	}
#endif
        return 0;
    }
    HI_postponed_free(asyncId);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_async_test(%d)\n", asyncId);
	}
#endif
    return 1;
}

int OpenCLDriver::HI_async_test_all() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_async_test_all()\n");
	}
#endif
    eventmap_opencl_t *eventMap = &threadQueueEventMap.at(get_thread_id());
    cl_int err, status;
    HostConf_t * tconf = getHostConf();

    std::set<int> queuesChecked;

    for(eventmap_opencl_t::iterator it = eventMap->begin(); it != eventMap->end(); ++it) {
        //fprintf(stderr, "[OpenCLDriver::HI_wait_all()] status is %d on queue %d (NVIDIA CUDA GPU)\n", err, it->first);
        err = clGetEventInfo(it->second,  CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
        if(err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR in OpenCLDriver::HI_async_test_all()] failed test on OpenCL queue %d with error %d (NVIDIA CUDA GPU)\n", it->first, err);
			exit(1);
        }
        if(status != CL_COMPLETE) {
            return 0;
        }
        queuesChecked.insert(it->first);
    }

    //release the waiting frees
    std::set<int>::iterator it;
    for (it=queuesChecked.begin(); it!=queuesChecked.end(); ++it) {
        HI_postponed_free(*it);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_async_test_all()\n");
	}
#endif
    return 1;
}

void OpenCLDriver::HI_malloc(void **devPtr, size_t size) {
    cl_int  err;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_malloc()\n");
	}
#endif
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    *devPtr = (void*) clCreateBuffer(clContext, CL_MEM_READ_WRITE, (size_t) size, NULL, &err);
    if( err != CL_SUCCESS ) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_malloc()] :failed to malloc on OpenCL with error %d (NVIDIA CUDA GPU)\n", err);
		exit(1);
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_malloc()\n");
	}
#endif

}

void OpenCLDriver::HI_free(void *devPtr) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tenter OpenCLDriver::HI_free()\n");
	}
#endif
    cl_int err = CL_SUCCESS;
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
	void *devPtr2;
    if( (HI_get_device_address(devPtr, &devPtr2, DEFAULT_QUEUE) == HI_error) ||
        (devPtr != devPtr2) ) {
        //Free device memory if it is not on unified memory.
    	err = clReleaseMemObject((cl_mem)devPtr);
	}
    if( err != CL_SUCCESS ) {
        fprintf(stderr, "[ERROR in OpenCLDriver::HI_free()] :failed to free on OpenCL with error %d (NVIDIA CUDA GPU)\n", err);
		exit(1);
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\texit OpenCLDriver::HI_free()\n");
	}
#endif
}

