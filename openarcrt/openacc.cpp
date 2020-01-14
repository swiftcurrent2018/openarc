#include <stdio.h>
#include <string.h>
#include <cstring>
#include "openacc.h"
#include "openaccrt_ext.h"

static const char *openarcrt_verbosity_env = "OPENARCRT_VERBOSITY";

int get_thread_id() {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    return thread_id;
}

int acc_get_num_devices( acc_device_t devtype, int threadID ) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_num_devices(thread ID = %d)\n", threadID);
	}
#endif
    HostConf *tconf = getHostConf(threadID);
    int count;
    tconf->setTranslationType();

    if( HI_hostinit_done == 0 ) {
        HI_hostinit(tconf->threadID);
    }
    if( (devtype == acc_device_not_host) || (devtype == acc_device_default) ) {
		tconf->setDefaultDevice();
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0 && OPENARC_ARCH != 5
        count = OpenCLDriver::HI_get_num_devices(tconf->acc_device_type_var);
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 5
        count = HipDriver::HI_get_num_devices(tconf->acc_device_type_var);
#else
		count = CudaDriver::HI_get_num_devices(tconf->acc_device_type_var);
#endif
    } else if( (devtype == acc_device_nvidia) || (devtype == acc_device_radeon)) {
        devtype = acc_device_gpu;
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0 && OPENARC_ARCH != 5
        count = OpenCLDriver::HI_get_num_devices(devtype);
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 5
        count = HipDriver::HI_get_num_devices(devtype);
#else
		count = CudaDriver::HI_get_num_devices(devtype);
#endif
    } else if( devtype == acc_device_host ) {
        //count = 1;
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0 && OPENARC_ARCH != 5
        count = OpenCLDriver::HI_get_num_devices(devtype);
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 5
        count = HipDriver::HI_get_num_devices(devtype);
#else
		count = CudaDriver::HI_get_num_devices(devtype);
#endif
    } else if( devtype == acc_device_xeonphi ) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH == 2
        count = OpenCLDriver::HI_get_num_devices(devtype);
#else
        count = 0;
#endif
    } else if( devtype == acc_device_altera ) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH == 3
        count = OpenCLDriver::HI_get_num_devices(devtype);
#else
        count = 0;
#endif
    } else {
        count = 0;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_num_devices(thread ID = %d)\n", threadID);
	}
#endif
    return count;
}

int acc_get_num_devices( acc_device_t devtype ) {
	return acc_get_num_devices(devtype, NO_THREAD_ID);
}

//This function also initializes a device indirectly, by calling
//acc_set_device_num() internally.
void acc_set_device_type( acc_device_t devtype, int threadID ) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_set_device_type(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);
    tconf->user_set_device_type_var =  devtype;
    if( devtype == acc_device_nvidia || devtype == acc_device_radeon || devtype == acc_device_gpu ) {
        tconf->acc_device_type_var = acc_device_gpu;
        tconf->isOnAccDevice = 1;
    } else if ( (devtype == acc_device_not_host) || (devtype == acc_device_default) ) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH == 3
        tconf->acc_device_type_var = acc_device_altera;
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 2
        tconf->acc_device_type_var = acc_device_xeonphi;
#else
        tconf->acc_device_type_var = acc_device_gpu;
#endif
        tconf->isOnAccDevice = 1;
    } else if ( devtype == acc_device_altera ) {
        tconf->acc_device_type_var = acc_device_altera;
        tconf->isOnAccDevice = 1;
    } else if ( devtype == acc_device_xeonphi ) {
        tconf->acc_device_type_var = acc_device_xeonphi;
        tconf->isOnAccDevice = 1;
    } else if ( devtype == acc_device_host ) {
        tconf->acc_device_type_var = acc_device_host;
        tconf->isOnAccDevice = 1;
    } else if ( devtype == acc_device_none ) {
        tconf->acc_device_type_var = devtype;
        tconf->isOnAccDevice = 0;
    } else if ( devtype == acc_device_current ) {
        tconf->user_set_device_type_var = tconf->acc_device_type_var;
		if( tconf->acc_device_type_var == acc_device_host ) {
        	tconf->isOnAccDevice = 1;
		} else if( tconf->acc_device_type_var == acc_device_none ) {
        	tconf->isOnAccDevice = 0;
		} else {
        	tconf->isOnAccDevice = 1;
		}
    } else {
        fprintf(stderr, "[ERROR in acc_set_device_type()] Not supported device type %d; exit!\n", devtype);
        exit(1);
    }

    tconf->setDefaultDevNum();
    acc_set_device_num(tconf->acc_device_num_var, tconf->user_set_device_type_var, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_set_device_type(devtype = %d, threadID = %d)\n", devtype, threadID);
	}
#endif
}

void acc_set_device_type( acc_device_t devtype ) {
	acc_set_device_type(devtype, NO_THREAD_ID);
}

acc_device_t acc_get_device_type(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_device_type(thread ID = %d)\n", threadID);
	}
#endif
	acc_device_t return_data;
    HostConf_t * tconf = getHostConf(threadID);
	if( tconf->acc_device_type_var == acc_device_none ) {
        tconf->setDefaultDevice();
	}
    return_data = tconf->user_set_device_type_var;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_device_type(thread ID = %d)\n", threadID);
	}
#endif
	return return_data;
}

acc_device_t acc_get_device_type() {
	return acc_get_device_type(NO_THREAD_ID);
}

//If the value of devicenum is negative, the runtime will revert to its
//default behavior, which is implementation-defined. If the value
//of the second argument is zero, the selected device number will
//be used for all attached accelerator types.
//The function is the only place where actual device initialization occurs
//by calling tconf->device->init().
void acc_set_device_num( int devnum, acc_device_t devtype, int threadID ) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_set_device_num(devnum = %d, devtype = %d, thread ID = %d)\n", devnum, devtype, threadID);
	}
#endif
	if( devnum < 0 ) {
		devnum = 0;
	}
    HostConf_t * tconf = setNGetHostConf(devnum, threadID);
    tconf->user_set_device_type_var = devtype;
    tconf->acc_device_num_var = devnum;
    if( devtype == acc_device_nvidia ||  devtype == acc_device_radeon ||  devtype == acc_device_gpu ) {
        devtype = acc_device_gpu;
        tconf->acc_device_type_var = acc_device_gpu;
    } else if( (devtype == acc_device_xeonphi) || (devtype == acc_device_altera) ) {
        tconf->acc_device_type_var = devtype;
    } else if (devtype == acc_device_not_host) {
        tconf->setDefaultDevice(); 
		//setDefaultDevice() will change user_set_device_type_var.
    	tconf->user_set_device_type_var = devtype;
		devtype = tconf->acc_device_type_var;
    } else if (devtype == acc_device_default) {
        tconf->setDefaultDevice();
		//setDefaultDevice() will change user_set_device_type_var.
		devtype = tconf->acc_device_type_var;
    } else if (devtype == acc_device_current) {
		devtype = tconf->acc_device_type_var;
    } else if( devtype == acc_device_host ) {
        tconf->acc_device_type_var = acc_device_host;
    } else {
        fprintf(stderr, "[ERROR in acc_set_device_num()] Not supported device type %d; exit!\n", devtype);
        exit(1);
    }

	if( HostConf::devMap.count(devtype) == 0 ) {
		//Current implementation allows only one type of devices per program, and thus changing device type
		//from the configured accelerator type to another is NOT allowed, except between the accelerator and
		//the host. However, the host is not supported yet.
		//Below call may cause infinite recursion!
    	//tconf->HI_init(devnum);
		if( devtype == acc_device_host ) {
			int numDevs = 1;
			tconf->acc_num_devices = numDevs;
			if( numDevs <= devnum ) {
				fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
				exit(1);
			}
        	tconf->device = NULL;
		} else {
        	fprintf(stderr, "[ERROR in acc_set_device_num()] the current configuration does not support device type %d; exit!\n", devtype);
        	exit(1);
		}
	}

	if( HostConf::devMap.count(devtype) > 0 ) {
		int numDevs = HostConf::devMap.at(devtype).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
#ifdef _OPENARC_PROFILE_
			fprintf(stderr, "Host Thread %d uses device %d of type %d\n",threadID, devnum, devtype);
#endif
		}
        tconf->device = HostConf::devMap.at(devtype).at(devnum);
        //printf("devType %d\n",devtype );
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_set_device_num);
#else
#ifdef _OPENMP
		#pragma omp critical(acc_set_device_num_critical)
#endif
#endif
		{
        	if(tconf->device->init_done != 1) {
            	tconf->device->init(tconf->threadID);
        	} else {
				tconf->device->updateKernelNameSet(tconf->kernelnames);
            	tconf->device->createKernelArgMap(tconf->threadID);
        	}
		}
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_set_device_num);
#endif
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_set_device_num(devnum = %d, devtype = %d, thread ID = %d)\n", devnum, devtype, threadID);
	}
#endif
}

void acc_set_device_num( int devnum, acc_device_t devtype) {
	acc_set_device_num(devnum, devtype, NO_THREAD_ID);
}

int acc_get_device_num( acc_device_t devtype, int threadID ) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_device_num(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
	int return_data;
    HostConf_t * tconf = getHostConf(threadID);
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon) || 
			(devtype == acc_device_gpu) || (devtype == acc_device_xeonphi) || 
			(devtype == acc_device_altera) ) {
        return_data = tconf->acc_device_num_var;
    } else if( devtype == acc_device_host ) {
        return_data = tconf->acc_device_num_var;
    } else if( devtype == acc_device_current ) {
        return_data = tconf->acc_device_num_var;
    } else {
        fprintf(stderr, "[ERROR in acc_get_device_num()] Not supported device type %d; exit!\n", devtype);
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_device_num(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
	return return_data;
}

int acc_get_device_num( acc_device_t devtype) {
	return acc_get_device_num(devtype, NO_THREAD_ID);
}

int acc_async_test( int asyncID, int threadID ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_async_test(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	int return_data = 1;
    HostConf_t * tconf = getHostConf(threadID);
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_async_test()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    //return_data = tconf->device->HI_async_test(asyncID+tconf->asyncID_offset);
    HostConf_t * ttconf;
	for(std::map<int, HostConf_t *>::iterator it = hostConfMap.begin(); it != hostConfMap.end(); ++it) {
		ttconf = it->second;
		if(ttconf->device != NULL) {
			if( ttconf->device->HI_async_test_ifpresent(asyncID+ttconf->asyncID_offset, ttconf->threadID) == 0 ) {
				return_data = 0;
				break;
			}
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_async_test(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	return return_data;
}

int acc_async_test( int asyncID) {
	return acc_async_test(asyncID, NO_THREAD_ID);
}

int acc_async_test_all(int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_async_test_all(thread ID = %d)\n", threadID);
	}
#endif
	int return_data;
    HostConf_t * tconf = getHostConf(threadID);
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_async_test_all()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in acc_async_test_all()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_data = tconf->device->HI_async_test_all(tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_async_test_all(thread ID = %d)\n", threadID);
	}
#endif
	return return_data;
}

int acc_async_test_all() {
	return acc_async_test_all(NO_THREAD_ID);
}

//[DEBUG] What if arg value is acc_async_noval?
void acc_wait( int arg, int threadID ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_wait(arg = %d, thread ID = %d)\n", arg, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_wait()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    //tconf->device->HI_wait(arg+tconf->asyncID_offset, tconf->threadID);
    HostConf_t * ttconf;
	for(std::map<int, HostConf_t *>::iterator it = hostConfMap.begin(); it != hostConfMap.end(); ++it) {
		ttconf = it->second;
		if(ttconf->device != NULL) {
			ttconf->device->HI_wait_ifpresent(arg+ttconf->asyncID_offset, ttconf->threadID);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_wait(arg = %d, thread ID = %d)\n", arg, threadID);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

void acc_wait( int arg) {
	acc_wait(arg, NO_THREAD_ID);
}

//[DEBUG] acc_async_wait() is renamed to acc_wait().
void acc_async_wait( int asyncID, int threadID ) {
	acc_wait(asyncID, threadID);
}

void acc_async_wait( int asyncID) {
	acc_async_wait( asyncID, NO_THREAD_ID);
}

void acc_wait_all(int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_wait_all(thread ID = %d)\n", threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_wait_all()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in acc_wait_all()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_wait_all(tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_wait_all(thread ID = %d)\n", threadID);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

void acc_wait_all() {
	acc_wait_all(NO_THREAD_ID);
}

//[DEBUG] acc_async_wait_all() is renamed to acc_wait_all().
void acc_async_wait_all(int threadID) {
	acc_wait_all(threadID);
}

void acc_async_wait_all() {
	acc_async_wait_all(NO_THREAD_ID);
}

void acc_init( acc_device_t devtype, int kernels, std::string kernelNames[], const char *fileNameBase, int threadID ) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_init(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
    HostConf_t * tconf = getInitHostConf(threadID);
	tconf->baseFileName = fileNameBase;
    //Set device type.
    if( (devtype == acc_device_default) || (devtype == acc_device_not_host) ) {
		tconf->setDefaultDevice();
	} else {
		//[CAUTION] device type should be set consistently with 
		//setDefaultDevice().
    	tconf->user_set_device_type_var =  devtype;
		if( (devtype == acc_device_nvidia) || (devtype == acc_device_radeon) ) {
        	tconf->acc_device_type_var = acc_device_gpu;
		} else {
        	tconf->acc_device_type_var = devtype;
		}
    }
    //tconf->HostConf::HI_specify_kernel_names(kernels, kernelNames, 0);
	if( tconf->HI_init_done == 0 ) {
    	tconf->initKernelNames(kernels, kernelNames);
		tconf->HI_init_done = 1;
    	tconf->HI_init(DEVICE_NUM_UNDEFINED);
	} else {
	//} else if( tconf->HI_kernels_registered == 0 ) {
		//acc_init() can be called multiple times.
        //printf("[%s:%d]\n", __FILE__, __LINE__);
    	tconf->addKernelNames(kernels, kernelNames);
		tconf->HI_kernels_registered = 1;
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in acc_init()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
		tconf->device->HI_register_kernels(tconf->kernelnames, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_init(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
}

void acc_init( acc_device_t devtype, int kernels, std::string kernelNames[], const char *fileNameBase) {
	acc_init( devtype, kernels, kernelNames, fileNameBase, NO_THREAD_ID);
}

void acc_init( acc_device_t devtype, int threadID ) {
#ifdef _OPENARC_PROFILE_
    if( HI_hostinit_done == 0 ) {
		int openarcrt_verbosity;
		char * envVar;
		envVar = getenv(openarcrt_verbosity_env);
		if( envVar != NULL ) {
			openarcrt_verbosity = atoi(envVar);
			if( openarcrt_verbosity > 0 ) {
				HI_openarcrt_verbosity = openarcrt_verbosity;
			}    
		}    
	}
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_init(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
    HostConf_t * tconf = getInitHostConf(threadID);
    //Set device type.
    if( (devtype == acc_device_default) || (devtype == acc_device_not_host) ) {
		tconf->setDefaultDevice();
	} else {
		//[CAUTION] device type should be set consistently with 
		//setDefaultDevice().
    	tconf->user_set_device_type_var =  devtype;
		if( (devtype == acc_device_nvidia) || (devtype == acc_device_radeon) ) {
        	tconf->acc_device_type_var = acc_device_gpu;
		} else {
        	tconf->acc_device_type_var = devtype;
		}
    }
    //tconf->HostConf::HI_specify_kernel_names(kernels, kernelNames, 0);
    //tconf->initKernelNames(kernels, kernelNames);
	if( tconf->HI_init_done == 0 ) {
		tconf->HI_init_done = 1;
    	tconf->HI_init(DEVICE_NUM_UNDEFINED);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_init(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
}

void acc_init( acc_device_t devtype) {
	acc_init(devtype, NO_THREAD_ID);
}

void acc_shutdown( acc_device_t devtype, int threadID ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_shutdown(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);
    if( (tconf == NULL) || (tconf->device == NULL) ) {
        return;
    }
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon) 
            || (devtype == acc_device_xeonphi) || (devtype == acc_device_gpu) 
			|| (devtype == acc_device_altera) || (devtype == acc_device_host) ) {
        if( tconf->device->init_done == 1 ) {
            fflush(stdout);
            fflush(stderr);
            tconf->isOnAccDevice = 0;

            //[DEBUG] below statements are moved into HI_reset()
            //fprintf(stderr, "[in acc_shutdown()] about to destroy!\n");
            //tconf->device->masterAddressTable.clear();
            //tconf->device->postponedFreeTable.clear();
            //tconf->device->destroy(tconf->threadID);
            //fprintf(stderr, "[in acc_shutdown()] destroy done!\n");
            //tconf->device->init_done = 0;
            //fprintf(stderr, "[in acc_shutdown()] about to reset\n");
            tconf->HI_reset();
            //fprintf(stderr, "[in acc_shutdown()] reset done!\n");
        }
    }
	tconf->HI_init_done = 0;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_shutdown(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
}

void acc_shutdown( acc_device_t devtype) {
	acc_shutdown(devtype, NO_THREAD_ID);
}

//DEBUG: below implementation can be called only by host threads.
//Call to this function within a GPU kernel should be overwritten
//by OpenACC-to-Device translator.
int acc_on_device( acc_device_t devtype, int threadID ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_on_device(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    //TODO:
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon)
            || (devtype == acc_device_xeonphi) || (devtype == acc_device_altera) 
			|| (devtype == acc_device_gpu) ) {
        //return tconf->isOnAccDevice;
        return 0;
    } else if( devtype == acc_device_host ) {
        //return tconf->isOnAccDevice == 0 ? 1 : 0;
        return 1;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_on_device(devtype = %d, thread ID = %d)\n", devtype, threadID);
	}
#endif
    return 0;
}

int acc_on_device( acc_device_t devtype) {
	return acc_on_device(devtype, NO_THREAD_ID);
}

d_void* acc_malloc(size_t size, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_malloc(size = %ld, thread ID = %d)\n",size, threadID);
	}
#endif
	if( size == 0 ) {
    	fprintf(stderr, "[ERROR in acc_malloc()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
    void *devPtr;
    if( tconf->isOnAccDevice ) {
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in acc_malloc()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
        tconf->device->HI_malloc(&devPtr, size, HI_MEM_READ_WRITE, tconf->threadID);
    } else {
        fprintf(stderr, "[ERROR in acc_malloc()] target accelerator device has not been set; exit!\n");
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	tconf->DMallocCnt++;
	tconf->DMallocSize += size;
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_malloc(size = %ld, thread ID = %d)\n",size, threadID);
	}
#endif
    return devPtr;
}

d_void* acc_malloc(size_t size) {
	return acc_malloc(size, NO_THREAD_ID);
}

void acc_free(void* devPtr, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_free(thread ID = %d)\n", threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice ) {
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in acc_free()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
        tconf->device->HI_free(devPtr, tconf->threadID);
    } else {
        fprintf(stderr, "[ERROR in acc_free()] target accelerator device has not been set; exit!\n");
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	tconf->DFreeCnt++;
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_free(thread ID = %d)\n", threadID);
	}
#endif
}

void acc_free(void* devPtr) {
	acc_free(devPtr, NO_THREAD_ID);
}

/////////////////////////////////////////////////////////// 
// OpenACC Runtime Library Routines added in Version 2.0 //
/////////////////////////////////////////////////////////// 
void acc_wait_async(int arg, int async, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_wait_async(arg = %d, async = %d, thread ID = %d)\n", arg, async, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_wait_async()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    //tconf->device->HI_wait_async(arg+tconf->asyncID_offset, async+tconf->asyncID_offset, tconf->threadID);
    HostConf_t * ttconf;
	for(std::map<int, HostConf_t *>::iterator it = hostConfMap.begin(); it != hostConfMap.end(); ++it) {
		ttconf = it->second;
		if(ttconf->device != NULL) {
			ttconf->device->HI_wait_async_ifpresent(arg+ttconf->asyncID_offset, async+ttconf->asyncID_offset, tconf->threadID);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_wait_async(arg = %d, async = %d, thread ID = %d)\n", arg, async, threadID);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

void acc_wait_async(int arg, int async) {
	acc_wait_async(arg, async, NO_THREAD_ID);
}

void acc_wait_all_async(int async, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_wait_all_async(async = %d, thread ID = %d)\n", async, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_wait_all_async()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in acc_wait_all_async()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_wait_all_async(async+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_wait_all_async(async = %d, thread ID = %d)\n", async, threadID);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

void acc_wait_all_async(int async) {
	acc_wait_all_async(async, NO_THREAD_ID);
}

void* acc_copyin(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin(size = %ld, thread ID = %d)\n", size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
	HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin(size = %ld, thread ID = %d)\n", size, threadID);
	}
#endif
	return devPtr;
}

void* acc_copyin(h_void* hostPtr, size_t size) {
	return acc_copyin(hostPtr, size, NO_THREAD_ID);
}

void* acc_pcopyin(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcopyin(size = %ld, thread ID = %d)\n", size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcopyin(size = %ld, thread ID = %d)\n", size, threadID);
	}
#endif
	return devPtr;
}

void* acc_pcopyin(h_void* hostPtr, size_t size) {
	return acc_pcopyin(hostPtr, size, NO_THREAD_ID);
}

void* acc_present_or_copyin(h_void* hostPtr, size_t size, int threadID) {
	return acc_pcopyin(hostPtr, size, threadID);
}

void* acc_present_or_copyin(h_void* hostPtr, size_t size) {
	return acc_present_or_copyin(hostPtr, size, NO_THREAD_ID);
}

void* acc_create(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
	return devPtr;
}

void* acc_create(h_void* hostPtr, size_t size) {
	return acc_create(hostPtr, size, NO_THREAD_ID);
}

void* acc_pcreate(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcreate(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcreate(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
	return devPtr;
}

void* acc_pcreate(h_void* hostPtr, size_t size) {
	return acc_pcreate(hostPtr, size, NO_THREAD_ID);
}

void* acc_present_or_create(h_void* hostPtr, size_t size, int threadID) {
	return acc_pcreate(hostPtr, size, threadID);
}

void* acc_present_or_create(h_void* hostPtr, size_t size) {
	return acc_present_or_create(hostPtr, size, NO_THREAD_ID);
}

//[FIXME] A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.
void acc_copyout(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyout(size = %ld, thread ID = %d)\n", size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_copyout() is not present on the device; exit!\n");
		exit(1);
	} else {
		HI_memcpy(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, threadID);
		HI_free(hostPtr, DEFAULT_QUEUE, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyout(size = %ld, thread ID = %d)\n", size, threadID);
	}
#endif
}

void acc_copyout(h_void* hostPtr, size_t size) {
	acc_copyout(hostPtr, size, NO_THREAD_ID);
}

//[FIXME] - A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.
//        - The current implementation does not consider size parameter;
//        free the whole data.
void acc_delete(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_delete(size = %ld, thread ID = %d)\n", size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_delete() is not present on the device; exit!\n");
		exit(1);
	} else {
		HI_free(hostPtr, DEFAULT_QUEUE, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_delete(size = %ld, thread ID = %d)\n", size, threadID);
	}
#endif
}

void acc_delete(h_void* hostPtr, size_t size) {
	acc_delete(hostPtr, size, NO_THREAD_ID);
}

void acc_update_device(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_update_device(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)==HI_success)) {
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
	} else {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_update_device(hostPtr = %lx, size = %ld, thread ID = %d) is not present on the device; exit!\n",(long unsigned int)hostPtr, size, threadID);
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_update_device(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
}

void acc_update_device(h_void* hostPtr, size_t size) {
	acc_update_device(hostPtr, size, NO_THREAD_ID);
}

void acc_update_self(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_update_self(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)==HI_success)) {
		HI_memcpy(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, threadID);
	} else {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_update_self(hostPtr = %lx, size = %ld, thread ID = %d) is not present on the device; exit!\n",(long unsigned int)hostPtr, size, threadID);
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_update_self(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
}

void acc_update_self(h_void* hostPtr, size_t size) {
	acc_update_self(hostPtr, size, NO_THREAD_ID);
}

void acc_map_data(h_void* hostPtr, d_void* devPtr, size_t size, int threadID) {
	void* tDevPtr;
	void* tHostPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_map_data(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
    //HostConf_t * tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &tDevPtr, DEFAULT_QUEUE, threadID)==HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] It is an error to call acc_map_data for host data that is already present on the device; exit!\n");
		exit(1);
	} else if ( HI_get_host_address((const void *)devPtr, &tHostPtr, DEFAULT_QUEUE, threadID)==HI_success ) {
		fprintf(stderr, "[OPENARCRT-ERROR] It is undefined to call acc_map_data with a device address that is already mapped to host data; exit!\n");
		exit(1);
	} else {
		HI_set_device_address((const void *)hostPtr, devPtr, size, DEFAULT_QUEUE, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_map_data(hostPtr = %lx, size = %ld, thread ID = %d)\n",(long unsigned int)hostPtr, size, threadID);
	}
#endif
}

void acc_map_data(h_void* hostPtr, d_void* devPtr, size_t size) {
	acc_map_data(hostPtr, devPtr, size, NO_THREAD_ID);
}

//[FIXME] It is undefined behavior to call acc_unmap_data with a host address
//unless that host address was mapped to device memory using acc_map_data, but
//this implementation does not check this.
void acc_unmap_data(h_void* hostPtr, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_unmap_data(hostPtr = %lx, thread ID = %d)\n",(long unsigned int)hostPtr, threadID);
	}
#endif
    HostConf *tconf = getHostConf(threadID);
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_unmap_data()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
	HI_remove_device_address((const void *)hostPtr, DEFAULT_QUEUE, tconf->threadID);
	//Unpin host data if they were implicitly pinned for asynchronous transfers.
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in acc_unmap_data()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_unpin_host_memory((const void *)hostPtr, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_unmap_data(hostPtr = %lx, thread ID = %d)\n",(long unsigned int)hostPtr, threadID);
	}
#endif
}

void acc_unmap_data(h_void* hostPtr) {
	acc_unmap_data(hostPtr, NO_THREAD_ID);
}

d_void* acc_deviceptr(h_void* hostPtr, int threadID) {
	void* devPtr;
	HI_error_t result;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_deviceptr(thread ID = %d)\n", threadID);
	}
#endif
    //HostConf *tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	result = HI_get_device_address((const void*)hostPtr, &devPtr, DEFAULT_QUEUE, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_deviceptr(thread ID = %d)\n", threadID);
	}
#endif
	if (result==HI_success) {
		return (d_void *)(devPtr);
	} else {
		return NULL;
	}
}

d_void* acc_deviceptr(h_void* hostPtr) {
	return acc_deviceptr(hostPtr, NO_THREAD_ID);
}

h_void* acc_hostptr(d_void* devPtr, int threadID) {
	void* hostPtr;
	HI_error_t result;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_hostptr(thread ID = %d)\n", threadID);
	}
#endif
    //HostConf *tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	result = HI_get_host_address((const void*)devPtr, &hostPtr, DEFAULT_QUEUE, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_hostptr(thread ID = %d)\n", threadID);
	}
#endif
	if (result==HI_success) {
		return (h_void *)(hostPtr);
	} else {
		return NULL;
	}
}

h_void* acc_hostptr(d_void* devPtr) {
	return acc_hostptr(devPtr, NO_THREAD_ID);
}


int acc_is_present(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
	int isPresent = 0;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_is_present(hostPtr = %lx, thread ID = %d)\n", (long unsigned int)hostPtr, threadID);
	}
#endif
    //HostConf *tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)==HI_success)) {
		isPresent = 1;
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_is_present(hostPtr = %lx, thread ID = %d)\n", (long unsigned int)hostPtr, threadID);
	}
#endif
	return isPresent;
}

int acc_is_present(h_void* hostPtr, size_t size) {
	return acc_is_present(hostPtr, size, NO_THREAD_ID);
}

void acc_memcpy_to_device(d_void* dest, h_void* src, size_t bytes, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_memcpy_to_device(thread ID = %d)\n", threadID);
	}
#endif
    //HostConf *tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_memcpy(dest, (const void*)src, bytes, HI_MemcpyHostToDevice, 0, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_memcpy_to_device(thread ID = %d)\n", threadID);
	}
#endif
}

void acc_memcpy_to_device(d_void* dest, h_void* src, size_t bytes) {
	acc_memcpy_to_device(dest, src, bytes, NO_THREAD_ID);
}

void acc_memcpy_from_device(h_void* dest, d_void* src, size_t bytes, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_memcpy_from_device(thread ID = %d)\n", threadID);
	}
#endif
    //HostConf *tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_memcpy(dest, (const void*)src, bytes, HI_MemcpyDeviceToHost, 0, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_memcpy_from_device(thread ID = %d)\n", threadID);
	}
#endif
}

void acc_memcpy_from_device(h_void* dest, d_void* src, size_t bytes) {
	acc_memcpy_from_device(dest, src, bytes, NO_THREAD_ID);
}

//////////////////////////////////////////////////////////////////////
// Experimental OpenACC Runtime Library Routines for Unified Memory //
//////////////////////////////////////////////////////////////////////
// If unified memory is supported,
// 		- Allocate unified memory, and copy data from hostPtr
// 		if hostPtr is not NULL.
// Else
// 		- Allocate host memory if hostPtr is NULL.
void* acc_copyin_unified(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin_unified(thread ID = %d)\n", threadID);
	}
#endif
    //HostConf *tconf = getHostConf(threadID);
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_malloc1D_unified(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
	HI_memcpy_unified(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin_unified(thread ID = %d)\n", threadID);
	}
#endif
	return devPtr;
}

void* acc_copyin_unified(h_void* hostPtr, size_t size) {
	return acc_copyin_unified(hostPtr, size, NO_THREAD_ID);
}

// If unified memory is supported,
// 		- If not present, 
// 			- Allocate unified memory, and copy data from hostPtr
// 			if hostPtr is not NULL.
// Else
// 		- Allocate host memory if hostPtr is NULL.
void* acc_pcopyin_unified(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcopyin_unified(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		HI_malloc1D_unified(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
		HI_memcpy_unified(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcopyin_unified(thread ID = %d)\n", threadID);
	}
#endif
	return devPtr;
}

void* acc_pcopyin_unified(h_void* hostPtr, size_t size) {
	return acc_pcopyin_unified(hostPtr, size, NO_THREAD_ID);
}

void* acc_present_or_copyin_unified(h_void* hostPtr, size_t size, int threadID) {
	return acc_pcopyin_unified(hostPtr, size, threadID);
}

void* acc_present_or_copyin_unified(h_void* hostPtr, size_t size) {
	return acc_present_or_copyin_unified(hostPtr, size, NO_THREAD_ID);
}

// If unified memory is supported,
// 		- Allocate unified memory.
// Else
// 		- Allocate host memory if hostPtr is NULL.
void* acc_create_unified(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create_unified(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_malloc1D_unified(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create_unified(thread ID = %d)\n", threadID);
	}
#endif
	return devPtr;
}

void* acc_create_unified(h_void* hostPtr, size_t size) {
	return acc_create_unified(hostPtr, size, NO_THREAD_ID);
}

// If unified memory is supported,
// 		- If not present
// 			- Allocate unified memory.
// Else
// 		- Allocate host memory if hostPtr is NULL.
void* acc_pcreate_unified(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcreate_unified(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		HI_malloc1D_unified(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcreate_unified(thread ID = %d)\n", threadID);
	}
#endif
	return devPtr;
}

void* acc_pcreate_unified(h_void* hostPtr, size_t size) {
	return acc_pcreate_unified(hostPtr, size, NO_THREAD_ID);
}

void* acc_present_or_create_unified(h_void* hostPtr, size_t size, int threadID) {
	return acc_pcreate_unified(hostPtr, size, threadID);
}

void* acc_present_or_create_unified(h_void* hostPtr, size_t size) {
	return acc_present_or_create_unified(hostPtr, size, NO_THREAD_ID);
}

//[FIXME] A call to this routine is NOT allowed within a data region for
//the specified data, but the current runtime does not check this.
//Free the allocated memory (either host or unified memory).
//This has the same effect as calling acc_delete_unified().
void acc_copyout_unified(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyout_unified(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
/*
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		//fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_copyout_unified() is not present on the device; exit!\n");
		//exit(1);
		HI_free_unified(hostPtr, DEFAULT_QUEUE, threadID);
	} else {
		HI_memcpy_unified(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, threadID);
		HI_free_unified(hostPtr, DEFAULT_QUEUE, threadID);
	}
*/
	HI_free_unified(hostPtr, DEFAULT_QUEUE, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyout_unified(thread ID = %d)\n", threadID);
	}
#endif
}

void acc_copyout_unified(h_void* hostPtr, size_t size) {
	acc_copyout_unified(hostPtr, size, NO_THREAD_ID);
}

//[FIXME] - A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.
//        - The current implementation does not consider size parameter;
//        free the whole data.
//Free the allocated memory (either host or unified memory).
void acc_delete_unified(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_delete_unified(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	//if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		//fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_delete_unified() is not present on the device; exit!\n");
		//exit(1);
		//If unified memory is not supported, just return.
		//return; 
	//} else {
		HI_free_unified(hostPtr, DEFAULT_QUEUE, threadID);
	//}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_delete_unified(thread ID = %d)\n", threadID);
	}
#endif
}

void acc_delete_unified(h_void* hostPtr, size_t size) {
	acc_delete_unified(hostPtr, size, NO_THREAD_ID);
}

/////////////////////////////////////////////////////////////////
// Additional OpenACC Runtime Library Routines Used by OpenARC //
/////////////////////////////////////////////////////////////////

void* acc_copyin_const(h_void* hostPtr, size_t size, int  threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin_const(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_ONLY, threadID);
	HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin_const(thread ID = %d)\n", threadID);
	}
#endif
	return devPtr;
}

void* acc_copyin_const(h_void* hostPtr, size_t size) {
	return acc_copyin_const(hostPtr, size, NO_THREAD_ID);
}

void* acc_pcopyin_const(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcopyin_const(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_ONLY, threadID);
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcopyin_const(thread ID = %d)\n", threadID);
	}
#endif
	return devPtr;
}

void* acc_pcopyin_const(h_void* hostPtr, size_t size) {
	return acc_pcopyin_const(hostPtr, size, NO_THREAD_ID);
}

void* acc_present_or_copyin_const(h_void* hostPtr, size_t size, int threadID) {
	return acc_pcopyin_const(hostPtr, size, threadID);
}

void* acc_present_or_copyin_const(h_void* hostPtr, size_t size) {
	return acc_present_or_copyin_const(hostPtr, size, NO_THREAD_ID);
}

void* acc_create_const(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create_const(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_ONLY, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create_const(thread ID = %d)\n", threadID);
	}
#endif
	return devPtr;
}

void* acc_create_const(h_void* hostPtr, size_t size) {
	return acc_create_const(hostPtr, size, NO_THREAD_ID);
}

void* acc_pcreate_const(h_void* hostPtr, size_t size, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcreate_const(thread ID = %d)\n", threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_ONLY, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcreate_const(thread ID = %d)\n", threadID);
	}
#endif
	return devPtr;
}

void* acc_pcreate_const(h_void* hostPtr, size_t size) {
	return acc_pcreate_const(hostPtr, size, NO_THREAD_ID);
}

void* acc_present_or_create_const(h_void* hostPtr, size_t size, int threadID) {
	return acc_pcreate_const(hostPtr, size, threadID);
}

void* acc_present_or_create_const(h_void* hostPtr, size_t size) {
	return acc_present_or_create_const(hostPtr, size, NO_THREAD_ID);
}

void* acc_copyin_async(h_void* hostPtr, size_t size, int async, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin_async(async = %d, thread ID = %d)\n", async, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
	if( async == acc_async_sync ) {
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
	} else {
		HI_memcpy_async(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, async, 0, NULL, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin_async(async = %d, thread ID = %d)\n", async, threadID);
	}
#endif
	return devPtr;
}

void* acc_copyin_async(h_void* hostPtr, size_t size, int async) {
	return acc_copyin_async(hostPtr, size, async, NO_THREAD_ID);
}

void* acc_copyin_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if( arg != acc_async_sync ) {
		acc_wait(arg, threadID);
	}
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
	if( async == acc_async_sync ) {
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
	} else {
		HI_memcpy_async(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, async, 0, NULL, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
	return devPtr;
}

void* acc_copyin_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	return acc_copyin_async_wait(hostPtr, size, async, arg, NO_THREAD_ID);
}

void* acc_pcopyin_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcopyin_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if( arg != acc_async_sync ) {
		acc_wait(arg, threadID);
	}
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
		if( async == acc_async_sync ) {
			HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, threadID);
		} else {
			HI_memcpy_async(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, async, 0, NULL, threadID);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcopyin_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
	return devPtr;
}

void* acc_pcopyin_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	return acc_pcopyin_async_wait(hostPtr, size, async, arg, NO_THREAD_ID);
}

void* acc_present_or_copyin_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID) {
	return acc_pcopyin_async_wait(hostPtr, size, async, arg, threadID);
}

void* acc_present_or_copyin_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	return acc_present_or_copyin_async_wait(hostPtr, size, async, arg, NO_THREAD_ID);
}

void* acc_create_async(h_void* hostPtr, size_t size, int async, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create_async(async = %d, thread ID = %d)\n", async, threadID);
	}
#endif
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create_async(async = %d, thread ID = %d)\n", async, threadID);
	}
#endif
	return devPtr;
}

void* acc_create_async(h_void* hostPtr, size_t size, int async) {
	return acc_create_async(hostPtr, size, async, NO_THREAD_ID);
}

void* acc_create_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if( arg != acc_async_sync ) {
		acc_wait(arg, threadID);
	}
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
	return devPtr;
}

void* acc_create_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	return acc_create_async_wait(hostPtr, size, async, arg, NO_THREAD_ID);
}

void* acc_pcreate_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcreate_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if( arg != acc_async_sync ) {
		acc_wait(arg, threadID);
	}
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE, threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcreate_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
	return devPtr;
}

void* acc_pcreate_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	return acc_pcreate_async_wait(hostPtr, size, async, arg, NO_THREAD_ID);
}

void* acc_present_or_create_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID) {
	return acc_pcreate_async_wait(hostPtr, size, async, arg, threadID);
}

void* acc_present_or_create_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	return acc_present_or_create_async_wait(hostPtr, size, async, arg, NO_THREAD_ID);
}

//[FIXME] A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.

void acc_copyout_async(h_void* hostPtr, size_t size, int async, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyout_async(async = %d, thread ID = %d)\n", async, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_copyout_async() is not present on the device; exit!\n");
		exit(1);
	} else {
		if( async == acc_async_sync ) {
			HI_memcpy(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, threadID);
			HI_free(hostPtr, DEFAULT_QUEUE);
		} else {
			HI_memcpy_async(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, async, 0, NULL, threadID);
			HI_free_async(hostPtr, async, threadID);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyout_async(async = %d, thread ID = %d)\n", async, threadID);
	}
#endif
}

void acc_copyout_async(h_void* hostPtr, size_t size, int async) {
	acc_copyout_async(hostPtr, size, async, NO_THREAD_ID);
}

void acc_copyout_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyout_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if( arg != acc_async_sync ) {
		acc_wait(arg, threadID);
	}
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_copyout_async_wait() is not present on the device; exit!\n");
		exit(1);
	} else {
		if( async == acc_async_sync ) {
			HI_memcpy(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, threadID);
			HI_free(hostPtr, DEFAULT_QUEUE, threadID);
		} else {
			HI_memcpy_async(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, async, 0, NULL, threadID);
			HI_free_async(hostPtr, async, threadID);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyout_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
}

void acc_copyout_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	acc_copyout_async_wait(hostPtr, size, async, arg, NO_THREAD_ID);
}

//[FIXME] - A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.
//        - The current implementation does not consider size parameter;
//        free the whole data.
void acc_delete_async(h_void* hostPtr, size_t size, int async, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_delete_async(async = %d, thread ID = %d)\n", async, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_delete_async() is not present on the device; exit!\n");
		exit(1);
	} else {
		if( async == acc_async_sync ) {
			HI_free(hostPtr, DEFAULT_QUEUE, threadID);
		} else {
			HI_free_async(hostPtr, async, threadID);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_delete_async(async = %d, thread ID = %d)\n", async, threadID);
	}
#endif
}

void acc_delete_async(h_void* hostPtr, size_t size, int async) {
	acc_delete_async(hostPtr, size, async, NO_THREAD_ID);
}

void acc_delete_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_delete_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
    if( threadID == NO_THREAD_ID ) {
        threadID = get_thread_id();
    }   
	if( arg != acc_async_sync ) {
		acc_wait(arg, threadID);
	}
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE, threadID)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_delete_async_wait() is not present on the device; exit!\n");
		exit(1);
	} else {
		if( async == acc_async_sync ) {
			HI_free(hostPtr, DEFAULT_QUEUE, threadID);
		} else {
			HI_free_async(hostPtr, async, threadID);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_delete_async_wait(async = %d, arg = %d, thread ID = %d)\n", async, arg, threadID);
	}
#endif
}
 
void acc_delete_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	acc_delete_async_wait(hostPtr, size, async, arg, NO_THREAD_ID);
}
