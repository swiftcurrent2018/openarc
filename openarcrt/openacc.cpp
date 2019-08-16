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

int acc_get_num_devices( acc_device_t devtype ) {
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
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_num_devices()\n");
	}
#endif
    HostConf *tconf = getHostConf();
    int count;
    tconf->setTranslationType();

    if( HI_hostinit_done == 0 ) {
        HI_hostinit(0);
    }
    if( (devtype == acc_device_not_host) || (devtype == acc_device_default) ) {
		tconf->setDefaultDevice();
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        count = OpenCLDriver::HI_get_num_devices(tconf->acc_device_type_var);
#else
		count = CudaDriver::HI_get_num_devices(tconf->acc_device_type_var);
#endif
    } else if( (devtype == acc_device_nvidia) || (devtype == acc_device_radeon)) {
        devtype = acc_device_gpu;
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        count = OpenCLDriver::HI_get_num_devices(devtype);
#else
		count = CudaDriver::HI_get_num_devices(devtype);
#endif
    } else if( devtype == acc_device_host ) {
        //count = 1;
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        count = OpenCLDriver::HI_get_num_devices(devtype);
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
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_num_devices()\n");
	}
#endif
    return count;
}

//This function also initializes a device indirectly, by calling
//acc_set_device_num() internally.
void acc_set_device_type( acc_device_t devtype ) {
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
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_set_device_type(%d)\n", devtype);
	}
#endif
    HostConf_t * tconf = getHostConf();
    tconf->user_set_device_type_var =  devtype;
    if( devtype == acc_device_nvidia || devtype == acc_device_radeon ) {
        //tconf->user_set_device_type_var = acc_device_gpu;
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
    acc_set_device_num(tconf->acc_device_num_var, tconf->user_set_device_type_var);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_set_device_type(%d)\n", devtype);
	}
#endif
}

acc_device_t acc_get_device_type(void) {
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
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_device_type()\n");
	}
#endif
	acc_device_t return_data;
    HostConf_t * tconf = getHostConf();
	if( tconf->acc_device_type_var == acc_device_none ) {
        tconf->setDefaultDevice();
	}
    return_data = tconf->user_set_device_type_var;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_device_type()\n");
	}
#endif
	return return_data;
}

//If the value of devicenum is negative, the runtime will revert to its
//default behavior, which is implementation-defined. If the value
//of the second argument is zero, the selected device number will
//be used for all attached accelerator types.
//The function is the only place where actual device initialization occurs
//by calling tconf->device->init().
void acc_set_device_num( int devnum, acc_device_t devtype ) {
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
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_set_device_num(%d, %d)\n", devnum, devtype);
	}
#endif
	if( devnum < 0 ) {
		devnum = 0;
	}
    HostConf_t * tconf = setNGetHostConf(devnum);
    tconf->user_set_device_type_var = devtype;
    if( devtype == acc_device_nvidia ||  devtype == acc_device_radeon ||  devtype == acc_device_gpu ) {
        //tconf->acc_device_num_var = devnum;
        devtype = acc_device_gpu;
		int numDevs = HostConf::devMap.at(devtype).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
#ifdef _OPENARC_PROFILE_
			fprintf(stderr, "Host Thread %d uses device %d of type %d\n",get_thread_id(), devnum, devtype);
#endif
		}
        tconf->device = HostConf::devMap.at(devtype).at(devnum);
        tconf->acc_device_type_var = acc_device_gpu;
        tconf->acc_device_num_var = devnum;
        //printf("devType %d\n",devtype );
#ifdef _OPENMP
		#pragma omp critical(acc_set_device_num_critical)
#endif
		{
        	if(tconf->device->init_done != 1) {
            	tconf->device->init();
        	} else {
            	tconf->device->createKernelArgMap();
        	}
		}
    } else if( (devtype == acc_device_xeonphi) || (devtype == acc_device_altera) ) {
		int numDevs = HostConf::devMap.at(devtype).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
#ifdef _OPENARC_PROFILE_
			fprintf(stderr, "Host Thread %d uses device %d of type %d\n",get_thread_id(), devnum, devtype);
#endif
		}
        tconf->device = HostConf::devMap.at(devtype).at(devnum);
        tconf->acc_device_type_var = devtype;
        tconf->acc_device_num_var = devnum;
#ifdef _OPENMP
		#pragma omp critical(acc_set_device_num_critical)
#endif
		{
        	if(tconf->device->init_done != 1) {
            	tconf->device->init();
        	} else {
            	tconf->device->createKernelArgMap();
        	}
		}
    } else if (devtype == acc_device_not_host) {
        tconf->setDefaultDevice();
    	tconf->user_set_device_type_var = devtype;
		int numDevs = HostConf::devMap.at(tconf->acc_device_type_var).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
#ifdef _OPENARC_PROFILE_
			fprintf(stderr, "Host Thread %d uses device %d of type %d\n",get_thread_id(), devnum, devtype);
#endif
		}
        tconf->device = HostConf::devMap.at(tconf->acc_device_type_var).at(devnum);
        tconf->acc_device_num_var = devnum;
#ifdef _OPENMP
		#pragma omp critical(acc_set_device_num_critical)
#endif
		{
        	if(tconf->device->init_done != 1) {
            	tconf->device->init();
        	} else {
            	tconf->device->createKernelArgMap();
        	}
		}
    } else if (devtype == acc_device_default) {
        tconf->setDefaultDevice();
		int numDevs = HostConf::devMap.at(tconf->acc_device_type_var).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
#ifdef _OPENARC_PROFILE_
			fprintf(stderr, "Host Thread %d uses device %d of type %d\n",get_thread_id(), devnum, devtype);
#endif
		}
        tconf->device = HostConf::devMap.at(tconf->acc_device_type_var).at(devnum);
        tconf->acc_device_num_var = devnum;
#ifdef _OPENMP
		#pragma omp critical(acc_set_device_num_critical)
#endif
		{
        	if(tconf->device->init_done != 1) {
            	tconf->device->init();
        	} else {
            	tconf->device->createKernelArgMap();
        	}
		}
    } else if (devtype == acc_device_current) {
		int numDevs = HostConf::devMap.at(tconf->acc_device_type_var).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
#ifdef _OPENARC_PROFILE_
			fprintf(stderr, "Host Thread %d uses device %d of type %d\n",get_thread_id(), devnum, tconf->acc_device_type_var);
#endif
		}
        tconf->device = HostConf::devMap.at(tconf->acc_device_type_var).at(devnum);
        tconf->acc_device_num_var = devnum;
#ifdef _OPENMP
		#pragma omp critical(acc_set_device_num_critical)
#endif
		{
        	if(tconf->device->init_done != 1) {
            	tconf->device->init();
        	} else {
            	tconf->device->createKernelArgMap();
        	}
		}
    } else if( devtype == acc_device_host ) {
/*
        //tconf->device = tconf->devMap.at(devtype).at(devnum);
        tconf->device = NULL;
        tconf->acc_device_num_var = devnum;
        tconf->acc_device_type_var = devtype;
#ifdef _OPENARC_PROFILE_
		fprintf(stderr, "Host Thread %d uses host device %d\n",get_thread_id(), devnum);
#endif
*/
		int numDevs = HostConf::devMap.at(devtype).size();
		if( numDevs <= devnum ) {
			fprintf(stderr, "[ERROR in acc_set_device_num()] device number (%d) should be smaller than the number of devices attached (%d); exit!\n", devnum, numDevs);
			exit(1);
		} else {
#ifdef _OPENARC_PROFILE_
			fprintf(stderr, "Host Thread %d uses device %d of type %d\n",get_thread_id(), devnum, devtype);
#endif
		}
        tconf->device = HostConf::devMap.at(devtype).at(devnum);
        tconf->acc_device_type_var = acc_device_host;
        tconf->acc_device_num_var = devnum;
        //printf("devType %d\n",devtype );
#ifdef _OPENMP
		#pragma omp critical(acc_set_device_num_critical)
#endif
		{
        	if(tconf->device->init_done != 1) {
            	tconf->device->init();
        	} else {
            	tconf->device->createKernelArgMap();
        	}
		}
    } else {
        fprintf(stderr, "[ERROR in acc_set_device_num()] Not supported device type %d; exit!\n", devtype);
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_set_device_num(%d, %d)\n", devnum, devtype);
	}
#endif
}

int acc_get_device_num( acc_device_t devtype ) {
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
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_get_device_num()\n");
	}
#endif
	int return_data;
    HostConf_t * tconf = getHostConf();
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
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_get_device_num()\n");
	}
#endif
	return return_data;
}

int acc_async_test( int asyncID ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_async_test(%d)\n", asyncID);
	}
#endif
	int return_data = 1;
    HostConf_t * tconf = getHostConf();
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_async_test()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    //return_data = tconf->device->HI_async_test(asyncID+tconf->asyncID_offset);
    HostConf_t * ttconf;
	for(std::vector<HostConf_t *>::iterator it = hostConfList.begin(); it != hostConfList.end(); ++it) {
		ttconf = *it;
		if(ttconf->device != NULL) {
			if( ttconf->device->HI_async_test_ifpresent(asyncID+ttconf->asyncID_offset) == 0 ) {
				return_data = 0;
				break;
			}
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_async_test(%d)\n", asyncID);
	}
#endif
	return return_data;
}

int acc_async_test_all() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_async_test_all()\n");
	}
#endif
	int return_data;
    HostConf_t * tconf = getHostConf();
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_async_test_all()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    return_data = tconf->device->HI_async_test_all();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_async_test_all()\n");
	}
#endif
	return return_data;
}

//[DEBUG] What if arg value is acc_async_noval?
void acc_wait( int arg ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_wait(%d)\n", arg);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_wait()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    //tconf->device->HI_wait(arg+tconf->asyncID_offset);
    HostConf_t * ttconf;
	for(std::vector<HostConf_t *>::iterator it = hostConfList.begin(); it != hostConfList.end(); ++it) {
		ttconf = *it;
		if(ttconf->device != NULL) {
			ttconf->device->HI_wait_ifpresent(arg+ttconf->asyncID_offset);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_wait(%d)\n", arg);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

//[DEBUG] acc_async_wait() is renamed to acc_wait().
void acc_async_wait( int asyncID ) {
	acc_wait(asyncID);
}

void acc_wait_all() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_wait_all()\n");
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_wait_all()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    tconf->device->HI_wait_all();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_wait_all()\n");
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

//[DEBUG] acc_async_wait_all() is renamed to acc_wait_all().
void acc_async_wait_all() {
	acc_wait_all();
}


void acc_init( acc_device_t devtype, int kernels, std::string kernelNames[], const char *fileNameBase ) {
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
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_init()\n");
	}
#endif
    HostConf_t * tconf = getInitHostConf();
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
    	tconf->addKernelNames(kernels, kernelNames);
		tconf->HI_kernels_registered = 1;
		tconf->device->HI_register_kernels(tconf->kernelnames);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_init()\n");
	}
#endif
}

void acc_init( acc_device_t devtype ) {
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
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_init()\n");
	}
#endif
    HostConf_t * tconf = getInitHostConf();
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
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_init()\n");
	}
#endif
}

void acc_shutdown( acc_device_t devtype ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_shutdown(%d)\n", devtype);
	}
#endif
    HostConf_t * tconf = getHostConf();
    if( tconf == NULL ) {
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
            //tconf->device->destroy();
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
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_shutdown(%d)\n", devtype);
	}
#endif
}

//DEBUG: below implementation can be called only by host threads.
//Call to this function within a GPU kernel should be overwritten
//by OpenACC-to-Device translator.
int acc_on_device( acc_device_t devtype ) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_on_device()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
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
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_on_device()\n");
	}
#endif
    return 0;
}

d_void* acc_malloc(size_t size) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_malloc()\n");
	}
#endif
	if( size == 0 ) {
    	fprintf(stderr, "[ERROR in acc_malloc()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    void *devPtr;
    if( tconf->isOnAccDevice ) {
        tconf->device->HI_malloc(&devPtr, size, HI_MEM_READ_WRITE);
    } else {
        fprintf(stderr, "[ERROR in acc_malloc()] target accelerator device has not been set; exit!\n");
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	tconf->DMallocCnt++;
	tconf->DMallocSize += size;
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_malloc()\n");
	}
#endif
    return devPtr;
}

void acc_free(void* devPtr) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_free()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    if( tconf->isOnAccDevice ) {
        tconf->device->HI_free(devPtr);
    } else {
        fprintf(stderr, "[ERROR in acc_free()] target accelerator device has not been set; exit!\n");
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
	tconf->DFreeCnt++;
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_free()\n");
	}
#endif
}

/////////////////////////////////////////////////////////// 
// OpenACC Runtime Library Routines added in Version 2.0 //
/////////////////////////////////////////////////////////// 
void acc_wait_async(int arg, int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_wait_async(%d, %d)\n", arg, async);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_wait_async()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    //tconf->device->HI_wait_async(arg+tconf->asyncID_offset, async+tconf->asyncID_offset);
    HostConf_t * ttconf;
	for(std::vector<HostConf_t *>::iterator it = hostConfList.begin(); it != hostConfList.end(); ++it) {
		ttconf = *it;
		if(ttconf->device != NULL) {
			ttconf->device->HI_wait_async_ifpresent(arg+ttconf->asyncID_offset, async+ttconf->asyncID_offset);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_wait_async(%d, %d)\n", arg, async);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

void acc_wait_all_async(int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_wait_all_async(%d)\n", async);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_wait_all_async()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
    tconf->device->HI_wait_all_async(async+tconf->asyncID_offset);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_wait_all_async(%d)\n", async);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

void* acc_copyin(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin()\n");
	}
#endif
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
	HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin()\n");
	}
#endif
	return devPtr;
}

void* acc_pcopyin(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcopyin()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcopyin()\n");
	}
#endif
	return devPtr;
}

void* acc_present_or_copyin(h_void* hostPtr, size_t size) {
	return acc_pcopyin(hostPtr, size);
}

void* acc_create(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create()\n");
	}
#endif
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create()\n");
	}
#endif
	return devPtr;
}

void* acc_pcreate(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcreate()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcreate()\n");
	}
#endif
	return devPtr;
}

void* acc_present_or_create(h_void* hostPtr, size_t size) {
	return acc_pcreate(hostPtr, size);
}

//[FIXME] A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.
void acc_copyout(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyout()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_copyout() is not present on the device; exit!\n");
		exit(1);
	} else {
		HI_memcpy(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0);
		HI_free(hostPtr, DEFAULT_QUEUE);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyout()\n");
	}
#endif
}

//[FIXME] - A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.
//        - The current implementation does not consider size parameter;
//        free the whole data.
void acc_delete(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_delete()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_delete() is not present on the device; exit!\n");
		exit(1);
	} else {
		HI_free(hostPtr, DEFAULT_QUEUE);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_delete()\n");
	}
#endif
}

void acc_update_device(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_update_device()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)==HI_success)) {
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
	} else {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_update_device() is not present on the device; exit!\n");
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_update_device()\n");
	}
#endif
}

void acc_update_self(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_update_self()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)==HI_success)) {
		HI_memcpy(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0);
	} else {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_update_self() is not present on the device; exit!\n");
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_update_self()\n");
	}
#endif
}

void acc_map_data(h_void* hostPtr, d_void* devPtr, size_t size) {
	void* tDevPtr;
	void* tHostPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_map_data()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &tDevPtr, DEFAULT_QUEUE)==HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] It is an error to call acc_map_data for host data that is already present on the device; exit!\n");
		exit(1);
	} else if ( HI_get_host_address((const void *)devPtr, &tHostPtr, DEFAULT_QUEUE)==HI_success ) {
		fprintf(stderr, "[OPENARCRT-ERROR] It is undefined to call acc_map_data with a device address that is already mapped to host data; exit!\n");
		exit(1);
	} else {
		HI_set_device_address((const void *)hostPtr, devPtr, size, DEFAULT_QUEUE);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_map_data()\n");
	}
#endif
}

//[FIXME] It is undefined behavior to call acc_unmap_data with a host address
//unless that host address was mapped to device memory using acc_map_data, but
//this implementation does not check this.
void acc_unmap_data(h_void* hostPtr) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_unmap_data()\n");
	}
#endif
    HostConf *tconf = getHostConf();
	if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in acc_unmap_data()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
	}
	HI_remove_device_address((const void *)hostPtr, DEFAULT_QUEUE);
	//Unpin host data if they were implicitly pinned for asynchronous transfers.
    tconf->device->HI_unpin_host_memory((const void *)hostPtr);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_unmap_data()\n");
	}
#endif
}

d_void* acc_deviceptr(h_void* hostPtr) {
	void* devPtr;
	HI_error_t result;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_deviceptr()\n");
	}
#endif
	result = HI_get_device_address((const void*)hostPtr, &devPtr, DEFAULT_QUEUE);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_deviceptr()\n");
	}
#endif
	if (result==HI_success) {
		return (d_void *)(devPtr);
	} else {
		return NULL;
	}
}

h_void* acc_hostptr(d_void* devPtr) {
	void* hostPtr;
	HI_error_t result;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_hostptr()\n");
	}
#endif
	result = HI_get_host_address((const void*)devPtr, &hostPtr, DEFAULT_QUEUE);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_hostptr()\n");
	}
#endif
	if (result==HI_success) {
		return (h_void *)(hostPtr);
	} else {
		return NULL;
	}
}


int acc_is_present(h_void* hostPtr, size_t size) {
	void* devPtr;
	int isPresent = 0;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter isPresent()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)==HI_success)) {
		isPresent = 1;
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit isPresent()\n");
	}
#endif
	return isPresent;
}

void acc_memcpy_to_device(d_void* dest, h_void* src, size_t bytes) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_memcpy_to_device()\n");
	}
#endif
	HI_memcpy(dest, (const void*)src, bytes, HI_MemcpyHostToDevice, 0);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_memcpy_to_device()\n");
	}
#endif
}

void acc_memcpy_from_device(h_void* dest, d_void* src, size_t bytes) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_memcpy_from_device()\n");
	}
#endif
	HI_memcpy(dest, (const void*)src, bytes, HI_MemcpyDeviceToHost, 0);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_memcpy_from_device()\n");
	}
#endif
}

//////////////////////////////////////////////////////////////////////
// Experimental OpenACC Runtime Library Routines for Unified Memory //
//////////////////////////////////////////////////////////////////////
// If unified memory is supported,
// 		- Allocate unified memory, and copy data from hostPtr
// 		if hostPtr is not NULL.
// Else
// 		- Allocate host memory if hostPtr is NULL.
void* acc_copyin_unified(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin_unified()\n");
	}
#endif
	HI_malloc1D_unified(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
	HI_memcpy_unified(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin_unified()\n");
	}
#endif
	return devPtr;
}

// If unified memory is supported,
// 		- If not present, 
// 			- Allocate unified memory, and copy data from hostPtr
// 			if hostPtr is not NULL.
// Else
// 		- Allocate host memory if hostPtr is NULL.
void* acc_pcopyin_unified(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcopyin_unified()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		HI_malloc1D_unified(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
		HI_memcpy_unified(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcopyin_unified()\n");
	}
#endif
	return devPtr;
}

void* acc_present_or_copyin_unified(h_void* hostPtr, size_t size) {
	return acc_pcopyin_unified(hostPtr, size);
}

// If unified memory is supported,
// 		- Allocate unified memory.
// Else
// 		- Allocate host memory if hostPtr is NULL.
void* acc_create_unified(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create_unified()\n");
	}
#endif
	HI_malloc1D_unified(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create_unified()\n");
	}
#endif
	return devPtr;
}

// If unified memory is supported,
// 		- If not present
// 			- Allocate unified memory.
// Else
// 		- Allocate host memory if hostPtr is NULL.
void* acc_pcreate_unified(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcreate_unified()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		HI_malloc1D_unified(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcreate_unified()\n");
	}
#endif
	return devPtr;
}

void* acc_present_or_create_unified(h_void* hostPtr, size_t size) {
	return acc_pcreate_unified(hostPtr, size);
}

//[FIXME] A call to this routine is NOT allowed within a data region for
//the specified data, but the current runtime does not check this.
//Free the allocated memory (either host or unified memory).
//This has the same effect as calling acc_delete_unified().
void acc_copyout_unified(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyout_unified()\n");
	}
#endif
/*
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		//fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_copyout_unified() is not present on the device; exit!\n");
		//exit(1);
		HI_free_unified(hostPtr, DEFAULT_QUEUE);
	} else {
		HI_memcpy_unified(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0);
		HI_free_unified(hostPtr, DEFAULT_QUEUE);
	}
*/
	HI_free_unified(hostPtr, DEFAULT_QUEUE);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyout_unified()\n");
	}
#endif
}

//[FIXME] - A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.
//        - The current implementation does not consider size parameter;
//        free the whole data.
//Free the allocated memory (either host or unified memory).
void acc_delete_unified(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_delete_unified()\n");
	}
#endif
	//if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		//fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_delete_unified() is not present on the device; exit!\n");
		//exit(1);
		//If unified memory is not supported, just return.
		//return; 
	//} else {
		HI_free_unified(hostPtr, DEFAULT_QUEUE);
	//}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_delete_unified()\n");
	}
#endif
}

/////////////////////////////////////////////////////////////////
// Additional OpenACC Runtime Library Routines Used by OpenARC //
/////////////////////////////////////////////////////////////////

void* acc_copyin_const(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin_const()\n");
	}
#endif
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_ONLY);
	HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin_const()\n");
	}
#endif
	return devPtr;
}

void* acc_pcopyin_const(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcopyin_const()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_ONLY);
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcopyin_const()\n");
	}
#endif
	return devPtr;
}

void* acc_present_or_copyin_const(h_void* hostPtr, size_t size) {
	return acc_pcopyin_const(hostPtr, size);
}

void* acc_create_const(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create_const()\n");
	}
#endif
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_ONLY);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create_const()\n");
	}
#endif
	return devPtr;
}

void* acc_pcreate_const(h_void* hostPtr, size_t size) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcreate_const()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_ONLY);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcreate_const()\n");
	}
#endif
	return devPtr;
}

void* acc_present_or_create_const(h_void* hostPtr, size_t size) {
	return acc_pcreate_const(hostPtr, size);
}

void* acc_copyin_async(h_void* hostPtr, size_t size, int async) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin_async()\n");
	}
#endif
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
	if( async == acc_async_sync ) {
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
	} else {
		HI_memcpy_async(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, async);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin_async()\n");
	}
#endif
	return devPtr;
}

void* acc_copyin_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyin_async_wait()\n");
	}
#endif
	if( arg != acc_async_sync ) {
		acc_wait(arg);
	}
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
	if( async == acc_async_sync ) {
		HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
	} else {
		HI_memcpy_async(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, async);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyin_async_wait()\n");
	}
#endif
	return devPtr;
}

void* acc_pcopyin_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcopyin_async_wait()\n");
	}
#endif
	if( arg != acc_async_sync ) {
		acc_wait(arg);
	}
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
		if( async == acc_async_sync ) {
			HI_memcpy(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0);
		} else {
			HI_memcpy_async(devPtr, hostPtr, size, HI_MemcpyHostToDevice, 0, async);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcopyin_async_wait()\n");
	}
#endif
	return devPtr;
}

void* acc_present_or_copyin_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	return acc_pcopyin_async_wait(hostPtr, size, async, arg);
}

void* acc_create_async(h_void* hostPtr, size_t size, int async) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create_async()\n");
	}
#endif
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create_async()\n");
	}
#endif
	return devPtr;
}

void* acc_create_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_create_async_wait()\n");
	}
#endif
	if( arg != acc_async_sync ) {
		acc_wait(arg);
	}
	HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_create_async_wait()\n");
	}
#endif
	return devPtr;
}

void* acc_pcreate_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_pcreate_async_wait()\n");
	}
#endif
	if( arg != acc_async_sync ) {
		acc_wait(arg);
	}
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		HI_malloc1D(hostPtr, &devPtr, size, DEFAULT_QUEUE, HI_MEM_READ_WRITE);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_pcreate_async_wait()\n");
	}
#endif
	return devPtr;
}

void* acc_present_or_create_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	return acc_pcreate_async_wait(hostPtr, size, async, arg);
}

//[FIXME] A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.

void acc_copyout_async(h_void* hostPtr, size_t size, int async) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyout_async()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_copyout_async() is not present on the device; exit!\n");
		exit(1);
	} else {
		if( async == acc_async_sync ) {
			HI_memcpy(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0);
			HI_free(hostPtr, DEFAULT_QUEUE);
		} else {
			HI_memcpy_async(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, async);
			HI_free_async(hostPtr, async);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyout_async()\n");
	}
#endif
}

void acc_copyout_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_copyout_async_wait()\n");
	}
#endif
	if( arg != acc_async_sync ) {
		acc_wait(arg);
	}
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_copyout_async_wait() is not present on the device; exit!\n");
		exit(1);
	} else {
		if( async == acc_async_sync ) {
			HI_memcpy(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0);
			HI_free(hostPtr, DEFAULT_QUEUE);
		} else {
			HI_memcpy_async(hostPtr, devPtr, size, HI_MemcpyDeviceToHost, 0, async);
			HI_free_async(hostPtr, async);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_copyout_async_wait()\n");
	}
#endif
}

//[FIXME] - A call to this routine is NOT alloed within a data region for
//the specified data, but the current runtime does not check this.
//        - The current implementation does not consider size parameter;
//        free the whole data.
void acc_delete_async(h_void* hostPtr, size_t size, int async) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_delete_async()\n");
	}
#endif
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_delete_async() is not present on the device; exit!\n");
		exit(1);
	} else {
		if( async == acc_async_sync ) {
			HI_free(hostPtr, DEFAULT_QUEUE);
		} else {
			HI_free_async(hostPtr, async);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_delete_async()\n");
	}
#endif
}

void acc_delete_async_wait(h_void* hostPtr, size_t size, int async, int arg) {
	void* devPtr;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] enter acc_delete_async_wait()\n");
	}
#endif
	if( arg != acc_async_sync ) {
		acc_wait(arg);
	}
	if ((HI_get_device_address(hostPtr, &devPtr, DEFAULT_QUEUE)!=HI_success)) {
		fprintf(stderr, "[OPENARCRT-ERROR] the argument data of acc_delete_async_wait() is not present on the device; exit!\n");
		exit(1);
	} else {
		if( async == acc_async_sync ) {
			HI_free(hostPtr, DEFAULT_QUEUE);
		} else {
			HI_free_async(hostPtr, async);
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO] exit acc_delete_async_wait()\n");
	}
#endif
}
