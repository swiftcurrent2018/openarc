#include <stdlib.h>
#include <limits.h>
#include "openacc.h"
#include "openaccrt.h"
#include "openaccrt_ext.h"
////////////////////////////////////////
// Functions used for resilience test //
////////////////////////////////////////
#include "resilience.cpp"

static const char *omp_num_threads_env = "OMP_NUM_THREADS";
static const char *acc_device_type_env = "ACC_DEVICE_TYPE";
static const char *acc_device_num_env = "ACC_DEVICE_NUM";
static const char *outputType = "OPENARC_ARCH";
static const char *openarcrt_verbosity_env = "OPENARCRT_VERBOSITY";
static const char *openarcrt_unifiedmemory_env = "OPENARCRT_UNIFIEDMEM";
static const char *openarcrt_prepinhostmemory_env = "OPENARCRT_PREPINHOSTMEM";
static const char *NVIDIA = "NVIDIA";
static const char *RADEON = "RADEON";
static const char *XEONPHI = "XEONPHI";

devmap_t HostConf::devMap;

int HI_hostinit_done = 0;
int HI_openarcrt_verbosity = 0;
int HI_use_unifiedmemory = 0;
int HI_prepin_host_memory = 1;

//Return a local time in seconds.
double HI_get_localtime () {
    struct timeval time;
    gettimeofday(&time, 0);
    return time.tv_sec + time.tv_usec / 1000000.0;
}

std::vector<HostConf_t *> hostConfList;

//DEBUG: for now, resilience test is enabled by default
#define _OPENARC_RESILIENCE_


////////////////////////
// Runtime init/reset //
////////////////////////
//[FIXME] if default device type is different from the one passed to acc_init(),
//the default device type should be updated to the passed value.
//==> tconf->acc_device_type_var is updated in acc_init().
void HI_hostinit(int numhostthreads) {
    int thread_id = get_thread_id();
    int currentListSize = hostConfList.size();
    int newListSize = numhostthreads;
	int openarcrt_verbosity = 0;
    char * envVar;
    envVar = getenv(openarcrt_verbosity_env);
	if( envVar != NULL ) {
		openarcrt_verbosity = atoi(envVar);
		if( openarcrt_verbosity > 0 ) {
			HI_openarcrt_verbosity = openarcrt_verbosity;
		}
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_hostinit(%d)\n", numhostthreads);
	}
#endif
    envVar = getenv(openarcrt_unifiedmemory_env);
	if( envVar != NULL ) {
		HI_use_unifiedmemory = atoi(envVar);
	} else {
		HI_use_unifiedmemory = 0;
	}
    envVar = getenv(openarcrt_prepinhostmemory_env);
	if( envVar != NULL ) {
		HI_prepin_host_memory = atoi(envVar);
	} else {
		//Default behavior is changed to no-prepinning.
		HI_prepin_host_memory = 0;
	}
	if( HI_prepin_host_memory == 1 ) {
#ifdef _OPENARC_PROFILE_
		fprintf(stderr, "[OPENARCRT-INFO] Host memory will be prepinned for fast memory transfers; to disable this, set environment variable %s to 0 (excessive prepinning may slow or crash the program.)\n", openarcrt_prepinhostmemory_env);
#endif
	}
    if( numhostthreads <= 0 ) {
		envVar = NULL;
        envVar = getenv(omp_num_threads_env);
        if( envVar == NULL ) {
#ifdef _OPENMP
			fprintf(stderr, "[OPENARCRT-ERROR] To use OpenMP, environment variable, %s should be set to the maximum number of OpenMP threads that the program uses; exit!\n", omp_num_threads_env);
			exit(1);
#endif
            newListSize = 1;
        } else {
            newListSize = atoi(envVar);
            if( newListSize <= 0 ) {
#ifdef _OPENMP
				fprintf(stderr, "[OPENARCRT-ERROR] To use OpenMP, environment variable, %s should be set to the maximum number of OpenMP threads that the program uses; exit!\n", omp_num_threads_env);
				exit(1);
#endif
                //[DEBUG] wrong value; use default value of 1.
                newListSize = 1;
            }
        }
    }
    if( newListSize > currentListSize ) {
#ifdef _OPENMP
        #pragma omp critical (HI_hostinit_critical)
#endif
        {
            currentListSize = hostConfList.size();
            for( int i=currentListSize; i<newListSize; i++ ) {
                HostConf_t * tconf = new HostConf_t;
                tconf->setDefaultDevNum();
                tconf->setDefaultDevice();
                tconf->createHostTables();
				tconf->initKernelNames(HI_num_kernels, HI_kernelNames);
				tconf->use_unifiedmemory = HI_use_unifiedmemory;
				tconf->prepin_host_memory = HI_prepin_host_memory;
                //tconf->HI_init_done=1;
                hostConfList.push_back(tconf);
            }
            HI_hostinit_done = 1;
        }
    } else {
    	HI_hostinit_done = 1;
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_hostinit(%d)\n", numhostthreads);
	}
#endif
}

//Get the initial host configuration, or create it if nox existing.
//This function is called only in acc_init() function.
HostConf_t * getInitHostConf() {
    HostConf_t * tconf = NULL;
    int thread_id = get_thread_id();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d enters getInitHostConf()\n", thread_id);
	}
#endif
    if( thread_id < hostConfList.size() ) {
        tconf = hostConfList.at(thread_id);
    }

    if( tconf == NULL ) {
#ifdef _OPENMP
        //HI_hostinit(omp_get_num_threads());
        HI_hostinit(0);
#else
        HI_hostinit(1);
#endif
        tconf = hostConfList.at(thread_id);
        if( tconf == NULL ) {
            fprintf(stderr, "[ERROR in getInitHostConf] No host configuration exists for the current host thread (thread ID: %d);\n", thread_id);
            exit(1);
        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d exits getInitHostConf()\n", thread_id);
	}
#endif
    return tconf;
}

//Similar to getInitHostConf(), but this also invokes HI_init() if not done.
//This function is called in most of existing OpenARC runtime APIs.
HostConf_t * getHostConf() {
    HostConf_t * tconf = NULL;
    int thread_id = get_thread_id();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d enters getHostConf()\n", thread_id);
	}
#endif
    if( thread_id < hostConfList.size() ) {
        tconf = hostConfList.at(thread_id);
    }

    if( tconf == NULL ) {
#ifdef _OPENMP
        //HI_hostinit(omp_get_num_threads());
        HI_hostinit(0);
#else
        HI_hostinit(1);
#endif
        tconf = hostConfList.at(thread_id);
        if( tconf == NULL ) {
            fprintf(stderr, "[ERROR in getHostConf] No host configuration exists for the current host thread (thread ID: %d);\n", thread_id);
            exit(1);
        }
    }
	if( tconf->HI_init_done == 0 ) {
		tconf->HI_init_done = 1; //This should execute first.
		tconf->HI_init(DEVICE_NUM_UNDEFINED);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d exits getHostConf()\n", thread_id);
	}
#endif
    return tconf;
}

//Similar to getHostConf(), but this also sets device number.
//This function is called only in acc_set_device_num() function.
HostConf_t * setNGetHostConf(int devNum) {
    HostConf_t * tconf = NULL;
    int thread_id = get_thread_id();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d enters getHostConf()\n", thread_id);
	}
#endif
    if( thread_id < hostConfList.size() ) {
        tconf = hostConfList.at(thread_id);
    }

    if( tconf == NULL ) {
#ifdef _OPENMP
        //HI_hostinit(omp_get_num_threads());
        HI_hostinit(0);
#else
        HI_hostinit(1);
#endif
        tconf = hostConfList.at(thread_id);
        if( tconf == NULL ) {
            fprintf(stderr, "[ERROR in getHostConf] No host configuration exists for the current host thread (thread ID: %d);\n", thread_id);
            exit(1);
        }
    }
	if( tconf->HI_init_done == 0 ) {
		tconf->HI_init_done = 1; //This should execute first.
		tconf->HI_init(devNum);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d exits getHostConf()\n", thread_id);
	}
#endif
    return tconf;
}

//Function to convert input string to uppercases.
static char *convertToUpper(char *str) {
    char *newstr, *p;
    p = newstr = strdup(str);
    //while((*p++=toupper(*p)));
    while((*p=toupper(*p))) {p++;} //Changed to unambiguous way
    return newstr;
}


void HostConf::setDefaultDevice() {
    char * envVar;
    char * envVarU;
    envVar = getenv(acc_device_type_env);
    if( envVar == NULL ) {
        //acc_device_gpu is default device type.
        acc_device_type_var = acc_device_gpu;
    } else {
        envVarU = convertToUpper(envVar);
        if( (strcmp(envVarU, NVIDIA) == 0) ||
       			(strcmp(envVarU, RADEON) == 0) ||
                (strcmp(envVarU, "ACC_DEVICE_DEFAULT") == 0) ) {
            //acc_set_device_type(acc_device_nvidia);
            acc_device_type_var = acc_device_gpu;
        } else if( strcmp(envVarU, XEONPHI) == 0 ) {
            //acc_set_device_type(acc_device_xeonphi);
            acc_device_type_var = acc_device_xeonphi;
        } else if( strcmp(envVarU, "ACC_DEVICE_NONE") == 0 ) {
            //acc_set_device_type(acc_device_none);
            acc_device_type_var = acc_device_none;
        } else if( strcmp(envVarU, "ACC_DEVICE_HOST") == 0 ) {
            //acc_set_device_type(acc_device_host);
            acc_device_type_var = acc_device_host;
        } else if( strcmp(envVarU, "ACC_DEVICE_NOT_HOST") == 0 ) {
            //acc_set_device_type(acc_device_not_host);
            acc_device_type_var = acc_device_not_host;
        } else {
            //acc_set_device_type(acc_device_none);
            acc_device_type_var = acc_device_none;
        }
        free(envVarU);
    }
}

void HostConf::setDefaultDevNum() {
    int dev;
    char * envVar;
    //Set device number.
    acc_device_t devtype = acc_device_type_var;
    envVar = getenv(acc_device_num_env);
    if( envVar == NULL ) {
        //default device number (0) will be used.
        dev = 0;
    } else {
        dev = atoi(envVar);
        if( dev < 0 ) {
            dev = 0;
        }
    }
#ifdef _PMAS_
    dev = get_thread_id();
#endif
    //acc_set_device_num(dev, devtype);
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon) || 
            (devtype == acc_device_gpu) || (devtype == acc_device_xeonphi) ) {
        acc_device_num_var = dev;
    } else if( devtype == acc_device_host ) {
        acc_device_num_var = dev;
    } else {
        fprintf(stderr, "[ERROR in setDefaultDevNum()] Not supported device type %d; exit!\n", devtype);
        exit(1);
    }
//Below may not work since this method is called within a contructor.
    /*
        if( isOnAccDevice > 0 ) {
     		//CUDA device number starts from 0, but OpenACC device number starts
     		//from 1. (0 is for default device in OpenACC.)
            if( dev == 0 ) {
                cudaSetDevice(dev);
            } else if( dev > 0 ) {
                cudaSetDevice((dev-1));
            } else {
                fprintf(stderr, "[ERROR in setDefaultDevNum()] Not supported device number: %d; exit!\n", dev);
                exit(1);
            }
        }
    */

}

void HostConf::setTranslationType()
{
    int dev;
    char * envVar;
    //Set target device type. 
    acc_device_t devtype = acc_device_type_var;
    envVar = getenv(outputType);
    if( envVar == NULL ) {
        //default device number (0) will be used.
        genOCL = 0;
    } else {
        genOCL = atoi(envVar);
        if( genOCL < 0 ) {
            genOCL = 0;
        }
    }

}

void HostConf::createHostTables() {
    prtcntmaptable = new countermap_t;
    hostmemstatusmaptable = new memstatusmap_t;
    devicememstatusmaptable = new memstatusmap_t;
}


//This function initializes OpenARC runtimes for devices, but 
//not the actual device initialization, which will be done by
//acc_set_device_num() function later.
//If DEVICE_NUM_UNDEFINED is passed as an argument to this function,
//acc_set_device_num() is directly called in this function.
//(DEVICE_NUM_UNDEFINED will be passed when this function is
//called in getHostConf() or acc_init() function.)
//(DEVICE_NUM_UNDERFINED is also passed when HI_malloc() is
//called in a device driver, but it will not be actually called
//since an enclosing wrapper function always calls getHostConf()
//before calling the driver-level HI_malloc() function.
void HostConf::HI_init(int devNum) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_init()\n");
	}
#endif
    if( HI_hostinit_done == 0 ) {
        HI_hostinit(0);
    }
#ifdef _OPENARC_PROFILE_
	double ltime1 = HI_get_localtime();
    //printf("====> Profiling is enabled!!\n");
    H2DMemTrCnt = 0;
    H2HMemTrCnt = 0;
    D2HMemTrCnt = 0;
    D2DMemTrCnt = 0;
    HMallocCnt = 0;
    DMallocCnt = 0;
    HFreeCnt = 0;
    DFreeCnt = 0;
	KernelSyncCnt = 0;
	PresentTableCnt = 0;
	WaitCnt = 0;
	RegKernelArgCnt = 0;
    H2DMemTrSize = 0;
    H2HMemTrSize = 0;
    D2HMemTrSize = 0;
    D2DMemTrSize = 0;
    totalWaitTime = 0.0;
    totalResultCompTime = 0.0;
    totalMemTrTime = 0.0;
    totalMallocTime = 0.0;
    totalFreeTime = 0.0;
    totalACCTime = ltime1;
    totalInitTime = ltime1;
    totalShutdownTime = 0.0;
    totalKernelSyncTime = 0.0;
    totalPresentTableTime = 0.0;
    totalRegKernelArgTime = 0.0;
	KernelCNTMap.clear();
	KernelTimingMap.clear();
    for (std::vector<std::string>::iterator it = kernelnames.begin() ; it != kernelnames.end(); ++it) {
        //const char *kernelName = (*it).c_str();
        //fprintf(stderr, "[HI_init()] Kernel name = %s\n", kernelName);
		KernelCNTMap[*it] = 0;
		KernelTimingMap[*it] = 0.0;
    }  
#endif
    int thread_id = get_thread_id();
    setTranslationType();
    if( acc_device_type_var != acc_device_host ) {
		//printf("init start with dev %d\n", acc_device_type_var);
        devnummap_t numDevMap;
        int numDevices;
#ifdef _OPENMP
        #pragma omp critical (HI_init_critical)
#endif
		{ //starts critical section.
			if( HostConf::devMap.count(acc_device_type_var) > 0 ) {
				numDevices = HostConf::devMap.at(acc_device_type_var).size();
			} else {
				numDevices = 0;
			}
			if( numDevices == 0 ) {
        		if(genOCL) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH > 0
            		numDevices = OpenCLDriver::HI_get_num_devices(acc_device_type_var);
#else
					fprintf(stderr, "[OPENARCRT-ERROR]To generate OpenCL program, the environment variable OPENARC_ARCH should be a positive integer.\n");
					exit(1);
#endif
        		}	else {
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
            	numDevices = CudaDriver::HI_get_num_devices(acc_device_type_var);
#endif
        		}
				//printf("Num dev %d\n", numDevices);
				//fprintf(stderr, "Init dev num %d\n", acc_device_num_var);
				//[FIXME] initializing multiple devices may not work since only the last
				//device context will be visible to the current host thread; even if 
				//device type/number is changed, 
				//it will not be changed (only the last device will be executed.)
				//Easy fix is to call Accelerator::init() whenever device type/number is 
				//changed or new host thread joins. However, this may create too 
				//many device contexts if device type/number is frequently changed.
				//Better way is to call Accelerator::init() only if context does not 
				//exist; otherwise, attach host thread to the the context for the new 
				//device type/number.
        		for(int i=0 ; i < numDevices; i++) {
            		Accelerator *dev;
            		if(genOCL) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH > 0
                		dev = new OpenCLDriver_t(acc_device_type_var, i, kernelnames, this, numDevices);
#else
						fprintf(stderr, "[OPENARCRT-ERROR]To generate OpenCL program, the environment variable OPENARC_ARCH should be a positive integer.\n");
						exit(1);
#endif
            		} else {
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
                		dev = new CudaDriver_t(acc_device_type_var, i, kernelnames, this, numDevices);
#endif
            		}
            		//printf("Dev created %d\n", i);
            		//(*dev).init(); //Init will be called in acc_set_device_num().
            		numDevMap[i] = dev;
        		}
        		//insert all devices of this type into the map
        		HostConf::devMap[acc_device_type_var] = numDevMap;
			}
		} //ends critical section.
		if( devNum == DEVICE_NUM_UNDEFINED ) {
			//[DEBUG] if devNum != DEVICE_NUM_UNDEFINED, device number 
			//will be set by a separate acc_set_device_num() call.
        	setDefaultDevNum();
        	acc_set_device_num(acc_device_num_var, acc_device_type_var);
		}
		//printf("init done for type %d\n", acc_device_type_var);
        isOnAccDevice = 1;
        HI_init_done = 1;
    } else if( acc_device_type_var == acc_device_host ) {
        isOnAccDevice = 0;
        HI_init_done = 1;
        acc_num_devices = 1;
    }
#ifdef _OPENARC_RESILIENCE_
    HI_set_srand();
#endif

    createHostTables();
#ifdef _OPENARC_PROFILE_
    totalInitTime = HI_get_localtime() - ltime1;
#endif

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_init()\n");
	}
#endif
}

void HostConf::HI_reset() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_reset()\n");
	}
#endif
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    /*
    delete addressmaptable;
    delete countermaptable;
    delete prtcntmaptable;
	#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
    delete asyncmaptable;
    #endif
    delete asynchostphostmaptable;
    delete asynchostsizemaptable;
    delete hostmemstatusmaptable;
    delete devicememstatusmaptable;
    delete asyncfreemaptable;
    */
    createHostTables();
	device->masterAddressTable.clear();
	device->postponedFreeTable.clear();
	device->destroy();
	device->init_done = 0;

#ifdef _OPENARC_PROFILE_
    //totalFreeTime += HI_get_localtime() - ltime;
    totalShutdownTime = HI_get_localtime() - ltime;
    totalACCTime = HI_get_localtime() - totalACCTime;
    int thread_id = get_thread_id();
	double tAvgTime;
    printf("\n/************************************/\n");
    printf("/* Profile Output for host thread %d */\n", thread_id);
    printf("/************************************/\n");
    printf("Number of Host-to-Device Memory Transfer Calls: %ld\n", H2DMemTrCnt);
    printf("Number of Device-to-Host Memory Transfer Calls: %ld\n", D2HMemTrCnt);
    printf("Number of Host-to-Host Memory Transfer Calls: %ld\n", H2HMemTrCnt);
    printf("Number of Device-to-Device Memory Transfer Calls: %ld\n", D2DMemTrCnt);
    printf("Number of Device Memory Allocation Calls: %ld\n", DMallocCnt);
    printf("Number of Host Memory Allocation Calls by OpenARC runtime: %ld\n", HMallocCnt);
    printf("Number of Device Memory Free Calls: %ld\n", DFreeCnt);
    printf("Number of Host Memory Free Calls by OpenARC runtime: %ld\n", HFreeCnt);
    printf("Number of Host-Kernel Synchronization Calls by OpenARC runtime: %ld\n", KernelSyncCnt);
    printf("Number of Internal Present Table Calls by OpenARC runtime: %ld\n", PresentTableCnt);
    printf("Number of Wait Calls: %ld\n", WaitCnt);
    printf("Number of Kernel Argument Register Calls: %ld\n", RegKernelArgCnt);
    printf("Size of Data Transferred From Host to Device: %lu\n", H2DMemTrSize);
    printf("Size of Data Transferred From Host to Host: %lu\n", H2HMemTrSize);
    printf("Size of Data Transferred From Device to Host: %lu\n", D2HMemTrSize);
    printf("Size of Data Transferred From Device to Device: %lu\n", D2DMemTrSize);
    printf("Total Memory Transfer Time: %lf sec\n", totalMemTrTime);
    printf("Total Memory Allocation Time: %lf sec\n", totalMallocTime);
    printf("Total Memory Free Time: %lf sec\n", totalFreeTime);
    printf("Total ACC Init Time: %lf sec\n", totalInitTime);
    printf("Total ACC Shutdown Time: %lf sec\n", totalShutdownTime);
	if( KernelSyncCnt > 0 ) {
		tAvgTime = totalKernelSyncTime/((double)KernelSyncCnt);
	} else {
		tAvgTime = totalKernelSyncTime;
	}
    printf("Total Host-Kernel Synchronization Time: %lf sec (%lf sec per call)\n", totalKernelSyncTime, tAvgTime);
	if( PresentTableCnt > 0 ) {
		tAvgTime = totalPresentTableTime/((double)PresentTableCnt);
	} else {
		tAvgTime = totalPresentTableTime;
	}
    printf("Total Internal Present Table Access Time: %lf sec (%lf sec per call)\n", totalPresentTableTime, tAvgTime);
	if( WaitCnt > 0 ) {
		tAvgTime = totalWaitTime/((double)WaitCnt);
	} else {
		tAvgTime = totalWaitTime;
	}
    printf("Total Wait Time: %lf sec (%lf sec per call)\n", totalWaitTime, tAvgTime);
	if( RegKernelArgCnt > 0 ) {
		tAvgTime = totalRegKernelArgTime/((double)RegKernelArgCnt);
	} else {
		tAvgTime = totalRegKernelArgTime;
	}
    printf("Total Kernel Arg Register Time: %lf sec (%lf sec per call)\n", totalRegKernelArgTime, tAvgTime);
	long nKernelCalls = 0;
	for(std::map<std::string, long>::iterator it=KernelCNTMap.begin(); it!=KernelCNTMap.end(); ++it) {
		printf("Number of Kernel Calls (%s): %ld\n", (it->first).c_str(), it->second);
		nKernelCalls += it->second;
	}
	printf("Total Number of All Kernel Calls: %ld\n", nKernelCalls);
	double totalKernelTimes = 0.0;
	for(std::map<std::string, double>::iterator it=KernelTimingMap.begin(); it!=KernelTimingMap.end(); ++it) {
		int tCnt = KernelCNTMap.at(it->first);
		if( tCnt > 0 ) {
			tAvgTime = (it->second)/((double)tCnt);
		} else {
			tAvgTime = (it->second);
		}
		printf("Total Execution Time of a Kernel (%s): %lf (%lf per call)\n", (it->first).c_str(), it->second, tAvgTime);
		totalKernelTimes += it->second;
	}
	printf("Total Execution Time of All Kernels: %lf\n", totalKernelTimes);
    printf("Total Time of Other Overhead: %lf sec\n", (totalACCTime - totalKernelTimes - totalWaitTime - totalResultCompTime - totalMemTrTime - totalMallocTime - totalFreeTime - totalInitTime - totalShutdownTime - totalKernelSyncTime - totalPresentTableTime - totalRegKernelArgTime));
    printf("Total Host-Device Execution Time: %lf sec\n", totalACCTime);
    if( totalResultCompTime != 0.0 ) {
        printf("Total Result-Comp Time for Kernel Verification: %lf sec\n", totalResultCompTime);
    }
    H2DMemTrCnt = 0;
    H2HMemTrCnt = 0;
    D2HMemTrCnt = 0;
    D2DMemTrCnt = 0;
    HMallocCnt = 0;
    DMallocCnt = 0;
    HFreeCnt = 0;
    DFreeCnt = 0;
	KernelSyncCnt = 0;
	PresentTableCnt = 0;
	WaitCnt = 0;
    H2DMemTrSize = 0;
    H2HMemTrSize = 0;
    D2HMemTrSize = 0;
    D2DMemTrSize = 0;
    totalWaitTime = 0.0;
    totalResultCompTime = 0.0;
    totalMemTrTime = 0.0;
    totalMallocTime = 0.0;
    totalFreeTime = 0.0;
    totalACCTime = 0.0;
    totalInitTime = 0.0;
    totalShutdownTime = 0.0;
	totalKernelSyncTime = 0.0;
	totalPresentTableTime = 0.0;
	totalWaitTime = 0.0;
	KernelCNTMap.clear();
	KernelTimingMap.clear();
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_reset()\n");
	}
#endif
}

//////////////////////
// Kernel Execution //
//////////////////////
HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args)
{
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_register_kernel_numargs()\n");
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t* conf = getHostConf();
    return_status = conf->device->HI_register_kernel_numargs(kernel_name, num_args);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_register_kernel_numargs()\n");
	}
	conf->RegKernelArgCnt++;
	conf->totalRegKernelArgTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value)
{
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_register_kernel_arg()\n");
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t* conf = getHostConf();
    return_status = conf->device->HI_register_kernel_arg(kernel_name, arg_index, arg_size, arg_value);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_register_kernel_arg()\n");
	}
	conf->RegKernelArgCnt++;
	conf->totalRegKernelArgTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async) {
	HI_error_t return_status;
	const char *kernelName = kernel_name.c_str();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_kernel_call(%d): %s\n", async, kernelName);
		fprintf(stderr, "                \t\tGang configuration: %d, %d, %d\n", gridSize[2], gridSize[1], gridSize[0]);
		fprintf(stderr, "                \t\tWorker configuration: %d, %d, %d\n", blockSize[2], blockSize[1], blockSize[0]);
	}
#endif
	//if( (gridSize[0] == 0) && (gridSize[1] == 0) && (gridSize[2] == 0) ) {
	if( gridSize[0] == 0 ) {
    	//fprintf(stderr, "[WARNING in HI_kernel_call()] the kernel, %s, is called with 0 gangs; skip executing this kernel.\n", kernel_name);
    	std::cerr << "[WARNING in HI_kernel_call()] the kernel, " << kernel_name << " is called with 0 gangs; skip executing this kernel." << std::endl;
        return HI_success;
	}
    HostConf_t* conf = getHostConf();
    return_status = conf->device->HI_kernel_call(kernel_name, gridSize, blockSize, async);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_kernel_call(%d): %s\n", async, kernelName);
	}
#endif
	return return_status;
}

HI_error_t HI_synchronize()
{
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_synchronize()\n");
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t* conf = getHostConf();
    return_status = conf->device->HI_synchronize();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_synchronize()\n");
	}
	conf->KernelSyncCnt++;
	conf->totalKernelSyncTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

/////////////////////////////
//Device Memory Allocation //
/////////////////////////////
HI_error_t HI_malloc1D( const void *hostPtr, void** devPtr, size_t count, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_malloc1D(%d)\n", asyncID);
	}
#endif
	if( count == 0 ) {
    	fprintf(stderr, "[ERROR in HI_malloc1D()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_malloc1D()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_malloc1D(hostPtr, devPtr, count, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_malloc1D(%d)\n", asyncID);
	}
#endif
	return return_status;
}

HI_error_t HI_malloc1D_unified( const void *hostPtr, void** devPtr, size_t count, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_malloc1D_unified(%d)\n", asyncID);
	}
#endif
	if( count == 0 ) {
    	fprintf(stderr, "[ERROR in HI_malloc1D_unified()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_malloc1D_unified(hostPtr, devPtr, count, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_malloc1D_unified(%d)\n", asyncID);
	}
#endif
	return return_status;
}

HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_malloc2D(%d)\n", asyncID);
	}
#endif
	if( (widthInBytes == 0) || (height == 0) ) {
    	fprintf(stderr, "[ERROR in HI_malloc2D()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_malloc2D()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_malloc2D( hostPtr, devPtr,pitch, widthInBytes, height, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_malloc2D(%d)\n", asyncID);
	}
#endif
	return return_status;
}

HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_malloc3D(%d)\n", asyncID);
	}
#endif
	if( (widthInBytes == 0) || (height == 0) || (depth == 0) ) {
    	fprintf(stderr, "[ERROR in HI_malloc2D()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_malloc3D()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_malloc3D( hostPtr, devPtr, pitch, widthInBytes, height, depth, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_malloc3D(%d)\n", asyncID);
	}
#endif
	return return_status;
}

HI_error_t HI_free( const void *hostPtr, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_free(%d)\n", asyncID);
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_free()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_free(hostPtr, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_free(%d)\n", asyncID);
	}
#endif
	return return_status;
}

HI_error_t HI_free_unified( const void *hostPtr, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_free_unified(%d)\n", asyncID);
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_free_unified()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_free_unified(hostPtr, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_free_unified(%d)\n", asyncID);
	}
#endif
	return return_status;
}

//Unlike HI_free(), this method does not do actual memory deallocation;
//instead, it tells that following synchronization calls (acc_wait,
//acc_async_test, etc.) dellocate GPU memory for the variable.
HI_error_t HI_free_async( const void *hostPtr, int asyncID ) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_free_async(%d)\n", asyncID);
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_free_async()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_free_async(hostPtr, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_free_async(%d)\n", asyncID);
	}
#endif
	return return_status;
}

//malloc used for allocating temporary data.
//If the method is called for a pointer to existing memory, the existing memory
//will be freed before allocating new memory.
void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_tempMalloc1D()\n");
	}
#endif
	if( count == 0 ) {
    	fprintf(stderr, "[ERROR in HI_tempMalloc1D()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    tconf->device->HI_tempMalloc1D( tempPtr, count, devType);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_tempMalloc1D()\n");
	}
#endif
}

//Used for de-allocating temporary data.
void HI_tempFree( void** tempPtr, acc_device_t devType) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_tempFree()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    tconf->device->HI_tempFree( tempPtr, devType);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_tempFree()\n");
	}
#endif
}

/////////////////////////////////////////////////
//Memory transfers between a host and a device //
/////////////////////////////////////////////////
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
enum cudaMemcpyKind toCudaMemcpyKind( HI_MemcpyKind_t kind ) {
    switch( kind ) {
    case HI_MemcpyHostToHost: {
        return cudaMemcpyHostToHost;
    }
    case HI_MemcpyHostToDevice: {
        return cudaMemcpyHostToDevice;
    }
    case HI_MemcpyDeviceToHost: {
        return cudaMemcpyDeviceToHost;
    }
    case HI_MemcpyDeviceToDevice: {
        return cudaMemcpyDeviceToDevice;
    }
    }
    return cudaMemcpyHostToHost;
}
#endif
// Copy count bytes from the memory area pointed by src to the memory area
// pointed by dst, where kind is one of HI_MemcpyHostToHost, HI_MemcpyHostToDevice,
// HI_MemcpyDeviceToHost, or HI_MemcpyDeviceToDevice.
//     - trType is one of the following:
//         0: normal memcopy; for CUDA, this simply wraps cudaMemcpy().
//         1: use cudaMemcpyToSymbol or cudaMemcpyFromSymbol
HI_error_t HI_memcpy(void *dst, const void *src, size_t count,
                           HI_MemcpyKind_t kind, int trType) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy(%d)\n", count);
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy()] NULL src pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_memcpy( dst, src, count, kind, trType);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy(%d)\n", count);
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count,
                           HI_MemcpyKind_t kind, int trType) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_unified(%d)\n", count);
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_unified()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
		//If src pointer is NULL, skip memory copy operation.
		return HI_success;
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_memcpy_unified( dst, src, count, kind, trType);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_unified(%d)\n", count);
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count,
                                 HI_MemcpyKind_t kind, int trType, int async) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_async(%d, %d)\n", async, count);
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_async()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_async()] NULL src pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_memcpy_async(dst, src, count, kind, trType, async);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_async(%d, %d)\n", async, count);
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count,
                                 HI_MemcpyKind_t kind, int trType, int async) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_asyncS(%d, %d)\n", count, async);
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_asyncS()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_asyncS()] NULL src pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_memcpy_asyncS(dst, src, count, kind, trType, async);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_asyncS(%d, %d)\n", count, async);
	}
#endif
	return return_status;
}

void HI_waitS1(int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_waitS1(%d)\n", async);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    tconf->device->HI_waitS1(async);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_waitS1(%d)\n", async);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

void  HI_waitS2(int async) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_waitS2(%d)\n", async);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    tconf->device->HI_waitS2(async);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_waitS2(%d)\n", async);
	}
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif

}

HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                             size_t widthInBytes, size_t height, HI_MemcpyKind_t kind) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy2D(%d)\n", widthInBytes*height);
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D()] NULL src pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_memcpy2D(dst, dpitch, src, spitch, widthInBytes, height, kind);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy2D(%d)\n", widthInBytes*height);
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src,
                                   size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy2D_async(%d, %d)\n", widthInBytes*height, async);
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D_async()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D_async()] NULL src pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_memcpy2D_async(dst, dpitch, src, spitch, widthInBytes, height, kind, async);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy2D_async(%d, %d)\n", widthInBytes*height, async);
	}
#endif
	return return_status;
}

//extern HI_error_t HI_memcpy3D(void *dst, size_t dpitch, const void *src, size_t spitch,
//	size_t widthInBytes, size_t height, size_t depth, HI_MemcpyKind_t kind);
//extern HI_error_t HI_memcpy3D_async(void *dst, size_t dpitch, const void *src,
//	size_t spitch, size_t widthInBytes, size_t height, size_t depth,
//	HI_MemcpyKind_t kind, int async);

////////////////////////////
//Internal mapping tables //
////////////////////////////
HI_error_t HI_get_device_address(const void * hostPtr, void **devPtr, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_get_device_address(%d)\n", asyncID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_get_device_address(hostPtr, devPtr, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_get_device_address(%d)\n", asyncID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_set_device_address(const void * hostPtr, void *devPtr, size_t size, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_set_device_address(%d)\n", asyncID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_set_device_address(hostPtr, devPtr, size, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_set_device_address(%d)\n", asyncID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_remove_device_address(const void * hostPtr, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_remove_device_address(%d)\n", asyncID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_remove_device_address(hostPtr, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_remove_device_address(%d)\n", asyncID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}


HI_error_t HI_get_host_address(const void * devPtr, void **hostPtr, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_get_host_address(%d)\n", asyncID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_get_host_address(devPtr, hostPtr, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_get_host_address(%d)\n", asyncID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_get_temphost_address(const void * hostPtr, void **temphostPtr, int asyncID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_get_temphost_address(%d)\n", asyncID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_get_temphost_address(hostPtr, temphostPtr, asyncID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_get_temphost_address(%d)\n", asyncID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

int HI_getninc_prtcounter(const void * hostPtr, void **devPtr, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_getninc_prtcounter(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();

    int result;
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType);
    countermap_t * prtcounter = tconf->prtcntmaptable;
    if( HI_get_device_address(hostPtr, devPtr, asyncID) == HI_success ) {
        if( prtcounter->count(hostPtr) > 0 ) {
            result = prtcounter->at(hostPtr);
            if( result <= 0 ) result = 1;
            (*prtcounter)[hostPtr] = result + 1;
        } else {
            result = 1;
            (*prtcounter)[hostPtr] = 2;
        }
    } else {
        *devPtr = 0;
        (*prtcounter)[hostPtr] = 1;
        result = 0;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_getninc_prtcounter(%d)\n", asyncID);
	}
#endif
    return result;
}

int HI_decnget_prtcounter(const void * hostPtr, void **devPtr, int asyncID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_decnget_prtcounter(%d)\n", asyncID);
	}
#endif
    HostConf_t * tconf = getHostConf();

    int result;
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType);
    countermap_t * prtcounter = tconf->prtcntmaptable;
    if( HI_get_device_address(hostPtr, devPtr, asyncID) == HI_success) {
        HI_get_device_address(hostPtr, devPtr, asyncID);
        if( prtcounter->count(hostPtr) > 0 ) {
            result = prtcounter->at(hostPtr);
            result = result -1;
            if( result < 0 ) result = 0;
        } else {
            result = 0;
        }
        (*prtcounter)[hostPtr] = result;
    } else {
        *devPtr = 0;
        (*prtcounter)[hostPtr] = 0;
        result = -1; //error!!
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_decnget_prtcounter(%d)\n", asyncID);
	}
#endif
    return result;
}

//extern size_t HI_get_pitch(const void *hostPtr);

/////////////////////////////////////////////////////////////////////////
//async integer argument => internal handler (ex: CUDA stream) mapping //
/////////////////////////////////////////////////////////////////////////


void HostConf::initKernelNames(int kernels, std::string kernelNames[]) {
	//[FIXME] kernel names should be added only if they do not exist.
    for(int i= 0 ; i< kernels; i++) {
        kernelnames.push_back(kernelNames[i]);
    }

}


//Compiler will insert this before the first read access of the variable.
void HI_check_read(const void * hostPtr, acc_device_t dtype, const char * varName, const char *refName, int loopIndex) {
    HostConf_t * tconf = getHostConf();

    //acc_device_t devType = acc_get_device_type();
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType);
    memstatusmap_t * devicememstatusmap = tconf->devicememstatusmaptable;
    memstatusmap_t * hostmemstatusmap = tconf->hostmemstatusmaptable;
    //Initialize status maps if not existing (HI_init_status).
    if( hostmemstatusmap->count(hostPtr) == 0 ) {
        (*hostmemstatusmap)[hostPtr] = HI_notstale;
        (*devicememstatusmap)[hostPtr] = HI_notstale;
    }
    if( dtype == acc_device_nvidia || (dtype == acc_device_radeon) || (dtype == acc_device_gpu)) {
        HI_memstatus_t devicestatus = (*devicememstatusmap)[hostPtr];
        if( devicestatus == HI_stale ) {
            //printf("[DEBUG-ERROR] variable %32s should be copied from host to device for %64s.\n", varName, refName);
            std::cout <<"[DEBUG-ERROR] variable " << varName << " should be copied from host to device for " << refName;
            if( loopIndex != INT_MIN ) {
                std::cout <<" (enclosing loop index = " << loopIndex <<")";
            }
            std::cout <<"." <<std::endl;
        }
    } else {
        HI_memstatus_t hoststatus = (*hostmemstatusmap)[hostPtr];
        if( hoststatus == HI_stale ) {
            //printf("[DEBUG-ERROR] variable %32s should be copied from device to host for %64s.\n", varName, refName);
            std::cout <<"[DEBUG-ERROR] variable " << varName << " should be copied from device to host for " << refName <<"." <<std::endl;
        }
    }
}

//Compiler will insert this before the first write access of the variable.
void HI_check_write(const void * hostPtr, acc_device_t dtype, const char * varName, const char *refName, int loopIndex) {
    HostConf_t * tconf = getHostConf();

    //acc_device_t devType = acc_get_device_type();
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType);
    memstatusmap_t * devicememstatusmap = tconf->devicememstatusmaptable;
    memstatusmap_t * hostmemstatusmap = tconf->hostmemstatusmaptable;
    //Initialize status maps if not existing (HI_init_status).
    if( hostmemstatusmap->count(hostPtr) == 0 ) {
        (*hostmemstatusmap)[hostPtr] = HI_notstale;
        (*devicememstatusmap)[hostPtr] = HI_notstale;
    }
    if( dtype == acc_device_nvidia || (dtype == acc_device_radeon) || (dtype == acc_device_gpu)) {
        HI_memstatus_t devicestatus = (*devicememstatusmap)[hostPtr];
        if( devicestatus == HI_stale ) {
            //printf("[DEBUG-WARNING] variable %32s should be copied from host to device for %64s unless it is completely overwritten.\n", varName, refName);
            std::cout <<"[DEBUG-WARNING] variable " << varName << " should be copied from host to device for " << refName;
            if( loopIndex != INT_MIN ) {
                std::cout <<" (enclosing loop index = " << loopIndex <<")";
            }
            std::cout <<", unless it is completely overwritten before it is read." <<std::endl;
        }
        (*hostmemstatusmap)[hostPtr] = HI_stale;
        (*devicememstatusmap)[hostPtr] = HI_notstale;
    } else {
        HI_memstatus_t hoststatus = (*hostmemstatusmap)[hostPtr];
        if( hoststatus == HI_stale ) {
            //printf("[DEBUG-WARNING] variable %32s should be copied from device to host for %64s unless it is completely overwritten.\n", varName, refName);
            std::cout <<"[DEBUG-WARNING] variable " << varName << " should be copied from device to host for " << refName;
            if( loopIndex != INT_MIN ) {
                std::cout <<" (enclosing loop index = " << loopIndex <<")";
            }
            std::cout <<", unless it is completely overwritten before it is read." <<std::endl;
        }
        (*devicememstatusmap)[hostPtr] = HI_stale;
        (*hostmemstatusmap)[hostPtr] = HI_notstale;
    }
}

//Compiler will insert this after each memory transfer call for the variable
//or after GPU memory is freed.
void HI_set_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, const char * varName, const char *refName, int loopIndex) {
    HostConf_t * tconf = getHostConf();

    //acc_device_t devType = acc_get_device_type();
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType);
    memstatusmap_t * devicememstatusmap = tconf->devicememstatusmaptable;
    memstatusmap_t * hostmemstatusmap = tconf->hostmemstatusmaptable;
    //Initialize status maps if not existing (HI_init_status).
    if( hostmemstatusmap->count(hostPtr) == 0 ) {
        (*hostmemstatusmap)[hostPtr] = HI_notstale;
        (*devicememstatusmap)[hostPtr] = HI_notstale;
    }
    if( dtype == acc_device_nvidia || (dtype == acc_device_radeon) || (dtype == acc_device_gpu)) {
        HI_memstatus_t devicestatus = (*devicememstatusmap)[hostPtr];
        if( status == HI_notstale ) {
            if( devicestatus == HI_notstale ) {
                //printf("[DEBUG-INFO] copying variable %32s from host to device for %64s seems to be redundant.\n", varName, refName);
                std::cout <<"[DEBUG-INFO] copying variable " << varName << " from host to device for " << refName;
                if( loopIndex != INT_MIN ) {
                    std::cout <<" (enclosing loop index = " << loopIndex <<")";
                }
                std::cout <<" seems to be redundant." <<std::endl;
            } else if( devicestatus == HI_maystale ) {
                std::cout <<"[DEBUG-INFO] copying variable " << varName << " from host to device for " << refName;
                if( loopIndex != INT_MIN ) {
                    std::cout <<" (enclosing loop index = " << loopIndex <<")";
                }
                std::cout <<" can be redundant if it is completely overwritten by the device befere it is read." <<std::endl;
            }
        }
        (*devicememstatusmap)[hostPtr] = status;
    } else {
        HI_memstatus_t hoststatus = (*hostmemstatusmap)[hostPtr];
        if( status == HI_notstale ) {
            if( hoststatus == HI_notstale ) {
                //printf("[DEBUG-INFO] copying variable %32s from device to host for %64s seems to be redundant.\n", varName, refName);
                std::cout <<"[DEBUG-INFO] copying variable " << varName << " from device to host for " << refName;
                if( loopIndex != INT_MIN ) {
                    std::cout <<" (enclosing loop index = " << loopIndex <<")";
                }
                std::cout <<" seems to be redundant." <<std::endl;
            } else if( hoststatus == HI_maystale ) {
                std::cout <<"[DEBUG-INFO] copying variable " << varName << " from device to host for " << refName;
                if( loopIndex != INT_MIN ) {
                    std::cout <<" (enclosing loop index = " << loopIndex <<")";
                }
                std::cout <<" can be redundant if it is completely overwritten by the host befere it is read." <<std::endl;
            }
        }
        (*hostmemstatusmap)[hostPtr] = status;
    }
}

//Compiler will insert this right after a kernel call if the compiler analyzes
//that either the following CPU region does not access the variable
//(CPU status = notstale) or the variable seems to be not upward-exposed in the
//following CPU region (CPU status = maystale).
//This method is also inserted right after a kernel call for each reduction variable
//(GPU status = stale) or if GPU variable is deallocated (GPU status = stale).
//This method is similar to HI_set_status(), but this does not check any error
//or redundancy.
void HI_reset_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, int asyncID) {
    HostConf_t * tconf = getHostConf();

    //acc_device_t devType = acc_get_device_type();
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType);
    if( dtype == acc_device_nvidia || (dtype == acc_device_radeon) || (dtype == acc_device_gpu)) {
        memstatusmap_t * devicememstatusmap = tconf->devicememstatusmaptable;
        //HI_memstatus_t devicestatus = (*devicememstatusmap)[hostPtr];
        if( status == HI_stale ) {
            //Set the status to stale if GPU variable is freed.
            std::multimap<int, std::map<const void *,void*> >::iterator it = tconf->device->masterAddressTable.find(asyncID);
            std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
            if(it2 == (it->second).end() ) {
                (*devicememstatusmap)[hostPtr] = status;
            }
            /*
            addressmap_t * addressmap = tconf->addressmaptable;
            if( addressmap->count(hostPtr) == 0 ) {
            	(*devicememstatusmap)[hostPtr] = status;
            }
            */
            else if (asyncID > INT_MIN) {
                /*
                asyncfreemap_t * asyncfreemap = tconf->asyncfreemaptable;
                pointerset_t * freeset;
                if( asyncfreemap->count(asyncID) > 0 ) {
                	freeset = asyncfreemap->at(asyncID);
                	//GPU variable will be freed asyncronously.
                	if( (freeset != 0) && (freeset->count(hostPtr) > 0) ) {
                		(*devicememstatusmap)[hostPtr] = status;
                	}
                }
                */
                std::multimap<int, const void*>::iterator hostPtrIter = tconf->device->postponedFreeTable.find(asyncID);

                while(hostPtrIter != tconf->device->postponedFreeTable.end()) {
                    //fprintf(stderr, "[in HI_postponed_free()] Freeing on stream %d, address %x\n", asyncID, hostPtrIter->second);
                    if(hostPtrIter->second == hostPtr) {
                        (*devicememstatusmap)[hostPtr] = status;
                        break;
                    }
                    hostPtrIter++;
                }
            } else {
                //Set the status to stale if the variable is reduction one.
                (*devicememstatusmap)[hostPtr] = status;
            }
        } else {
            (*devicememstatusmap)[hostPtr] = status;
        }
    } else {
        memstatusmap_t * hostmemstatusmap = tconf->hostmemstatusmaptable;
        //HI_memstatus_t hoststatus = (*hostmemstatusmap)[hostPtr];
        (*hostmemstatusmap)[hostPtr] = status;
    }
}

HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_bind_tex()\n");
	}
#endif
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_bind_tex(texName, type, devPtr, size);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_bind_tex()\n");
	}
#endif
	return return_status;
}


HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_const(%d)\n", count);
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D_async()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf();
    return_status = tconf->device->HI_memcpy_const(hostPtr, constName, kind, count);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_const(%d)\n", count);
	}
#endif
	return return_status;
}

//This call ensures that the corresponding queue exists. If not, it is created.
void HI_set_async(int asyncId) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_set_async(%d)\n", asyncId);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf();
    tconf->device->HI_set_async(asyncId);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_set_async(%d)\n", asyncId);
	}
	tconf->totalKernelSyncTime += (HI_get_localtime() - ltime);
#endif
}
