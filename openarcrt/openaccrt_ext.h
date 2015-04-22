#ifndef __OPENARC_EXT_HEADER__

#define __OPENARC_EXT_HEADER__

#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

typedef std::map<const void *, void *> addressmap_t;
typedef std::map<const void *, int> countermap_t;
typedef std::map<const void *, size_t> sizemap_t;
typedef std::map<int, addressmap_t *> asyncphostmap_t;
typedef std::map<int, sizemap_t *> asynchostsizemap_t;
typedef std::map<const void *, HI_memstatus_t> memstatusmap_t;

#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
typedef std::map<int, cudaStream_t> asyncmap_t;
typedef cudaStream_t HI_async_handle_t;
typedef std::map<int, CUevent> eventmap_cuda_t;
#endif

#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
typedef std::map<int, cl_event> eventmap_opencl_t;
#endif
typedef std::map<int, pointerset_t *> asyncfreemap_t;

typedef class HostConf HostConf_t;
//[DEBUG by Seyong Lee] below are deprecated.
//Below structure is needed by neither CUDA nor OpenCL.
/*
typedef struct
{
    size_t arg_size;
    void* arg_val;
} argument_t;
typedef std::map<int, argument_t> argmap_t;
*/

typedef struct
{
	int num_args;
	void** kernelParams;
} kernelParams_t;


#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
typedef std::map<Accelerator *, std::map<std::string, CUfunction> > kernelmapcuda_t;
#else
typedef std::map<Accelerator *, std::map<std::string, cl_kernel> > kernelmapopencl_t;
#endif

#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
typedef class CudaDriver: public Accelerator
{
private:
    std::map<int,  CUstream> queueMap;
    std::map<int, eventmap_cuda_t > threadQueueEventMap;

    //HI_error_t pin_host_memory(const void* hostPtr, size_t size);
    HI_error_t pin_host_memory_if_unpinned(const void* hostPtr, size_t size);
    //void unpin_host_memory(const void* hostPtr);
    void dec_pinned_host_memory_counter(const void* hostPtr);
    void inc_pinned_host_memory_counter(const void* hostPtr);
    void unpin_host_memory_all(int asyncID);
    void unpin_host_memory_all();
	void release_freed_device_memory(int asyncID);
	void release_freed_device_memory();
public:
	//[DEBUG] changed to non-static variable.
    std::vector<std::string> kernelNameList;

    //A map of pinned memory and its usage count. If count value is 0, then the runtime can unpin the host memory.
    static std::map<CUdeviceptr,int> pinnedHostMemCounter;
    static std::vector<const void *> hostMemToUnpin;

	//[DEBUG] changed to non-static variable.
    //std::map<std::string, CUfunction> kernelMap;
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;

    CudaDriver(acc_device_t devType, int devNum, std::vector<std::string>kernelNames, HostConf_t *conf, int numDevices);
    HI_error_t init();
    HI_error_t HI_register_kernels(std::vector<std::string>kernelNames);
    HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args);
    HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value);
    HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE);
    HI_error_t HI_synchronize();
    HI_error_t destroy();
    HI_error_t HI_malloc1D(const void *hostPtr, void **devPtr, int count, int asyncID);
    HI_error_t HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType);
    HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID);
    HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID);
    HI_error_t HI_free( const void *hostPtr, int asyncID);
    HI_error_t HI_pin_host_memory(const void* hostPtr, size_t size);
    void HI_unpin_host_memory(const void* hostPtr);

    HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async);
    HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async);
    HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind);
    HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async);
    HI_error_t HI_memcpy2D_asyncS(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async);

    void HI_tempFree( void** tempPtr, acc_device_t devType);
    void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType);
	
	// Experimental API to support unified memory //
    HI_error_t HI_malloc1D_unified(const void *hostPtr, void **devPtr, int count, int asyncID);
    HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType);
    HI_error_t HI_free_unified( const void *hostPtr, int asyncID);

    static int HI_get_num_devices(acc_device_t devType);
    void HI_malloc(void **devPtr, size_t size);
    void HI_free(void *devPtr);
    HI_error_t createKernelArgMap();
    HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size);
    HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count);
    void HI_set_async(int asyncId);
    void HI_wait(int arg);
    void HI_waitS1(int arg);
    void HI_waitS2(int arg);
    void HI_wait_all();
    void HI_wait_async(int arg, int async);
    void HI_wait_all_async(int async);
    int HI_async_test(int asyncId);
    int HI_async_test_all();
    CUstream getQueue(int async) {
		if( queueMap.count(async + 2) == 0 ) {
			fprintf(stderr, "[ERROR in getQueue()] queue does not exist for async ID = %d\n", async);
			exit(1);
		} 
        return queueMap.at(async + 2);
    }

    CUevent getEvent(int async) {
		int thread_id = get_thread_id();
		if( (threadQueueEventMap.count(thread_id) == 0) || (threadQueueEventMap.at(thread_id).count(async + 2) == 0) ) {
			fprintf(stderr, "[ERROR in getEvent()] event does not exist for async ID = %d and thread ID = %d\n", async, thread_id);
			exit(1);
		}
        return threadQueueEventMap.at(get_thread_id()).at(async + 2);
    }

} CudaDriver_t;
#endif

#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
typedef class OpenCLDriver: public Accelerator
{
private:
    std::map<int,  cl_command_queue> queueMap;
    std::map<int, eventmap_opencl_t > threadQueueEventMap;

public:
	//[DEBUG] changed to non-static variable.
    std::vector<std::string> kernelNameList;
    cl_platform_id clPlatform;
    cl_device_id clDevice;
    static cl_context clContext;
    cl_command_queue clQueue;
    cl_program clProgram;

    OpenCLDriver(acc_device_t devType, int devNum, std::vector<std::string>kernelNames, HostConf_t *conf, int numDevices);
    HI_error_t init();
    HI_error_t HI_register_kernels(std::vector<std::string>kernelNames);
    HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args);
    HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value);
    HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE);
    HI_error_t HI_synchronize();
    HI_error_t destroy();
    HI_error_t HI_malloc1D(const void *hostPtr, void **devPtr, int count, int asyncID);
    HI_error_t HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType);
    HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID);
    HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID);
    HI_error_t HI_free( const void *hostPtr, int asyncID);
    HI_error_t HI_pin_host_memory(const void* hostPtr, size_t size);
    void HI_unpin_host_memory(const void* hostPtr);

    HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async);
    HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async);
    HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind);
    HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async);

    void HI_tempFree( void** tempPtr, acc_device_t devType);
    void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType);
	
	// Experimental API to support unified memory //
    HI_error_t HI_malloc1D_unified(const void *hostPtr, void **devPtr, int count, int asyncID);
    HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType);
    HI_error_t HI_free_unified( const void *hostPtr, int asyncID);

    static int HI_get_num_devices(acc_device_t devType);
    void HI_malloc(void **devPtr, size_t size);
    void HI_free(void *devPtr);
    HI_error_t createKernelArgMap();

    void HI_set_async(int asyncId);
    void HI_wait(int arg);
    void HI_waitS1(int arg);
    void HI_waitS2(int arg);
    void HI_wait_all();
    void HI_wait_async(int arg, int async);
    void HI_wait_all_async(int async);
    int HI_async_test(int asyncId);
    int HI_async_test_all();

    cl_command_queue getQueue(int async) {
		if( queueMap.count(async + 2) == 0 ) {
			fprintf(stderr, "[ERROR in getQueue()] queue does not exist for async = %d\n", async);
			exit(1);
		} 
        return queueMap.at(async + 2);
    }

    cl_event * getEvent(int async) {
		int thread_id = get_thread_id();
		if( (threadQueueEventMap.count(thread_id) == 0) || (threadQueueEventMap.at(thread_id).count(async + 2) == 0) ) {
			fprintf(stderr, "[ERROR in getEvent()] event does not exist for async ID = %d and thread ID = %d\n", async, thread_id);
			exit(1);
		}
        return &(threadQueueEventMap.at(get_thread_id()).at(async + 2));
    }


} OpenCLDriver_t;
#endif

typedef std::map<int, Accelerator_t*> devnummap_t;
typedef std::map<acc_device_t, devnummap_t> devmap_t;
//[DEBUG by Seyong Lee] below is deprecated.
//typedef std::map<Accelerator *, std::map<std::string, argmap_t*> > kernelargsmap_t;
typedef std::map<Accelerator *, std::map<std::string, kernelParams_t*> > kernelargsmap_t;
typedef std::map<std::string, long> kernelcnt_t;
typedef std::map<std::string, double> kerneltiming_t;

class HostConf
{
public:
    Accelerator_t *device;
    kernelargsmap_t kernelArgsMap;
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
    kernelmapcuda_t kernelsMap;
#else
    kernelmapopencl_t kernelsMap;
#endif
    std::vector<std::string> kernelnames;
    static devmap_t devMap;
    HostConf() {
        HI_init_done = 0;
        HI_kernels_registered = 0;
        acc_device_type_var = acc_device_none;
        acc_device_num_var = 0;
        acc_num_devices = 0;
        isOnAccDevice = 0;
		use_unifiedmemory = 1;
		prepin_host_memory = 1;
#ifdef _OPENARC_PROFILE_
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
		totalACCTime = 0.0;
		totalInitTime = 0.0;
		totalShutdownTime = 0.0;
		totalKernelSyncTime = 0.0;
		totalPresentTableTime = 0.0;
		totalRegKernelArgTime = 0.0;
		KernelCNTMap.clear();
		KernelTimingMap.clear();
#endif
        setDefaultDevice();
        setDefaultDevNum();
    }

    ~HostConf() {
        HI_reset();
        delete device;
    }

    int HI_init_done;
    int HI_kernels_registered;
    acc_device_t acc_device_type_var;
    acc_device_t user_set_device_type_var;
    int acc_device_num_var;
    int acc_num_devices;
    int isOnAccDevice;
	int use_unifiedmemory;
	int prepin_host_memory;

#ifdef _OPENARC_PROFILE_
    long H2DMemTrCnt;
    long H2HMemTrCnt;
    long D2HMemTrCnt;
    long D2DMemTrCnt;
    long HMallocCnt;
    long DMallocCnt;
    long HFreeCnt;
    long DFreeCnt;
	long KernelSyncCnt;
	long PresentTableCnt;
	long WaitCnt;
	long RegKernelArgCnt;
    unsigned long H2DMemTrSize;
    unsigned long H2HMemTrSize;
    unsigned long D2HMemTrSize;
    unsigned long D2DMemTrSize;
    double start_time;
    double end_time;
    double totalWaitTime;
    double totalResultCompTime;
    double totalMemTrTime;
    double totalMallocTime;
    double totalFreeTime;
    double totalACCTime;
    double totalInitTime;
    double totalShutdownTime;
	double totalKernelSyncTime;
	double totalPresentTableTime;
	double totalRegKernelArgTime;
	kernelcnt_t KernelCNTMap;
	kerneltiming_t KernelTimingMap;
#endif


    memstatusmap_t *hostmemstatusmaptable;
    memstatusmap_t *devicememstatusmaptable;
    countermap_t  *prtcntmaptable;

    void HI_init(int devNum);
    void HI_reset();
    void setDefaultDevice();
    void setDefaultDevNum();
    void initKernelNames(int kernels, std::string kernelNames[]);

    int genOCL;
    void setTranslationType();
    void createHostTables();

};



extern std::vector<HostConf_t *> hostConfList;

extern int HI_hostinit_done;

extern int HI_openarcrt_verbosity;

extern int HI_num_kernels;

extern std::string *HI_kernelNames;

////////////////////////
// Runtime init/reset //
////////////////////////
extern HostConf_t * getInitHostConf();
extern HostConf_t * getHostConf();
extern HostConf_t * setNGetHostConf(int devNum);


#endif
