#ifndef __OPENARC_HEADER__

#define __OPENARC_HEADER__

#include <cstring>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

//Comment out below to disable pthread-based thread safety.
#define _THREAD_SAFETY
#ifdef _OPENMP
#include <omp.h>
//#ifndef _THREAD_SAFETY
//#define _THREAD_SAFETY
//#endif
#endif

#ifdef _THREAD_SAFETY
#include <pthread.h>
#endif

#define DEFAULT_QUEUE -2
#define DEFAULT_ASYNC_QUEUE -1
#define DEVICE_NUM_UNDEFINED -1
#define MAX_NUM_QUEUES_PER_THREAD 1048576
#define NO_THREAD_ID -1

//VICTIM_CACHE_MODE = 0 
//  - Victim cache is not used
//VICTIM_CACHE_MODE = 1 
//  - Victim cache stores freed device memory
//  - The freed device memory can be reused if the size matches.
//  - More applicable, but host memory should be pinned again.
//VICTIM_CACHE_MODE = 2
//  - Victim cache stores both pinned host memory and corresponding device memory.
//  - The pinned host memory and device memory are reused only for the same 
//  host pointer; saving both memory pinning cost and device memory allocation cost 
//  but less applicable.
//  - Too much prepinning can cause slowdown or crash of the program.
//  - If OPENARCRT_PREPINHOSTMEM is set to 0, reuse only device memory.
//
//For OpenCL devices, VICTIM_CACHE_MODE = 1 is always applied.
#define VICTIM_CACHE_MODE 1

//PRESENT_TABLE_SEARCH_MODE = 0
//	- Assume that the elements in the container follow a strict order at all times
//	- Use a linear search if the table does not have a key entry matching the input key.
//	- base pointer search time: O(logn)
//	- intermediate pointer search time: O(n)
//PRESENT_TABLE_SEARCH_MODE = 1
//	- Assume that the elements in the container follow a strict order at all times
//	- Exploit the strict ordering if the table does not have a key entry matching the input key.
//	- base pointer search time: O(logn)
//	- intermediate pointer search time: O(logn)
#define PRESENT_TABLE_SEARCH_MODE 1

#ifdef _THREAD_SAFETY
extern pthread_mutex_t mutex_HI_init;
extern pthread_mutex_t mutex_HI_hostinit;
extern pthread_mutex_t mutex_HI_kernelnames;
extern pthread_mutex_t mutex_pin_host_memory;
extern pthread_mutex_t mutex_victim_cache;
extern pthread_mutex_t mutex_tempMalloc;
extern pthread_mutex_t mutex_set_async;
extern pthread_mutex_t mutex_set_device_num;
extern pthread_mutex_t mutex_clContext;
#endif

typedef enum {
    HI_success = 0,
    HI_error = 1
} HI_error_t;

typedef enum {
    HI_MemcpyHostToHost = 0,
    HI_MemcpyHostToDevice = 1,
    HI_MemcpyDeviceToHost = 2,
    HI_MemcpyDeviceToDevice = 3
} HI_MemcpyKind_t;

typedef enum {
    HI_MEM_READ_WRITE = 1,
    HI_MEM_READ_ONLY = 2,
    HI_MEM_WRITE_ONLY = 4,
} HI_MallocKind_t;

typedef enum {
    HI_notstale = 0,
    HI_stale = 1,
    HI_maystale = 2
} HI_memstatus_t;

typedef enum {
    HI_int = 0,
    HI_float = 1,
} HI_datatype_t;

typedef struct _HI_device_mem_handle_t {
    void* basePtr;
    size_t offset;
} HI_device_mem_handle_t;

typedef struct _addresstable_entity_t {
    void* basePtr;
    size_t size;
    _addresstable_entity_t(void* _basePtr, size_t _size) : basePtr(_basePtr), size(_size) {}
} addresstable_entity_t;

typedef std::map<const void *, void *> addressmap_t;
typedef std::map<int, addressmap_t *> addresstable_t;
typedef std::multimap<int, const void *> asyncfreetable_t;
typedef std::map<int, asyncfreetable_t *> asyncfreetablemap_t;
typedef std::multimap<int, void **> asynctempfreetable_t;
typedef std::map<int, asynctempfreetable_t *> asynctempfreetablemap_t;
typedef std::multimap<int, acc_device_t> asynctempfreetable2_t;
typedef std::map<int, asynctempfreetable2_t *> asynctempfreetablemap2_t;
typedef std::set<const void *> pointerset_t;
typedef std::map<int, addresstable_t *> addresstablemap_t;
typedef std::multimap<size_t, void *> memPool_t;
typedef std::map<const void *, size_t> sizemap_t;
typedef std::map<int, memPool_t *> memPoolmap_t;
typedef std::map<int, sizemap_t *> memPoolSizemap_t;
typedef std::map<void *, int> countermap_t;
typedef std::map<int, addressmap_t *> asyncphostmap_t;
typedef std::map<int, sizemap_t *> asynchostsizemap_t;
typedef std::map<const void *, HI_memstatus_t> memstatusmap_t;
#ifdef _OPENARC_PROFILE_
typedef std::map<int, long> presenttablecnt_t;
#endif

extern int HI_openarcrt_verbosity;
extern int HI_hostinit_done;
extern int HI_num_hostthreads;

//[CAUTION] For each device, there exists only one Accelerator object,
//and thus accessing Accelerator members are not thread-safe.
typedef class Accelerator
{
public:
    // Device info
    acc_device_t dev;
    int device_num;
	int num_devices;
    int init_done;
	int unifiedMemSupported;
	int compute_capability_major;
	int compute_capability_minor;
    int maxGridX, maxGridY, maxGridZ;
    int maxBlockX, maxBlockY, maxBlockZ;
    int maxNumThreadsPerBlock;
	int max1DTexRefWidth4LM;

	//kernel names that will be offloaded to this device.
	std::set<std::string> kernelNameSet;

	//Output kernel file name base. (Default value: "openarc_kernel")
	std::string fileNameBase;

    //Host-device address mapping table, augmented with stream id
    //addresstable_t masterAddressTable;
	addresstablemap_t masterAddressTableMap;

	//device-address-to-memory-handle mapping table, which is needed
	//for OpenCL backend only.
	//Current implementation uses a fake device virtual address for OpenCL,
	//which should be translated to actual cl_mem handle.
	addresstable_t masterHandleTable;

    //Auxiliary Host-device address mapping table used as a victim cache. 
    addresstable_t auxAddressTable;

	//temporarily allocated memory set.
	pointerset_t tempMallocSet;
    
    //Host-TempHost address mapping table, augmented with stream id
    addresstable_t tempHostAddressTable;

    //This table can have duplicate entries, owing to the HI_free_async
    //calls in a loop. To handle this, HI_free ensures that on a duplicate
    //pair, no free operation is performed
    //asyncfreetable_t postponedFreeTable;
	asyncfreetablemap_t postponedFreeTableMap;
	asynctempfreetablemap_t postponedTempFreeTableMap;
	asynctempfreetablemap2_t postponedTempFreeTableMap2;
	//memPool_t memPool;
	memPoolmap_t memPoolMap;
	memPoolSizemap_t tempMallocSizeMap;
#ifdef _OPENARC_PROFILE_
	presenttablecnt_t presentTableCntMap;
#endif
	

	virtual ~Accelerator() {};

    // Kernel Initialization
    virtual HI_error_t init(int threadID=NO_THREAD_ID) = 0;
    virtual HI_error_t destroy(int threadID=NO_THREAD_ID)=0;

    // Kernel Execution
    virtual HI_error_t HI_register_kernels(std::set<std::string>kernelNames, int threadID=NO_THREAD_ID) = 0;
    virtual HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args, int threadID=NO_THREAD_ID) = 0;
    virtual HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value, int arg_type, int threadID=NO_THREAD_ID) = 0;
    virtual HI_error_t HI_kernel_call(std::string kernel_name, size_t gridSize[3], size_t blockSize[3], int async=DEFAULT_QUEUE, int num_waits=0, int *waits=NULL, int threadID=NO_THREAD_ID) = 0;
    virtual HI_error_t HI_synchronize( int forcedSync = 0, int threadID=NO_THREAD_ID )=0;
    void updateKernelNameSet(std::set<std::string>kernelNames) {
    	for (std::set<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
        	if( kernelNameSet.count(*it) == 0 ) {
            	//Add a new kernel.
            	kernelNameSet.insert(*it);
        	}    
    	}    
	};

    // Memory Allocation
    virtual HI_error_t HI_malloc1D(const void *hostPtr, void **devPtr, size_t count, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE, int threadID=NO_THREAD_ID)= 0;
    virtual HI_error_t HI_malloc2D( const void *host_ptr, void** dev_ptr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE, int threadID=NO_THREAD_ID)=0;
    virtual HI_error_t HI_malloc3D( const void *host_ptr, void** dev_ptr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE, int threadID=NO_THREAD_ID)=0;
    virtual HI_error_t HI_free( const void *host_ptr, int asyncID, int threadID=NO_THREAD_ID)=0;
	virtual HI_error_t HI_pin_host_memory(const void * hostPtr, size_t size, int threadID=NO_THREAD_ID)=0;
	virtual void HI_unpin_host_memory(const void* hostPtr, int threadID=NO_THREAD_ID)=0;

    // Memory Transfer
    virtual HI_error_t HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int threadID=NO_THREAD_ID)=0;

    virtual HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID=NO_THREAD_ID)=0;
    virtual HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID=NO_THREAD_ID)=0;
    virtual HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int threadID=NO_THREAD_ID)=0;
    virtual HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async, int num_waits, int *waits, int threadID=NO_THREAD_ID)=0;

    virtual void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int threadID=NO_THREAD_ID)=0;
    virtual void HI_tempFree( void** tempPtr, acc_device_t devType, int threadID=NO_THREAD_ID)=0;
	
	// Experimental API to support unified memory //
    virtual HI_error_t  HI_malloc1D_unified(const void *hostPtr, void **devPtr, size_t count, int asyncID, HI_MallocKind_t flags, int threadID=NO_THREAD_ID)= 0;
    virtual HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int threadID=NO_THREAD_ID)=0;
    virtual HI_error_t HI_free_unified( const void *host_ptr, int asyncID, int threadID=NO_THREAD_ID)=0;

    virtual HI_error_t createKernelArgMap(int threadID=NO_THREAD_ID) {
        return HI_success;
    }
    virtual HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size) {
        return HI_success;
    }
    virtual HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID=NO_THREAD_ID) {
        return HI_success;
    }
    virtual HI_error_t HI_memcpy_const_async(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int async, int num_waits, int *waits, int threadID=NO_THREAD_ID) {
        return HI_success;
    }
    virtual HI_error_t HI_present_or_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID=NO_THREAD_ID) {
        return HI_success;
    }
    virtual void HI_set_async(int asyncId, int threadID=NO_THREAD_ID)=0;
    virtual void HI_set_context(int threadID=NO_THREAD_ID){}
    virtual void HI_wait(int arg, int threadID=NO_THREAD_ID) {}
    virtual void HI_wait_ifpresent(int arg, int threadID=NO_THREAD_ID) {}
    virtual void HI_wait_async(int arg, int async, int threadID=NO_THREAD_ID) {}
    virtual void HI_wait_async_ifpresent(int arg, int async, int threadID=NO_THREAD_ID) {}
    virtual void HI_waitS1(int arg, int threadID=NO_THREAD_ID) {}
    virtual void HI_waitS2(int arg, int threadID=NO_THREAD_ID) {}
    virtual void HI_wait_all(int threadID=NO_THREAD_ID) {}
    virtual void HI_wait_all_async(int async, int threadID=NO_THREAD_ID) {}
    virtual int HI_async_test(int asyncId, int threadID=NO_THREAD_ID)=0;
    virtual int HI_async_test_ifpresent(int asyncId, int threadID=NO_THREAD_ID)=0;
    virtual int HI_async_test_all(int threadID=NO_THREAD_ID)=0;
    virtual void HI_wait_for_events(int async, int num_waits, int* waits, int threadID=NO_THREAD_ID)=0;

    virtual void HI_malloc(void **devPtr, size_t size, HI_MallocKind_t flags, int threadID=NO_THREAD_ID) = 0;
    virtual void HI_free(void *devPtr, int threadID=NO_THREAD_ID) = 0;

    HI_error_t HI_get_device_address(const void *hostPtr, void **devPtr, int asyncID, int tid) {
		size_t offset;
		HI_error_t result = HI_get_device_address(hostPtr, devPtr, &offset, NULL, asyncID, tid);
		if( result == HI_success ) {
			//*devPtr contains a device pointer corresponding to the hostPtr.
			*devPtr = (void *)((size_t)*devPtr + offset); 
		} else {
			*devPtr = NULL;
		}
		return result;
    }

    HI_error_t HI_get_device_address(const void *hostPtr, void **devPtrBase, size_t* offset, int asyncID, int tid) {
        return HI_get_device_address(hostPtr, devPtrBase, offset, NULL, asyncID, tid);
    }

    HI_error_t HI_get_device_address(const void *hostPtr, void **devPtrBase, size_t *offset, size_t *size, int asyncID, int tid) {
		bool emptyTable1 = false;
#if PRESENT_TABLE_SEARCH_MODE == 0
		bool emptyTable2 = false;
#endif
    	addresstable_t *masterAddressTable = masterAddressTableMap[tid];
        int defaultAsyncID = DEFAULT_QUEUE+tid*MAX_NUM_QUEUES_PER_THREAD;
		addresstable_t::iterator it = masterAddressTable->find(asyncID);
        if(it == masterAddressTable->end() ) {
            addressmap_t* emptyMap = new addressmap_t();
            masterAddressTable->insert(std::pair<int, addressmap_t *> (asyncID, emptyMap));
			it = masterAddressTable->find(asyncID);
			emptyTable1 = true;
		}
		addressmap_t *tAddressMap = it->second;
#if PRESENT_TABLE_SEARCH_MODE == 0
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	tAddressMap->find(hostPtr);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        if(it2 != tAddressMap->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            *devPtrBase = aet->basePtr;
			if( size ) *size = aet->size;
			if( offset ) *offset = 0;
            return  HI_success;
        } else {
            //check on the default stream
			if( defaultAsyncID != asyncID ) {
            	it = masterAddressTable->find(defaultAsyncID);
        		if(it == masterAddressTable->end() ) {
            		addressmap_t *emptyMap = new addressmap_t();
            		masterAddressTable->insert(std::pair<int, addressmap_t*> (defaultAsyncID, emptyMap));
					//it = masterAddressTable->find(asyncID);
					emptyTable2 = true;
				} else {
					tAddressMap = it->second;
            		it2 =	tAddressMap->find(hostPtr);
#ifdef _OPENARC_PROFILE_
    				if( HI_openarcrt_verbosity > 1 ) {
						ptit->second++;
    				}    
#endif
            		if(it2 != tAddressMap->end() ) {
            			addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            			*devPtrBase = aet->basePtr;
						if( size ) *size = aet->size;
						if( offset ) *offset = 0;
            			return  HI_success;
            		}
				}
			} else {
				emptyTable2 = emptyTable1;
			}
		}

		if( emptyTable1 && emptyTable2 ) {
			*devPtrBase = NULL;
			if( size ) *size = 0;
			if( offset ) *offset = 0;
			return HI_error;
		} 

		//Check whether hostPtr is within the range of an allocated memory region 
		//in the addressTable.
		it = masterAddressTable->find(asyncID);
		tAddressMap = it->second;
		for (addressmap_t::iterator it2 = tAddressMap->begin(); it2 != tAddressMap->end(); ++it2) {
            const void* aet_host = it2->first;
#ifdef _OPENARC_PROFILE_
    		if( HI_openarcrt_verbosity > 1 ) {
				ptit->second++;
    		}    
#endif
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            if (hostPtr >= aet_host && (size_t) hostPtr < (size_t) aet_host + aet->size) {
                *devPtrBase = aet->basePtr;
				if( size ) *size = aet->size;
                if (offset) *offset = (size_t) hostPtr - (size_t) aet_host;
                return  HI_success;
            }
        }

        //check on the default stream
		if( defaultAsyncID != asyncID ) {
        	it = masterAddressTable->find(DEFAULT_QUEUE+tid*MAX_NUM_QUEUES_PER_THREAD);
			tAddressMap = it->second;
        	for (addressmap_t::iterator it2 = tAddressMap->begin(); it2 != tAddressMap->end(); ++it2) {
            	const void* aet_host = it2->first;
#ifdef _OPENARC_PROFILE_
    			if( HI_openarcrt_verbosity > 1 ) {
					ptit->second++;
    			}    
#endif
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	if (hostPtr >= aet_host && (size_t) hostPtr < (size_t) aet_host + aet->size) {
                	*devPtrBase = aet->basePtr;
					if( size ) *size = aet->size;
                	if (offset) *offset = (size_t) hostPtr - (size_t) aet_host;
                	return  HI_success;
            	}
        	}
		}
#else
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	tAddressMap->lower_bound(hostPtr);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        if(it2 != tAddressMap->end() ) {
			if( it2->first == hostPtr ) {
				//found the entry matching the key, hostPtr.
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	*devPtrBase = aet->basePtr;
				if( size ) *size = aet->size;
				if( offset ) *offset = 0;
				return HI_success;
			} else {
				//hostPtr may belong to an entry before the current one.
				if( it2 == tAddressMap->begin() ) {
					//There is no entry before the current one.
					//return NULL; //Do not return here; check the default stream.
				} else {
					--it2; 
            		const void* aet_hostPtr = it2->first;
            		addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            		if (hostPtr >= aet_hostPtr && (size_t) hostPtr < (size_t) aet_hostPtr + aet->size) {
                		*devPtrBase = aet->basePtr;
						if( size ) *size = aet->size;
                		if (offset) *offset = (size_t) hostPtr - (size_t) aet_hostPtr;
                		return  HI_success;
            		}
				}
			}
		} else if( !tAddressMap->empty() ) {
			//hostPtr may belong to the last entry.
        	addressmap_t::reverse_iterator it3 = tAddressMap->rbegin();
            const void* aet_hostPtr = it3->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it3->second;
            if (hostPtr >= aet_hostPtr && (size_t) hostPtr < (size_t) aet_hostPtr + aet->size) {
                *devPtrBase = aet->basePtr;
				if( size ) *size = aet->size;
                if (offset) *offset = (size_t) hostPtr - (size_t) aet_hostPtr;
                return  HI_success;
            }
		}

		if( asyncID != defaultAsyncID ) {
        	//check on the default stream
        	it = masterAddressTable->find(defaultAsyncID);
        	if(it == masterAddressTable->end() ) {
            	addressmap_t* emptyMap = new addressmap_t();
            	masterAddressTable->insert(std::pair<int, addressmap_t *> (defaultAsyncID, emptyMap));
				it = masterAddressTable->find(defaultAsyncID);
			}
			tAddressMap = it->second;
        	addressmap_t::iterator it2 =	tAddressMap->lower_bound(hostPtr);
#ifdef _OPENARC_PROFILE_
    		if( HI_openarcrt_verbosity > 1 ) {
				ptit->second++;
    		}    
#endif
        	if(it2 != tAddressMap->end() ) {
				if( it2->first == hostPtr ) {
					//found the entry matching the key, hostPtr.
            		addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            		*devPtrBase = aet->basePtr;
					if( size ) *size = aet->size;
					if( offset ) *offset = 0;
					return HI_success;
				} else {
					//hostPtr may belong to an entry before the current one.
					if( it2 == tAddressMap->begin() ) {
						//There is no entry before the current one.
						*devPtrBase = NULL;
						if( size ) *size = 0;
						if( offset ) *offset = 0;
        				return HI_error;
					} else {
						--it2; 
            			const void* aet_hostPtr = it2->first;
            			addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            			if (hostPtr >= aet_hostPtr && (size_t) hostPtr < (size_t) aet_hostPtr + aet->size) {
                			*devPtrBase = aet->basePtr;
							if( size ) *size = aet->size;
                			if (offset) *offset = (size_t) hostPtr - (size_t) aet_hostPtr;
                			return  HI_success;
            			}
					}
				}
			} else if( !tAddressMap->empty() ) {
				//hostPtr may belong to the last entry.
        		addressmap_t::reverse_iterator it3 = tAddressMap->rbegin();
            	const void* aet_hostPtr = it3->first;
            	addresstable_entity_t *aet = (addresstable_entity_t*) it3->second;
            	if (hostPtr >= aet_hostPtr && (size_t) hostPtr < (size_t) aet_hostPtr + aet->size) {
                	*devPtrBase = aet->basePtr;
					if( size ) *size = aet->size;
                	if (offset) *offset = (size_t) hostPtr - (size_t) aet_hostPtr;
                	return  HI_success;
            	}
			}
		}
#endif
        //fprintf(stderr, "[ERROR in get_device_address()] No mapping found for the host pointer\n");
		*devPtrBase = NULL;
		if( size ) *size = 0;
		if( offset ) *offset = 0;
        return HI_error;
    }

    HI_error_t HI_set_device_address(const void *hostPtr, void * devPtr, size_t size, int asyncID, int tid) {
    	addresstable_t *masterAddressTable = masterAddressTableMap[tid];
        addresstable_t::iterator it = masterAddressTable->find(asyncID);
        //fprintf(stderr, "[in set_device_address()] Setting address\n");
        if(it == masterAddressTable->end() ) {
            //fprintf(stderr, "[in set_device_address()] No mapping found for the asyncID\n");
            addressmap_t *emptyMap = new addressmap_t();
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
			(*emptyMap)[hostPtr] = (void *) aet;
			//emptyMap->insert(std::pair<const void*, void*>(hostPtr, (void *) aet));
            masterAddressTable->insert(std::pair<int, addressmap_t *> (asyncID, emptyMap));
            //it = masterAddressTable->find(asyncID);
        } else {
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
        	(*(it->second))[hostPtr] = (void*) aet;
        	//(it->second)->insert(std::pair<const void*, void*>(hostPtr, (void*) aet));
		}
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        return  HI_success;
    }

    HI_error_t HI_remove_device_address(const void *hostPtr, int asyncID, int tid) {
    	addresstable_t *masterAddressTable = masterAddressTableMap[tid];
        addresstable_t::iterator it = masterAddressTable->find(asyncID);
        addressmap_t::iterator it2 =	(it->second)->find(hostPtr);

#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        if(it2 != (it->second)->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            delete aet;
            (it->second)->erase(it2);
            return  HI_success;
        } else {
            fprintf(stderr, "[ERROR in remove_device_address()] No mapping found for the host pointer on async ID %d\n", asyncID);
            return HI_error;
        }
    }

    void HI_print_device_address_mapping_summary(int tid) {
    	addresstable_t *masterAddressTable = masterAddressTableMap[tid];
		memPool_t *memPool = memPoolMap[tid];
		size_t num_table_entries = 0;
		size_t total_allocated_device_memory = 0;
        for (addresstable_t::iterator it = masterAddressTable->begin(); it != masterAddressTable->end(); ++it) {
			addressmap_t *tAddressMap = it->second;
        	for (addressmap_t::iterator it2 = tAddressMap->begin(); it2 != tAddressMap->end(); ++it2) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
				total_allocated_device_memory += aet->size;
				num_table_entries++;
			}
		}
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\tSummary of host-to-device-address mapping table for host thread %d\n", tid);
        fprintf(stderr, "                \t\t\tNumber of mapping entries = %lu\n", num_table_entries);
        fprintf(stderr, "                \t\t\tTotal allocated device memory = %lu\n", total_allocated_device_memory);
		num_table_entries = 0;
		total_allocated_device_memory = 0;
        for (memPool_t::iterator it = memPool->begin(); it != memPool->end(); ++it) {
			total_allocated_device_memory += it->first * memPool->count(it->first);
			num_table_entries++;
		}
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\tSummary of device-memory pool table for host thread %d\n", tid);
        fprintf(stderr, "                \t\t\tNumber of mapping entries = %lu\n", num_table_entries);
        fprintf(stderr, "                \t\t\tTotal reserved device memory pool = %lu\n", total_allocated_device_memory);
    }

    void HI_print_device_address_mapping_entries(int tid) {
    	addresstable_t *masterAddressTable = masterAddressTableMap[tid];
		memPool_t *memPool = memPoolMap[tid];
    	addressmap_t *myHandleMap = masterHandleTable[tid];
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\tHost-to-device-address mapping table entries for host thread %d\n", tid);
        fprintf(stderr, "                \t\t\tHostPtr\tDevPtr\n");
        for (addresstable_t::iterator it = masterAddressTable->begin(); it != masterAddressTable->end(); ++it) {
			addressmap_t *tAddressMap = it->second;
        	for (addressmap_t::iterator it2 = tAddressMap->begin(); it2 != tAddressMap->end(); ++it2) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
        		fprintf(stderr, "                \t\t\t%lx\t%lx\n", (unsigned long)it2->first, (unsigned long)aet->basePtr);
			}
		}
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\tDevPtr-to-MemHandle mapping table entries for host thread %d\n", tid);
        fprintf(stderr, "                \t\t\tDevPtr\tMemHandle\n");
		for(addressmap_t::iterator it = myHandleMap->begin(); it != myHandleMap->end(); ++it ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it->second;
        	fprintf(stderr, "                \t\t\t%lx\t%lx\n", (unsigned long)it->first, (unsigned long)aet->basePtr);
		}
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\tDevPtr in the memory pool for host thread %d\n", tid);
        fprintf(stderr, "                \t\t\tDevPtr\n");
        for (memPool_t::iterator it = memPool->begin(); it != memPool->end(); ++it) {
        	fprintf(stderr, "                \t\t\t%lx\n", (unsigned long)it->second);
		}
    }

    HI_error_t HI_get_host_address(const void *devPtr, void** hostPtr, int asyncID, int tid) {
    	addresstable_t *masterAddressTable = masterAddressTableMap[tid];
        int defaultAsyncID = DEFAULT_QUEUE+tid*MAX_NUM_QUEUES_PER_THREAD;
        addresstable_t::iterator it = masterAddressTable->find(asyncID);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
    	}    
#endif
        if(it != masterAddressTable->end() ) {
			addressmap_t *tAddressMap = it->second;
			for( addressmap_t::iterator it3 = tAddressMap->begin(); it3 != tAddressMap->end(); ++it3 ) {
#ifdef _OPENARC_PROFILE_
    			if( HI_openarcrt_verbosity > 1 ) {
					ptit->second++;
    			}    
#endif
            	addresstable_entity_t *aet = (addresstable_entity_t*) it3->second;
				if( aet->basePtr == devPtr ) {
					*hostPtr = (void *)it3->first;
					return HI_success;
				} else if (devPtr >= aet->basePtr && (size_t) devPtr < (size_t) aet->basePtr + aet->size) {
                	*hostPtr = (void*) ((size_t) it3->first + ((size_t) devPtr - (size_t) aet->basePtr));
					return HI_success;
            	}
			}
		}

		if( asyncID != defaultAsyncID ) {
			//Check default queue.
        	addresstable_t::iterator it = masterAddressTable->find(defaultAsyncID);
        	if(it != masterAddressTable->end() ) {
				addressmap_t *tAddressMap = it->second;
				for( addressmap_t::iterator it3 = tAddressMap->begin(); it3 != tAddressMap->end(); ++it3 ) {
#ifdef _OPENARC_PROFILE_
    				if( HI_openarcrt_verbosity > 1 ) {
						ptit->second++;
    				}    
#endif
            		addresstable_entity_t *aet = (addresstable_entity_t*) it3->second;
					if( aet->basePtr == devPtr ) {
						*hostPtr = (void *)it3->first;
						return HI_success;
					} else if (devPtr >= aet->basePtr && (size_t) devPtr < (size_t) aet->basePtr + aet->size) {
                		*hostPtr = (void*) ((size_t) it3->first + ((size_t) devPtr - (size_t) aet->basePtr));
						return HI_success;
            		}
				}
			}
		}

		*hostPtr = NULL;
		return HI_error;
    }

    const void * HI_get_base_address_of_host_memory(const void *hostPtr, int asyncID, int tid) {
    	addresstable_t *masterAddressTable = masterAddressTableMap[tid];
		addresstable_t::iterator it = masterAddressTable->find(asyncID);
        int defaultAsyncID = DEFAULT_QUEUE+tid*MAX_NUM_QUEUES_PER_THREAD;
	
        if(it == masterAddressTable->end() ) {
            addressmap_t* emptyMap = new addressmap_t();
            masterAddressTable->insert(std::pair<int, addressmap_t *> (asyncID, emptyMap));
			it = masterAddressTable->find(asyncID);
		}
		addressmap_t *tAddressMap = it->second;
#if PRESENT_TABLE_SEARCH_MODE == 0
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	tAddressMap->find(hostPtr);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
    	}    
#endif
        if(it2 != tAddressMap->end() ) {
            return  hostPtr;
        } else {
            //check on the default stream
            it = masterAddressTable->find(defaultAsyncID);
        	if(it == masterAddressTable->end() ) {
            	addressmap_t* emptyMap = new addressmap_t();
            	masterAddressTable->insert(std::pair<int, addressmap_t *> (defaultAsyncID, emptyMap));
				it = masterAddressTable->find(defaultAsyncID);
			}
			tAddressMap = it->second;
            it2 =	tAddressMap->find(hostPtr);
#ifdef _OPENARC_PROFILE_
    		if( HI_openarcrt_verbosity > 1 ) {
				ptit->second++;
    		}    
#endif
            if(it2 != tAddressMap->end() ) {
            	return  hostPtr;
            }
		}

		//Check whether hostPtr is within the range of an allocated memory region 
		//in the addressTable.
		for (addressmap_t::iterator it2 = tAddressMap->begin(); it2 != tAddressMap->end(); ++it2) {
#ifdef _OPENARC_PROFILE_
    		if( HI_openarcrt_verbosity > 1 ) {
				ptit->second++;
    		}    
#endif
            const void* aet_host = it2->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            if (hostPtr >= aet_host && (size_t) hostPtr < (size_t) aet_host + aet->size) {
                return  aet_host;
            }
        }

		if( asyncID != defaultAsyncID ) {
        	//check on the default stream
        	it = masterAddressTable->find(defaultAsyncID);
			tAddressMap = it->second;
        	for (addressmap_t::iterator it2 = tAddressMap->begin(); it2 != tAddressMap->end(); ++it2) {
#ifdef _OPENARC_PROFILE_
    			if( HI_openarcrt_verbosity > 1 ) {
					ptit->second++;
    			}    
#endif
            	const void* aet_host = it2->first;
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	if (hostPtr >= aet_host && (size_t) hostPtr < (size_t) aet_host + aet->size) {
                	return  aet_host;
            	}
        	}
		}
#else
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	tAddressMap->lower_bound(hostPtr);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        if(it2 != tAddressMap->end() ) {
			if( it2->first == hostPtr ) {
				//found the entry matching the key, hostPtr.
				return hostPtr;
			} else {
				//hostPtr may belong to an entry before the current one.
				if( it2 == tAddressMap->begin() ) {
					//There is no entry before the current one.
					//return NULL; //Do not return here; check the default stream.
				} else {
					--it2; 
            		const void* aet_hostPtr = it2->first;
            		addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            		if (hostPtr >= aet_hostPtr && (size_t) hostPtr < (size_t) aet_hostPtr + aet->size) {
						return aet_hostPtr;
            		}
				}
			}
		} else if( !tAddressMap->empty() ) {
			//hostPtr may belong to the last entry.
        	addressmap_t::reverse_iterator it3 = tAddressMap->rbegin();
            const void* aet_hostPtr = it3->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it3->second;
            if (hostPtr >= aet_hostPtr && (size_t) hostPtr < (size_t) aet_hostPtr + aet->size) {
				return aet_hostPtr;
            }
		}

		if( asyncID != defaultAsyncID ) {
        	//check on the default stream
        	it = masterAddressTable->find(defaultAsyncID);
        	if(it == masterAddressTable->end() ) {
            	addressmap_t* emptyMap = new addressmap_t();
            	masterAddressTable->insert(std::pair<int, addressmap_t *> (defaultAsyncID, emptyMap));
				it = masterAddressTable->find(defaultAsyncID);
			}
			tAddressMap = it->second;
        	addressmap_t::iterator it2 =	tAddressMap->lower_bound(hostPtr);
#ifdef _OPENARC_PROFILE_
    		if( HI_openarcrt_verbosity > 1 ) {
				ptit->second++;
    		}    
#endif
        	if(it2 != tAddressMap->end() ) {
				if( it2->first == hostPtr ) {
					//found the entry matching the key, hostPtr.
					return hostPtr;
				} else {
					//hostPtr may belong to an entry before the current one.
					if( it2 == tAddressMap->begin() ) {
						//There is no entry before the current one.
						return NULL; 
					} else {
						--it2; 
            			const void* aet_hostPtr = it2->first;
            			addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            			if (hostPtr >= aet_hostPtr && (size_t) hostPtr < (size_t) aet_hostPtr + aet->size) {
							return aet_hostPtr;
            			}
					}
				}
			} else if( !tAddressMap->empty() ) {
				//hostPtr may belong to the last entry.
        		addressmap_t::reverse_iterator it3 = tAddressMap->rbegin();
            	const void* aet_hostPtr = it3->first;
            	addresstable_entity_t *aet = (addresstable_entity_t*) it3->second;
            	if (hostPtr >= aet_hostPtr && (size_t) hostPtr < (size_t) aet_hostPtr + aet->size) {
					return aet_hostPtr;
            	}
			}
		}
#endif
		//No entry is found.
        return NULL;
    }

    HI_error_t HI_get_device_address_from_victim_cache(const void *hostPtr, void **devPtrBase, size_t *offset, size_t *size, int asyncID, int tid) {
		HI_error_t ret = HI_error;
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
    	#pragma omp critical (victim_cache_critical)
#endif
#endif
		{
		addresstable_t::iterator it = auxAddressTable.find(asyncID);
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	(it->second)->find(hostPtr);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        if(it2 != (it->second)->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            *devPtrBase = aet->basePtr;
			if( size ) *size = aet->size;
			if( offset ) *offset = 0;
            //*devPtrBase = it2->second;
            ret = HI_success;
        } else {
            //check on the default stream
            it = auxAddressTable.find(DEFAULT_QUEUE+tid*MAX_NUM_QUEUES_PER_THREAD);
            it2 =	(it->second)->find(hostPtr);
            if(it2 != (it->second)->end() ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	*devPtrBase = aet->basePtr;
				if( size ) *size = aet->size;
				if( offset ) *offset = 0;
            //	*devPtrBase = it2->second;
            	ret = HI_success;
            }
		}
		}
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
        return ret;
    }

    HI_error_t HI_set_device_address_in_victim_cache (const void *hostPtr, void * devPtr, size_t size, int asyncID) {
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
    	#pragma omp critical (victim_cache_critical)
#endif
#endif
		{
        addresstable_t::iterator it = auxAddressTable.find(asyncID);
        if(it == auxAddressTable.end() ) {
            addressmap_t *emptyMap = new addressmap_t();
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
			(*emptyMap)[hostPtr] = (void *) aet;
            auxAddressTable.insert(std::pair<int, addressmap_t*> (asyncID, emptyMap));
        } else {
        	//(it->second).insert(std::pair<const void *,void*>(hostPtr, devPtr));
        	//(it->second)[hostPtr] = devPtr;
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
        	(*(it->second))[hostPtr] = (void*) aet;
		}
		}
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
        return  HI_success;
    }

    HI_error_t HI_remove_device_address_from_victim_cache (const void *hostPtr, int asyncID) {
		HI_error_t ret;
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
    	#pragma omp critical (victim_cache_critical)
#endif
#endif
		{
        addresstable_t::iterator it = auxAddressTable.find(asyncID);
        addressmap_t::iterator it2 =	(it->second)->find(hostPtr);

        if(it2 != (it->second)->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            delete aet;
            (it->second)->erase(it2);
            ret =  HI_success;
        } else {
            fprintf(stderr, "[ERROR in remove_device_address_from_victim_cache()] No mapping found for the host pointer on async ID %d\n", asyncID);
            ret = HI_error;
        }
		}
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
		return ret;
    }

    HI_error_t HI_reset_victim_cache ( int asyncID ) {
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
    	#pragma omp critical (victim_cache_critical)
#endif
#endif
		{
        addresstable_t::iterator it = auxAddressTable.find(asyncID);
        while(it != auxAddressTable.end()) {
			for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	delete aet;
			}
			(it->second)->clear();
            it++;
        }
		}
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
		return  HI_success;
    }

    HI_error_t HI_reset_victim_cache_all ( ) {
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
    	#pragma omp critical (victim_cache_critical)
#endif
#endif
		for( addresstable_t::iterator it = auxAddressTable.begin(); it != auxAddressTable.end(); ++it ) {
			for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	delete aet;
			}
			(it->second)->clear();
		}
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
		return  HI_success;
    }

    HI_error_t HI_get_device_mem_handle(const void *devPtr, HI_device_mem_handle_t *memHandle, int tid) {
    	return HI_get_device_mem_handle(devPtr, memHandle, NULL, tid);
	}

    HI_error_t HI_get_device_mem_handle(const void *devPtr, HI_device_mem_handle_t *memHandle, size_t *size, int tid) {
    	addressmap_t *myHandleMap = masterHandleTable[tid];
#if PRESENT_TABLE_SEARCH_MODE == 0
		//Check whether devPtr exists as an entry to myHandleMap, 
		//which will be true if devPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	myHandleMap->find(devPtr);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        if(it2 != myHandleMap->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            memHandle->basePtr = aet->basePtr;
            memHandle->offset = 0;
			if( size != NULL ) {
				*size = aet->size;
			}
            return  HI_success;
		}

		//Check whether devPtr is within the range of an allocated memory region 
		//in the addressTable.
		for (addressmap_t::iterator it2 = myHandleMap->begin(); it2 != myHandleMap->end(); ++it2) {
            const void* aet_devPtr = it2->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
#ifdef _OPENARC_PROFILE_
    		if( HI_openarcrt_verbosity > 1 ) {
				ptit->second++;
    		}    
#endif
            if (devPtr >= aet_devPtr && (size_t) devPtr < (size_t) aet_devPtr + aet->size) {
                memHandle->basePtr = aet->basePtr;
                memHandle->offset = (size_t) devPtr - (size_t) aet_devPtr;
				if( size != NULL ) {
					*size = aet->size;
				}
                return  HI_success;
            }
        }
#else
		//Check whether devPtr exists as an entry to myHandleMap, 
		//which will be true if devPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	myHandleMap->lower_bound(devPtr);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        if(it2 != myHandleMap->end() ) {
			if( it2->first == devPtr ) {
				//found the entry matching the key, devPtr.
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	memHandle->basePtr = aet->basePtr;
            	memHandle->offset = 0;
				if( size != NULL ) {
					*size = aet->size;
				}
            	return  HI_success;
			} else {
				//devPtr may belong to an entry before the current one.
				if( it2 == myHandleMap->begin() ) {
					//There is no entry before the current one.
					memHandle->basePtr = NULL;
					memHandle->offset = 0;
					if( size != NULL ) {
						*size = 0;
					}
					return HI_error;
				} else {
					--it2; 
            		const void* aet_devPtr = it2->first;
            		addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            		if (devPtr >= aet_devPtr && (size_t) devPtr < (size_t) aet_devPtr + aet->size) {
                		memHandle->basePtr = aet->basePtr;
                		memHandle->offset = (size_t) devPtr - (size_t) aet_devPtr;
						if( size != NULL ) {
							*size = aet->size;
						}
                		return  HI_success;
            		}
				}
			}
		} else if( !myHandleMap->empty() ) {
			//devPtr may belong to the last entry.
        	addressmap_t::reverse_iterator it3 = myHandleMap->rbegin();
            const void* aet_devPtr = it3->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it3->second;
            if (devPtr >= aet_devPtr && (size_t) devPtr < (size_t) aet_devPtr + aet->size) {
            	memHandle->basePtr = aet->basePtr;
                memHandle->offset = (size_t) devPtr - (size_t) aet_devPtr;
				if( size != NULL ) {
					*size = aet->size;
				}
                return  HI_success;
            }
		}
#endif

        //fprintf(stderr, "[ERROR in get_device_mem_handle()] No mapping found for the device pointer\n");
		memHandle->basePtr = NULL;
		memHandle->offset = 0;
		if( size != NULL ) {
			*size = 0;
		}
        return HI_error;
    }

    HI_error_t HI_set_device_mem_handle(const void *devPtr, void * handle, size_t size, int tid) {
    	addressmap_t *myHandleMap = masterHandleTable[tid];
        //fprintf(stderr, "[in set_device_mem_handle()] Setting address\n");
        addresstable_entity_t *aet = new addresstable_entity_t(handle, size);
        (*myHandleMap)[devPtr] = (void*) aet;
        //myHandleMap->insert(std::pair<const void*, void*>(devPtr, (void*) aet));
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        return  HI_success;
    }

    HI_error_t HI_remove_device_mem_handle(const void *devPtr, int tid) {
    	addressmap_t *myHandleMap = masterHandleTable[tid];
        addressmap_t::iterator it2 = myHandleMap->find(devPtr);
#ifdef _OPENARC_PROFILE_
		presenttablecnt_t::iterator ptit = presentTableCntMap.find(tid);
    	if( HI_openarcrt_verbosity > 1 ) {
			if(ptit == presentTableCntMap.end()) {
				presentTableCntMap.insert(std::pair<int, long> (tid, 0));
			}
			ptit->second++;
    	}    
#endif
        if(it2 != myHandleMap->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            delete aet;
            myHandleMap->erase(it2);
            return  HI_success;
        } else {
            fprintf(stderr, "[ERROR in remove_device_mem_handle()] No mapping found for the device pointer\n");
            return HI_error;
        }
    }

    HI_error_t HI_free_async( const void *hostPtr, int asyncID, int tid) {
        //fprintf(stderr, "[in HI_free_async()] with asyncID %d\n", asyncID);
    	asyncfreetable_t *postponedFreeTable = postponedFreeTableMap[tid];
        postponedFreeTable->insert(std::pair<int, const void *>(asyncID, hostPtr));
        return HI_success;
    }

    HI_error_t HI_postponed_free(int asyncID, int tid) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 3 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\tenter HI_postponed_free()\n");
    }    
#endif
    	asyncfreetable_t *postponedFreeTable = postponedFreeTableMap[tid];
        std::multimap<int, const void*>::iterator hostPtrIter = postponedFreeTable->find(asyncID);

        while(hostPtrIter != postponedFreeTable->end()) {
            //fprintf(stderr, "[in HI_postponed_free()] Freeing on stream %d, address %x\n", asyncID, hostPtrIter->second);
            HI_free(hostPtrIter->second, asyncID, tid);
            hostPtrIter++;
        }

        postponedFreeTable->erase(asyncID);
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 3 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\texit HI_postponed_free()\n");
    }    
#endif
        return HI_success;
    }

    HI_error_t HI_tempFree_async( void **tempPtr, acc_device_t devType, int asyncID, int tid) {
        //fprintf(stderr, "[in HI_tempFree_async()] with asyncID %d\n", asyncID);
		if( postponedTempFreeTableMap.count(tid) == 0 ) {
        	fprintf(stderr, "[ERROR in HI_tempFree_async()] No mapping found for thread ID = %d\n", tid);
		} else {
    		asynctempfreetable_t *postponedTempFreeTable = postponedTempFreeTableMap[tid];
        	postponedTempFreeTable->insert(std::pair<int, void **>(asyncID, tempPtr));
    		asynctempfreetable2_t *postponedTempFreeTable2 = postponedTempFreeTableMap2[tid];
        	postponedTempFreeTable2->insert(std::pair<int, acc_device_t>(asyncID, devType));
		}
        return HI_success;
    }

    HI_error_t HI_postponed_tempFree(int asyncID, acc_device_t devType, int tid) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 3 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\tenter HI_postponed_tempFree(devType = %d, thread ID = %d)\n", devType, tid);
    }    
#endif
		if( postponedTempFreeTableMap.count(tid) == 0 ) {
        	fprintf(stderr, "[ERROR in HI_postponed_tempFree()] No mapping found for thread ID = %d\n", tid);
		} else {
    		asynctempfreetable_t *postponedTempFreeTable = postponedTempFreeTableMap[tid];
        	std::multimap<int, void**>::iterator tempPtrIter = postponedTempFreeTable->find(asyncID);
    		asynctempfreetable2_t *postponedTempFreeTable2 = postponedTempFreeTableMap2[tid];
        	std::multimap<int, acc_device_t>::iterator tempPtrIter2 = postponedTempFreeTable2->find(asyncID);

        	while((tempPtrIter != postponedTempFreeTable->end()) && (tempPtrIter2 != postponedTempFreeTable2->end())) {
            	//fprintf(stderr, "[in HI_postponed_TempFree()] Freeing on stream %d, address %x\n", asyncID, tempPtrIter->second);
            	HI_tempFree(tempPtrIter->second, tempPtrIter2->second, tid);
            	tempPtrIter++;
            	tempPtrIter2++;
        	}
        	if(tempPtrIter != postponedTempFreeTable->end()) {
        		fprintf(stderr, "[ERROR in HI_postponed_tempFree()] postponedTempFreeTable has more entries for thread ID = %d\n", tid);
			}
        	if(tempPtrIter2 != postponedTempFreeTable2->end()) {
        		fprintf(stderr, "[ERROR in HI_postponed_tempFree()] postponedTempFreeTable2 has more entries for thread ID = %d\n", tid);
			}

        	postponedTempFreeTable->erase(asyncID);
        	postponedTempFreeTable2->erase(asyncID);
		}
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 3 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\texit HI_postponed_tempFree(devType = %d, thread ID = %d)\n", devType, tid);
    }    
#endif
        return HI_success;
    }

    HI_error_t HI_get_temphost_address(const void *hostPtr, void **temphostPtr, int asyncID, int tid) {
        addresstable_t::iterator it = tempHostAddressTable.find(asyncID);
        addressmap_t::iterator it2 =	(it->second)->find(hostPtr);
        if(it2 != (it->second)->end() ) {
            *temphostPtr = it2->second;
            return  HI_success;
        } else {
            //check on the default stream
            it = tempHostAddressTable.find(DEFAULT_QUEUE+tid*MAX_NUM_QUEUES_PER_THREAD);
            it2 =	(it->second)->find(hostPtr);
            if(it2 != (it->second)->end() ) {
                *temphostPtr = it2->second;
                return  HI_success;
            }
            //fprintf(stderr, "[ERROR in get_temphost_address()] No mapping found for the host pointer\n");
            return HI_error;
        }
    }

    HI_error_t HI_set_temphost_address(const void *hostPtr, void * temphostPtr, int asyncID) {
        addresstable_t::iterator it = tempHostAddressTable.find(asyncID);
        //fprintf(stderr, "[in set_temphost_address()] Setting address\n");
        if(it == tempHostAddressTable.end() ) {
            //fprintf(stderr, "[in set_temphost_address()] No mapping found for the asyncID\n");
            addressmap_t * emptyMap = new addressmap_t();
            tempHostAddressTable.insert(std::pair<int, addressmap_t*> (asyncID, emptyMap));
            it = tempHostAddressTable.find(asyncID);
        }

        //(it->second).insert(std::pair<const void *,void*>(hostPtr, temphostPtr));
        (*(it->second))[hostPtr] = temphostPtr;
        return  HI_success;
    }

    HI_error_t HI_remove_temphost_address(const void *hostPtr, int asyncID) {
        addresstable_t::iterator it = tempHostAddressTable.find(asyncID);
		if( it != tempHostAddressTable.end() ) {
        	addressmap_t::iterator it2 =	(it->second)->find(hostPtr);
        	if(it2 != (it->second)->end() ) {
            	(it->second)->erase(it2);
            	return  HI_success;
        	} else {
            	fprintf(stderr, "[ERROR in remove_temphost_address()] No mapping found for the host pointer on async ID %d\n", asyncID);
            	return HI_error;
        	}
		} else {
           fprintf(stderr, "[ERROR in remove_temphost_address()] No mapping found for the host pointer on async ID %d\n", asyncID);
           return HI_error;
		}
    }

    void HI_free_temphosts(int asyncID ) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 3 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\tenter HI_free_temphosts()\n");
    }    
#endif
        addresstable_t::iterator it = tempHostAddressTable.find(asyncID);
		if (it != tempHostAddressTable.end()) {
			for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
				HI_tempFree(&(it2->second), acc_device_host);
			}
			(it->second)->clear();
		}
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 3 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\t\texit HI_free_temphosts()\n");
    }    
#endif
    }

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

} Accelerator_t;

///////////////////////////////////////////////////
// Overloaded OpenACC runtime API from openacc.h //
///////////////////////////////////////////////////
///////////////////////////////////////////
// OpenACC V1.0 Runtime Library Routines //
///////////////////////////////////////////
extern void acc_init( acc_device_t devtype, int kernels, std::string kernelNames[], const char *fileNameBase = "openarc_kernel", int threadID=NO_THREAD_ID);
extern int acc_get_num_devices( acc_device_t devtype, int threadID );
extern void acc_set_device_type( acc_device_t devtype, int threadID );
extern acc_device_t acc_get_device_type(int threadID);
extern void acc_set_device_num( int devnum, acc_device_t devtype, int threadID );
extern int acc_get_device_num( acc_device_t devtype, int threadID );
extern int acc_async_test( int asyncID, int threadID );
extern int acc_async_test_all(int threadID);
extern void acc_async_wait( int asyncID, int threadID ); //renamed to acc_wait()
extern void acc_async_wait_all(int threadID); //renamed to acc_wait_all()
extern void acc_shutdown( acc_device_t devtype, int threadID );
extern int acc_on_device( acc_device_t devtype, int threadID );
extern d_void* acc_malloc(size_t size, int threadID);
extern void acc_free(d_void* devPtr, int threadID);

/////////////////////////////////////////////////////////// 
// OpenACC Runtime Library Routines added in Version 2.0 //
/////////////////////////////////////////////////////////// 
//acc_async_wait() and acc_async_wait_all() are renamed to acc_wait() and
//acc_wait_all() in V2.0.
extern void acc_wait( int arg, int threadID );
extern void acc_wait_all(int threadID);
extern void acc_wait_async(int arg, int async, int threadID);
extern void acc_wait_all_async(int async, int threadID);
extern void* acc_copyin(h_void* hostPtr, size_t size, int threadID);
extern void* acc_copyin_async(h_void* hostPtr, size_t size, int async, int threadID);
extern void* acc_pcopyin(h_void* hostPtr, size_t size, int threadID);
extern void* acc_present_or_copyin(h_void* hostPtr, size_t size, int threadID);
extern void* acc_create(h_void* hostPtr, size_t size, int threadID);
extern void* acc_create_async(h_void* hostPtr, size_t size, int async, int threadID);
extern void* acc_pcreate(h_void* hostPtr, size_t size, int threadID);
extern void* acc_present_or_create(h_void* hostPtr, size_t size, int threadID);
extern void acc_copyout(h_void* hostPtr, size_t size, int threadID);
extern void acc_copyout_async(h_void* hostPtr, size_t size, int async, int threadID);
extern void acc_delete(h_void* hostPtr, size_t size, int threadID);
extern void acc_delete_async(h_void* hostPtr, size_t size, int async, int threadID);
extern void acc_update_device(h_void* hostPtr, size_t size, int threadID);
extern void acc_update_device_async(h_void* hostPtr, size_t size, int async, int threadID);
extern void acc_update_self(h_void* hostPtr, size_t size, int threadID);
extern void acc_update_self_async(h_void* hostPtr, size_t size, int async, int threadID);
extern void acc_map_data(h_void* hostPtr, d_void* devPtr, size_t size, int threadID);
extern void acc_unmap_data(h_void* hostPtr, int threadID);
extern d_void* acc_deviceptr(h_void* hostPtr, int threadID);
extern h_void* acc_hostptr(d_void* devPtr, int threadID);
extern int acc_is_present(h_void* hostPtr, size_t size, int threadID);
extern void acc_memcpy_to_device(d_void* dest, h_void* src, size_t bytes, int threadID);
extern void acc_memcpy_from_device(h_void* dest, d_void* src, size_t bytes, int threadID);

/////////////////////////////////////////////////////////// 
// OpenACC Runtime Library Routines added in Version 2.5 //
/////////////////////////////////////////////////////////// 
extern void acc_memcpy_device(d_void* dest, d_void* src, size_t bytes, int threadID);
extern void acc_memcpy_device_async(d_void* dest, d_void* src, size_t bytes, int async, int threadID);

/////////////////////////////////////////////////////////// 
// OpenACC Runtime Library Routines added in Version 2.6 //
/////////////////////////////////////////////////////////// 
extern void acc_attach(h_void** hostPtr, int threadID);
extern void acc_attach_async(h_void** hostPtr, int async, int threadID);
extern void acc_detach(h_void** hostPtr, int threadID);
extern void acc_detach_async(h_void** hostPtr, int async, int threadID);
extern void acc_detach_finalize(h_void** hostPtr, int threadID);
extern void acc_detach_finalize_async(h_void** hostPtr, int async, int threadID);
//extern size_t acc_get_property(int devicenum, acc_device_t devicetype, acc_device_property_t property, int threadID);
//extern const char* acc_get_property_string(int devicenum, acc_device_t devicetype, acc_device_property_t property, int threadID);

//////////////////////////////////////////////////////////////////////
// Experimental OpenACC Runtime Library Routines for Unified Memory //
// (Currently, these work only for specific versions of CUDA GPUs.) //
//////////////////////////////////////////////////////////////////////
extern void* acc_copyin_unified(h_void* hostPtr, size_t size, int threadID);
extern void* acc_pcopyin_unified(h_void* hostPtr, size_t size, int threadID);
extern void* acc_present_or_copyin_unified(h_void* hostPtr, size_t size, int threadID);
extern void* acc_create_unified(h_void* hostPtr, size_t size, int threadID);
extern void* acc_pcreate_unified(h_void* hostPtr, size_t size, int threadID);
extern void* acc_present_or_create_unified(h_void* hostPtr, size_t size, int threadID);
extern void acc_copyout_unified(h_void* hostPtr, size_t size, int threadID);
extern void acc_delete_unified(h_void* hostPtr, size_t size, int threadID);

/////////////////////////////////////////////////////////////////
// Additional OpenACC Runtime Library Routines Used by OpenARC //
/////////////////////////////////////////////////////////////////
extern void* acc_copyin_const(h_void* hostPtr, size_t size, int threadID);
extern void* acc_pcopyin_const(h_void* hostPtr, size_t size, int threadID);
extern void* acc_present_or_copyin_const(h_void* hostPtr, size_t size, int threadID);
extern void* acc_create_const(h_void* hostPtr, size_t size, int threadID);
extern void* acc_pcreate_const(h_void* hostPtr, size_t size, int threadID);
extern void* acc_present_or_create_const(h_void* hostPtr, size_t size, int threadID);
extern void* acc_copyin_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID);
extern void* acc_pcopyin_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID);
extern void* acc_present_or_copyin_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID);
extern void* acc_create_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID);
extern void* acc_pcreate_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID);
extern void* acc_present_or_create_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID);
extern void acc_copyout_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID);
extern void acc_delete_async_wait(h_void* hostPtr, size_t size, int async, int arg, int threadID);

////////////////////////
// Runtime init/reset //
////////////////////////
extern void HI_hostinit(int threadID);

//////////////////////
// Kernel Execution //
//////////////////////
//Set the number of arguments to be passed to a kernel.
extern HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args, int threadID=NO_THREAD_ID);
//Register an argument to be passed to a kernel.
extern HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value, int arg_type, int threadID=NO_THREAD_ID);
//Launch a kernel.
extern HI_error_t HI_kernel_call(std::string kernel_name, size_t gridSize[3], size_t blockSize[3], int async=DEFAULT_QUEUE, int num_waits=0, int *waits=NULL, int threadID=NO_THREAD_ID);
extern HI_error_t HI_synchronize( int forcedSync = 0, int threadID=NO_THREAD_ID);

/////////////////////////////
//Device Memory Allocation //
/////////////////////////////
extern HI_error_t HI_malloc1D( const void *hostPtr, void** devPtr, size_t count, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE, int threadID=NO_THREAD_ID);
extern HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE, int threadID=NO_THREAD_ID);
extern HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE, int threadID=NO_THREAD_ID);
extern HI_error_t HI_free( const void *hostPtr, int asyncID, int threadID=NO_THREAD_ID);
extern HI_error_t HI_free_async( const void *hostPtr, int asyncID, int threadID=NO_THREAD_ID);
extern void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int threadID=NO_THREAD_ID);
extern void HI_tempFree( void** tempPtr, acc_device_t devType, int threadID=NO_THREAD_ID);
extern void HI_tempFree_async( void** tempPtr, acc_device_t devType, int asyncID, int threadID=NO_THREAD_ID);

/////////////////////////////////////////////////
//Memory transfers between a host and a device //
/////////////////////////////////////////////////
extern HI_error_t HI_memcpy(void *dst, const void *src, size_t count,
                                  HI_MemcpyKind_t kind, int trType, int threadID=NO_THREAD_ID);
extern HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count,
                                        HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID=NO_THREAD_ID);
extern HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count,
                                        HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID=NO_THREAD_ID);
extern HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                                    size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int threadID=NO_THREAD_ID);
extern HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async, int num_waits, int *waits, int threadID=NO_THREAD_ID);
//extern HI_error_t HI_memcpy3D(void *dst, size_t dpitch, const void *src, size_t spitch,
//	size_t widthInBytes, size_t height, size_t depth, HI_MemcpyKind_t kind, int threadID=NO_THREAD_ID);
//extern HI_error_t HI_memcpy3D_async(void *dst, size_t dpitch, const void *src,
//	size_t spitch, size_t widthInBytes, size_t height, size_t depth,
//	HI_MemcpyKind_t kind, int async, int num_waits=0, int *waits=NULL, int threadID=NO_THREAD_ID);
extern HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID=NO_THREAD_ID);
extern HI_error_t HI_memcpy_const_async(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int async, int num_waits, int *waits, int threadID=NO_THREAD_ID);
extern HI_error_t HI_present_or_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID=NO_THREAD_ID);

////////////////////////////////////////////////
// Experimental API to support unified memory //
////////////////////////////////////////////////
extern HI_error_t HI_malloc1D_unified( const void *hostPtr, void** devPtr, size_t count, int asyncID, HI_MallocKind_t flags, int threadID=NO_THREAD_ID);
extern HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count,
                                  HI_MemcpyKind_t kind, int trType, int threadID=NO_THREAD_ID);
extern HI_error_t HI_free_unified( const void *hostPtr, int asyncID, int threadID=NO_THREAD_ID);

////////////////////////////
//Internal mapping tables //
////////////////////////////
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtr, int asyncID, int threadID=NO_THREAD_ID);
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtrBase, size_t * offset, int asyncID, int threadID=NO_THREAD_ID);
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtrBase, size_t * offset, size_t * size, int asyncID, int threadID=NO_THREAD_ID);
extern HI_error_t HI_set_device_address(const void * hostPtr, void * devPtr, size_t size, int asyncID, int threadID=NO_THREAD_ID);
extern HI_error_t HI_remove_device_address(const void * hostPtr, int asyncID, int threadID=NO_THREAD_ID);
extern HI_error_t HI_get_host_address(const void *devPtr, void** hostPtr, int asyncID, int threadID=NO_THREAD_ID);
extern HI_error_t HI_get_temphost_address(const void * hostPtr, void ** temphostPtr, int asyncID, int threadID=NO_THREAD_ID);
//extern HI_error_t HI_set_temphost_address(const void * hostPtr, void * temphostPtr, int asyncID, int threadID=NO_THREAD_ID);
//extern HI_error_t HI_remove_temphost_address(const void * hostPtr, int threadID=NO_THREAD_ID);
//Get and increase an internal reference counter of the present table mapping for the host variable. (It also returns the corresponding device pointer.)
extern int HI_getninc_prtcounter(const void * hostPtr, void **devPtr, int asyncID, int threadID=NO_THREAD_ID);
//Decrease and get an internal reference counter of the present table mapping for the host variable. (It also returns the corresponding device pointer.)
extern int HI_decnget_prtcounter(const void * hostPtr, void **devPtr, int asyncID, int threadID=NO_THREAD_ID);

/////////////////////////////////////////////////////////////////////////
//async integer argument => internal handler (ex: CUDA stream) mapping //
/////////////////////////////////////////////////////////////////////////
//extern HI_error_t HI_create_async_handle( int async, int threadID=NO_THREAD_ID);
//extern int HI_contain_async_handle( int async , int threadID=NO_THREAD_ID);
//extern HI_error_t HI_delete_async_handle( int async , int threadID=NO_THREAD_ID);
extern void HI_set_async(int asyncId, int threadID=NO_THREAD_ID);
extern void HI_set_context(int threadID=NO_THREAD_ID);
////////////////////////////////
//Memory management functions //
////////////////////////////////
extern void HI_check_read(const void * hostPtr, acc_device_t dtype, const char *varName, const char *refName, int loopIndex, int threadID=NO_THREAD_ID);
extern void HI_check_write(const void * hostPtr, acc_device_t dtype, const char *varName, const char *refName, int loopIndex, int threadID=NO_THREAD_ID);
extern void HI_set_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, const char * varName, const char * refName, int loopIndex, int threadID=NO_THREAD_ID);
extern void HI_reset_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, int asyncID, int threadID=NO_THREAD_ID);
//Below is deprecated
extern void HI_init_status(const void * hostPtr, int threadID=NO_THREAD_ID);

////////////////////
//Texture function //
////////////////////
extern HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size, int threadID=NO_THREAD_ID);

////////////////////
//Misc. functions //
////////////////////
extern double HI_get_localtime();
extern const char* HI_get_device_type_string( acc_device_t devtype );


////////////////////////////////////////////
//Functions used for program verification //
////////////////////////////////////////////
extern void HI_waitS1(int asyncId, int threadID=NO_THREAD_ID);
extern void HI_waitS2(int asyncId, int threadID=NO_THREAD_ID);

///////////////////////////////////////////
//Functions used for OpenMP4 translation //
///////////////////////////////////////////
#include "omp_helper.h"

///////////////////////////////////////
//Functions used for resilience test //
///////////////////////////////////////
#include "resilience.h"



#endif
