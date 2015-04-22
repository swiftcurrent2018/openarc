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

#ifdef _OPENMP
#include <omp.h>
#endif

#define DEFAULT_QUEUE -2
#define DEFAULT_ASYNC_QUEUE -1
#define DEVICE_NUM_UNDEFINED -1

//VICTIM_CACHE_MODE = 0 
//  - Victim cache stores freed device memory
//  - The freed device memory can be reused if the size matches.
//  - More applicable, but host memory should be pinned again.
//VICTIM_CACHE_MODE = 1
//  - Victim cache stores both pinned host memory and corresponding device memory.
//  - The pinned host memory and device memory are reused only for the same 
//  host pointer; saving both memory pinning cost and device memory allocation cost 
//  but less applicable.
//  - Too much prepinning can cause slowdown or crash of the program.
//  - If OPENARCRT_PREPINHOSTMEM is set to 0, reuse only device memory.
//
//For OpenCL devices, VICTIM_CACHE_MODE = 0 is always applied.
#define VICTIM_CACHE_MODE 0

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
    HI_notstale = 0,
    HI_stale = 1,
    HI_maystale = 2
} HI_memstatus_t;

typedef enum {
    HI_int = 0,
    HI_float = 1,
} HI_datatype_t;

typedef struct _addresstable_entity_t {
    void* devPtr;
    size_t size;
    _addresstable_entity_t(void* _devPtr, size_t _size) : devPtr(_devPtr), size(_size) {}
} addresstable_entity_t;

typedef std::multimap<int, std::map<const void *, void *> > addresstable_t;
typedef std::multimap<int, const void *> asyncfreetable_t;
typedef std::set<const void *> pointerset_t;

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

    //Host-device address mapping table, augmented with stream id
    addresstable_t masterAddressTable;

    //Auxiliary Host-device address mapping table used as a victim cache. 
    addresstable_t auxAddressTable;

	//temporarily allocated memory set.
	pointerset_t tempMallocSet;
    
    //Host-TempHost address mapping table, augmented with stream id
    addresstable_t tempHostAddressTable;

    //This table can have duplicate entries, owing to the HI_free_async
    //calls in a loop. To handle this, HI_free ensures that on a duplicate
    //pair, no free operation is performed
    asyncfreetable_t postponedFreeTable;
	std::multimap<size_t, void *> memPool;

	virtual ~Accelerator() {};

    // Kernel Initialization
    virtual HI_error_t init() = 0;
    virtual HI_error_t destroy()=0;

    // Kernel Execution
    virtual HI_error_t HI_register_kernels(std::vector<std::string>kernelNames) = 0;
    virtual HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args) = 0;
    virtual HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value) = 0;
    virtual HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE) = 0;
    virtual HI_error_t HI_synchronize( )=0;

    // Memory Allocation
    virtual HI_error_t HI_malloc1D(const void *hostPtr, void **devPtr, int count, int asyncID)= 0;
    virtual HI_error_t HI_malloc2D( const void *host_ptr, void** dev_ptr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID)=0;
    virtual HI_error_t HI_malloc3D( const void *host_ptr, void** dev_ptr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID)=0;
    virtual HI_error_t HI_free( const void *host_ptr, int asyncID)=0;
	virtual HI_error_t HI_pin_host_memory(const void * hostPtr, size_t size)=0;
	virtual void HI_unpin_host_memory(const void* hostPtr)=0;

    // Memory Transfer
    virtual HI_error_t HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType)=0;

    virtual HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async)=0;
    virtual HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async)=0;
    virtual HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind)=0;
    virtual HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async)=0;

    virtual void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType)=0;
    virtual void HI_tempFree( void** tempPtr, acc_device_t devType)=0;
	
	// Experimental API to support unified memory //
    virtual HI_error_t  HI_malloc1D_unified(const void *hostPtr, void **devPtr, int count, int asyncID)= 0;
    virtual HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType)=0;
    virtual HI_error_t HI_free_unified( const void *host_ptr, int asyncID)=0;

    virtual HI_error_t createKernelArgMap() {
        return HI_success;
    }
    virtual HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size) {
        return HI_success;
    }
    virtual HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count) {
        return HI_success;
    }
    virtual void HI_set_async(int asyncId)=0;
    virtual void HI_wait(int arg) {}
    virtual void HI_wait_async(int arg, int async) {}
    virtual void HI_waitS1(int arg) {}
    virtual void HI_waitS2(int arg) {}
    virtual void HI_wait_all() {}
    virtual void HI_wait_all_async(int async) {}
    virtual int HI_async_test(int asyncId)=0;
    virtual int HI_async_test_all()=0;

    virtual void HI_malloc(void **devPtr, size_t size) = 0;
    virtual void HI_free(void *devPtr) = 0;

    HI_error_t HI_get_device_address(const void *hostPtr, void **devPtr, int asyncID) {
        return HI_get_device_address(hostPtr, devPtr, NULL, asyncID);
    }

    HI_error_t HI_get_device_address(const void *hostPtr, void **devPtr, size_t* offset, int asyncID) {
        return HI_get_device_address(hostPtr, devPtr, NULL, NULL, asyncID);
    }

    HI_error_t HI_get_device_address(const void *hostPtr, void **devPtr, size_t *offset, size_t *size, int asyncID) {
		std::multimap<int, std::map<const void *,void*> >::iterator it = masterAddressTable.find(asyncID);
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
        if(it2 != (it->second).end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            *devPtr = aet->devPtr;
			if( size ) *size = aet->size;
			if( offset ) *offset = 0;
            //*devPtr = it2->second;
            return  HI_success;
        } else {
            //check on the default stream
            it = masterAddressTable.find(DEFAULT_QUEUE);
            it2 =	(it->second).find(hostPtr);
            if(it2 != (it->second).end() ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	*devPtr = aet->devPtr;
				if( size ) *size = aet->size;
				if( offset ) *offset = 0;
            //	*devPtr = it2->second;
            	return  HI_success;
            }
		}

		//Check whether hostPtr is within the range of an allocated memory region 
		//in the addressTable.
		for (std::map<const void *,void*>::iterator it2 = (it->second).begin(); it2 != (it->second).end(); ++it2) {
            const void* aet_host = it2->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            if (hostPtr >= aet_host && (size_t) hostPtr < (size_t) aet_host + aet->size) {
                *devPtr = aet->devPtr;
				if( size ) *size = aet->size;
                if (offset) *offset = (size_t) hostPtr - (size_t) aet_host;
                return  HI_success;
            }
        }

        //check on the default stream
        it = masterAddressTable.find(DEFAULT_QUEUE);
        for (std::map<const void *,void*>::iterator it2 = (it->second).begin(); it2 != (it->second).end(); ++it2) {
            const void* aet_host = it2->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            if (hostPtr >= aet_host && (size_t) hostPtr < (size_t) aet_host + aet->size) {
                *devPtr = aet->devPtr;
				if( size ) *size = aet->size;
                if (offset) *offset = (size_t) hostPtr - (size_t) aet_host;
                return  HI_success;
            }
        }
        //fprintf(stderr, "[ERROR in get_device_address()] No mapping found for the host pointer\n");
        return HI_error;
    }

    HI_error_t HI_set_device_address(const void *hostPtr, void * devPtr, size_t size, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = masterAddressTable.find(asyncID);
        //fprintf(stderr, "[in set_device_address()] Setting address\n");
        if(it == masterAddressTable.end() ) {
            //fprintf(stderr, "[in set_device_address()] No mapping found for the asyncID\n");
            std::map<const void *,void*> emptyMap;
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
			emptyMap[hostPtr] = (void *) aet;
            masterAddressTable.insert(std::pair<int, std::map<const void *,void*> > (asyncID, emptyMap));
            //it = masterAddressTable.find(asyncID);
        } else {
        	//(it->second).insert(std::pair<const void *,void*>(hostPtr, devPtr));
        	//(it->second)[hostPtr] = devPtr;
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
        	(it->second)[hostPtr] = (void*) aet;
		}
        return  HI_success;
    }

    HI_error_t HI_remove_device_address(const void *hostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = masterAddressTable.find(asyncID);
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);

        if(it2 != (it->second).end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            delete aet;
            (it->second).erase(it2);
            return  HI_success;
        } else {
            fprintf(stderr, "[ERROR in remove_device_address()] No mapping found for the host pointer on async ID %d\n", asyncID);
            return HI_error;
        }
    }

    HI_error_t HI_get_host_address(const void *devPtr, void** hostPtr, int asyncID) {
		int containDevPtr = 0;
        std::multimap<int, std::map<const void *,void*> >::iterator it = masterAddressTable.find(asyncID);
        if(it == masterAddressTable.end() ) {
			return HI_error;
		}
		for( std::map<const void *, void*>::iterator it3 = (it->second).begin(); it3 != (it->second).end(); ++it3 ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it3->second;
			if( aet->devPtr == devPtr ) {
				containDevPtr = 1;
				*hostPtr = (void *)it3->first;
				break;
			} else if (devPtr >= aet->devPtr && (size_t) devPtr < (size_t) aet->devPtr + aet->size) {
				containDevPtr = 1;
                *hostPtr = (void*) ((size_t) it3->first + ((size_t) devPtr - (size_t) aet->devPtr));
                break;
            }
/*
			if( it3->second == devPtr ) {
				containDevPtr = 1;
				*hostPtr = (void *)it3->first;
				break;
			} 
*/
		}
		if( containDevPtr == 1 ) {
			return HI_success;
		} else {
			return HI_error;
		}
    }

    const void * HI_get_base_address_of_host_memory(const void *hostPtr, int asyncID) {
		std::multimap<int, std::map<const void *,void*> >::iterator it = masterAddressTable.find(asyncID);
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
        if(it2 != (it->second).end() ) {
            return  hostPtr;
        } else {
            //check on the default stream
            it = masterAddressTable.find(DEFAULT_QUEUE);
            it2 =	(it->second).find(hostPtr);
            if(it2 != (it->second).end() ) {
            	return  hostPtr;
            }
		}

		//Check whether hostPtr is within the range of an allocated memory region 
		//in the addressTable.
		for (std::map<const void *,void*>::iterator it2 = (it->second).begin(); it2 != (it->second).end(); ++it2) {
            const void* aet_host = it2->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            if (hostPtr >= aet_host && (size_t) hostPtr < (size_t) aet_host + aet->size) {
                return  aet_host;
            }
        }

        //check on the default stream
        it = masterAddressTable.find(DEFAULT_QUEUE);
        for (std::map<const void *,void*>::iterator it2 = (it->second).begin(); it2 != (it->second).end(); ++it2) {
            const void* aet_host = it2->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            if (hostPtr >= aet_host && (size_t) hostPtr < (size_t) aet_host + aet->size) {
                return  aet_host;
            }
        }
		//No entry is found.
        return 0;
    }

    HI_error_t HI_get_device_address_from_victim_cache(const void *hostPtr, void **devPtr, size_t *offset, size_t *size, int asyncID) {
		std::multimap<int, std::map<const void *,void*> >::iterator it = auxAddressTable.find(asyncID);
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
        if(it2 != (it->second).end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            *devPtr = aet->devPtr;
			if( size ) *size = aet->size;
			if( offset ) *offset = 0;
            //*devPtr = it2->second;
            return  HI_success;
        } else {
            //check on the default stream
            it = auxAddressTable.find(DEFAULT_QUEUE);
            it2 =	(it->second).find(hostPtr);
            if(it2 != (it->second).end() ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	*devPtr = aet->devPtr;
				if( size ) *size = aet->size;
				if( offset ) *offset = 0;
            //	*devPtr = it2->second;
            	return  HI_success;
            }
		}

        return HI_error;
    }

    HI_error_t HI_set_device_address_in_victim_cache (const void *hostPtr, void * devPtr, size_t size, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = auxAddressTable.find(asyncID);
        if(it == auxAddressTable.end() ) {
            std::map<const void *,void*> emptyMap;
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
			emptyMap[hostPtr] = (void *) aet;
            auxAddressTable.insert(std::pair<int, std::map<const void *,void*> > (asyncID, emptyMap));
            //it = auxAddressTable.find(asyncID);
        } else {
        	//(it->second).insert(std::pair<const void *,void*>(hostPtr, devPtr));
        	//(it->second)[hostPtr] = devPtr;
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
        	(it->second)[hostPtr] = (void*) aet;
		}
        return  HI_success;
    }

    HI_error_t HI_remove_device_address_from_victim_cache (const void *hostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = auxAddressTable.find(asyncID);
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);

        if(it2 != (it->second).end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            delete aet;
            (it->second).erase(it2);
            return  HI_success;
        } else {
            fprintf(stderr, "[ERROR in remove_device_address_from_victim_cache()] No mapping found for the host pointer on async ID %d\n", asyncID);
            return HI_error;
        }
    }

    HI_error_t HI_reset_victim_cache ( int asyncID ) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = auxAddressTable.find(asyncID);
        while(it != auxAddressTable.end()) {
			for( std::map<const void*,void*>::iterator it2 = (it->second).begin(); it2 != (it->second).end(); ++it2 ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	delete aet;
			}
			(it->second).clear();
            it++;
        }
		return  HI_success;
    }

    HI_error_t HI_reset_victim_cache_all ( ) {
		for( std::multimap<int, std::map<const void*,void*> >::iterator it = auxAddressTable.begin(); it != auxAddressTable.end(); ++it ) {
			for( std::map<const void*,void*>::iterator it2 = (it->second).begin(); it2 != (it->second).end(); ++it2 ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	delete aet;
			}
			(it->second).clear();
		}
		return  HI_success;
    }

    HI_error_t HI_free_async( const void *hostPtr, int asyncID ) {
        //fprintf(stderr, "[in HI_free_async()] with asyncID %d\n", asyncID);
        postponedFreeTable.insert(std::pair<int, const void *>(asyncID, hostPtr));
        return HI_success;
    }

    HI_error_t HI_postponed_free(int asyncID ) {
#if _OPENARC_DEBUG_ == 1 
        fprintf(stderr, "[enter HI_postponed_free()]\n");
#endif
        std::multimap<int, const void*>::iterator hostPtrIter = postponedFreeTable.find(asyncID);

        while(hostPtrIter != postponedFreeTable.end()) {
            //fprintf(stderr, "[in HI_postponed_free()] Freeing on stream %d, address %x\n", asyncID, hostPtrIter->second);
            HI_free(hostPtrIter->second, asyncID);
            hostPtrIter++;
        }

        postponedFreeTable.erase(asyncID);
#if _OPENARC_DEBUG_ == 1 
        fprintf(stderr, "[exit HI_postponed_free()]\n");
#endif
        return HI_success;
    }

    HI_error_t HI_get_temphost_address(const void *hostPtr, void **temphostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = tempHostAddressTable.find(asyncID);
        std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
        if(it2 != (it->second).end() ) {
            *temphostPtr = it2->second;
            return  HI_success;
        } else {
            //check on the default stream
            it = tempHostAddressTable.find(DEFAULT_QUEUE);
            it2 =	(it->second).find(hostPtr);
            if(it2 != (it->second).end() ) {
                *temphostPtr = it2->second;
                return  HI_success;
            }
            //fprintf(stderr, "[ERROR in get_temphost_address()] No mapping found for the host pointer\n");
            return HI_error;
        }
    }

    HI_error_t HI_set_temphost_address(const void *hostPtr, void * temphostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = tempHostAddressTable.find(asyncID);
        //fprintf(stderr, "[in set_temphost_address()] Setting address\n");
        if(it == tempHostAddressTable.end() ) {
            //fprintf(stderr, "[in set_temphost_address()] No mapping found for the asyncID\n");
            std::map<const void *,void*> emptyMap;
            tempHostAddressTable.insert(std::pair<int, std::map<const void *,void*> > (asyncID, emptyMap));
            it = tempHostAddressTable.find(asyncID);
        }

        //(it->second).insert(std::pair<const void *,void*>(hostPtr, temphostPtr));
        (it->second)[hostPtr] = temphostPtr;
        return  HI_success;
    }

    HI_error_t HI_remove_temphost_address(const void *hostPtr, int asyncID) {
        std::multimap<int, std::map<const void *,void*> >::iterator it = tempHostAddressTable.find(asyncID);
		if( it != tempHostAddressTable.end() ) {
        	std::map<const void *,void*>::iterator it2 =	(it->second).find(hostPtr);
        	if(it2 != (it->second).end() ) {
            	(it->second).erase(it2);
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
#if _OPENARC_DEBUG_ == 1 
        fprintf(stderr, "[enter HI_free_temphosts()]\n");
#endif
        std::multimap<int, std::map<const void *,void*> >::iterator it = tempHostAddressTable.find(asyncID);
		if (it != tempHostAddressTable.end()) {
			for( std::map<const void*,void*>::iterator it2 = (it->second).begin(); it2 != (it->second).end(); ++it2 ) {
				HI_tempFree(&(it2->second), acc_device_host);
			}
			(it->second).clear();
		}
#if _OPENARC_DEBUG_ == 1 
        fprintf(stderr, "[exit HI_free_temphosts()]\n");
#endif
    }

} Accelerator_t;

//////////////////////////
// Moved from openacc.h //
//////////////////////////
extern void acc_init( acc_device_t devtype, int kernels, std::string kernelNames[]);

////////////////////////
// Runtime init/reset //
////////////////////////
extern void HI_hostinit(int numhostthreads);

//////////////////////
// Kernel Execution //
//////////////////////
extern HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args);
extern HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value);
extern HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE);
extern HI_error_t HI_synchronize();

/////////////////////////////
//Device Memory Allocation //
/////////////////////////////
extern HI_error_t HI_malloc1D( const void *hostPtr, void** devPtr, size_t count, int asyncID);
extern HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID);
extern HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID);
extern HI_error_t HI_free( const void *hostPtr, int asyncID);
extern HI_error_t HI_free_async( const void *hostPtr, int asyncID);
extern void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType);
extern void HI_tempFree( void** tempPtr, acc_device_t devType);

/////////////////////////////////////////////////
//Memory transfers between a host and a device //
/////////////////////////////////////////////////
extern HI_error_t HI_memcpy(void *dst, const void *src, size_t count,
                                  HI_MemcpyKind_t kind, int trType);
extern HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count,
                                        HI_MemcpyKind_t kind, int trType, int async);
extern HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count,
                                        HI_MemcpyKind_t kind, int trType, int async);
extern HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                                    size_t widthInBytes, size_t height, HI_MemcpyKind_t kind);
extern HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async);
//extern HI_error_t HI_memcpy3D(void *dst, size_t dpitch, const void *src, size_t spitch,
//	size_t widthInBytes, size_t height, size_t depth, HI_MemcpyKind_t kind);
//extern HI_error_t HI_memcpy3D_async(void *dst, size_t dpitch, const void *src,
//	size_t spitch, size_t widthInBytes, size_t height, size_t depth,
//	HI_MemcpyKind_t kind, int async);
extern HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count);

////////////////////////////////////////////////
// Experimental API to support unified memory //
////////////////////////////////////////////////
extern HI_error_t HI_malloc1D_unified( const void *hostPtr, void** devPtr, size_t count, int asyncID);
extern HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count,
                                  HI_MemcpyKind_t kind, int trType);
extern HI_error_t HI_free_unified( const void *hostPtr, int asyncID);

////////////////////////////
//Internal mapping tables //
////////////////////////////
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtr, int asyncID);
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtr, size_t * offset, int asyncID);
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtr, size_t * offset, size_t * size, int asyncID);
extern HI_error_t HI_set_device_address(const void * hostPtr, void * devPtr, size_t size, int asyncID);
extern HI_error_t HI_remove_device_address(const void * hostPtr, int asyncID);
extern HI_error_t HI_get_host_address(const void *devPtr, void** hostPtr, int asyncID);
extern HI_error_t HI_get_temphost_address(const void * hostPtr, void ** temphostPtr, int asyncID);
//extern HI_error_t HI_set_temphost_address(const void * hostPtr, void * temphostPtr, int asyncID);
//extern HI_error_t HI_remove_temphost_address(const void * hostPtr);
extern int HI_getninc_prtcounter(const void * hostPtr, void **devPtr, int asyncID);
extern int HI_decnget_prtcounter(const void * hostPtr, void **devPtr, int asyncID);

/////////////////////////////////////////////////////////////////////////
//async integer argument => internal handler (ex: CUDA stream) mapping //
/////////////////////////////////////////////////////////////////////////
extern HI_error_t HI_create_async_handle( int async);
extern int HI_contain_async_handle( int async );
extern HI_error_t HI_delete_async_handle( int async );
extern void HI_set_async(int asyncId);
////////////////////////////////
//Memory management functions //
////////////////////////////////
extern void HI_check_read(const void * hostPtr, acc_device_t dtype, const char *varName, const char *refName, int loopIndex);
extern void HI_check_write(const void * hostPtr, acc_device_t dtype, const char *varName, const char *refName, int loopIndex);
extern void HI_set_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, const char * varName, const char * refName, int loopIndex);
extern void HI_reset_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, int asyncID);
//Below is deprecated
extern void HI_init_status(const void * hostPtr);

////////////////////
//Texture function //
////////////////////
extern HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size);

////////////////////
//Misc. functions //
////////////////////
extern double HI_get_localtime();


////////////////////////////////////////////
//Functions used for program verification //
////////////////////////////////////////////
extern void HI_waitS1(int asyncId);
extern void HI_waitS2(int asyncId);


///////////////////////////////////////
//Functions used for resilience test //
///////////////////////////////////////
#include "resilience.h"



#endif
