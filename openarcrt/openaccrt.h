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
#define MAX_NUM_QUEUES_PER_THREAD 1048576

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
typedef std::set<const void *> pointerset_t;
typedef std::map<int, addresstable_t *> addresstablemap_t;
typedef std::multimap<size_t, void *> memPool_t;
typedef std::map<int, memPool_t *> memPoolmap_t;
typedef std::map<void *, int> countermap_t;
typedef std::map<const void *, size_t> sizemap_t;
typedef std::map<int, addressmap_t *> asyncphostmap_t;
typedef std::map<int, sizemap_t *> asynchostsizemap_t;
typedef std::map<const void *, HI_memstatus_t> memstatusmap_t;

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
	//memPool_t memPool;
	memPoolmap_t memPoolMap;
	

	virtual ~Accelerator() {};

    // Kernel Initialization
    virtual HI_error_t init() = 0;
    virtual HI_error_t destroy()=0;

    // Kernel Execution
    virtual HI_error_t HI_register_kernels(std::set<std::string>kernelNames) = 0;
    virtual HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args) = 0;
    virtual HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value, int arg_type) = 0;
    virtual HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE) = 0;
    virtual HI_error_t HI_synchronize( )=0;

    // Memory Allocation
    virtual HI_error_t HI_malloc1D(const void *hostPtr, void **devPtr, size_t count, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE)= 0;
    virtual HI_error_t HI_malloc2D( const void *host_ptr, void** dev_ptr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE)=0;
    virtual HI_error_t HI_malloc3D( const void *host_ptr, void** dev_ptr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE)=0;
    virtual HI_error_t HI_free( const void *host_ptr, int asyncID)=0;
	virtual HI_error_t HI_pin_host_memory(const void * hostPtr, size_t size)=0;
	virtual void HI_unpin_host_memory(const void* hostPtr)=0;

    // Memory Transfer
    virtual HI_error_t HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType)=0;

    virtual HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async)=0;
    virtual HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async)=0;
    virtual HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind)=0;
    virtual HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async)=0;

    virtual void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags=HI_MEM_READ_WRITE)=0;
    virtual void HI_tempFree( void** tempPtr, acc_device_t devType)=0;
	
	// Experimental API to support unified memory //
    virtual HI_error_t  HI_malloc1D_unified(const void *hostPtr, void **devPtr, size_t count, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE)= 0;
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
    virtual void HI_wait_ifpresent(int arg) {}
    virtual void HI_wait_async(int arg, int async) {}
    virtual void HI_wait_async_ifpresent(int arg, int async) {}
    virtual void HI_waitS1(int arg) {}
    virtual void HI_waitS2(int arg) {}
    virtual void HI_wait_all() {}
    virtual void HI_wait_all_async(int async) {}
    virtual int HI_async_test(int asyncId)=0;
    virtual int HI_async_test_ifpresent(int asyncId)=0;
    virtual int HI_async_test_all()=0;

    virtual void HI_malloc(void **devPtr, size_t size, HI_MallocKind_t flags=HI_MEM_READ_WRITE) = 0;
    virtual void HI_free(void *devPtr) = 0;

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
            masterAddressTable->insert(std::pair<int, addressmap_t *> (asyncID, emptyMap));
            //it = masterAddressTable->find(asyncID);
        } else {
        	addresstable_entity_t *aet = new addresstable_entity_t(devPtr, size);
        	(*(it->second))[hostPtr] = (void*) aet;
		}
        return  HI_success;
    }

    HI_error_t HI_remove_device_address(const void *hostPtr, int asyncID, int tid) {
    	addresstable_t *masterAddressTable = masterAddressTableMap[tid];
        addresstable_t::iterator it = masterAddressTable->find(asyncID);
        addressmap_t::iterator it2 =	(it->second)->find(hostPtr);

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
        if(it != masterAddressTable->end() ) {
			addressmap_t *tAddressMap = it->second;
			for( addressmap_t::iterator it3 = tAddressMap->begin(); it3 != tAddressMap->end(); ++it3 ) {
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
            if(it2 != tAddressMap->end() ) {
            	return  hostPtr;
            }
		}

		//Check whether hostPtr is within the range of an allocated memory region 
		//in the addressTable.
		for (addressmap_t::iterator it2 = tAddressMap->begin(); it2 != tAddressMap->end(); ++it2) {
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
		addresstable_t::iterator it = auxAddressTable.find(asyncID);
		//Check whether hostPtr exists as an entry to addressTable (it->second), 
		//which will be true if hostPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	(it->second)->find(hostPtr);
        if(it2 != (it->second)->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            *devPtrBase = aet->basePtr;
			if( size ) *size = aet->size;
			if( offset ) *offset = 0;
            //*devPtrBase = it2->second;
            return  HI_success;
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
            	return  HI_success;
            }
		}

        return HI_error;
    }

    HI_error_t HI_set_device_address_in_victim_cache (const void *hostPtr, void * devPtr, size_t size, int asyncID) {
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
        return  HI_success;
    }

    HI_error_t HI_remove_device_address_from_victim_cache (const void *hostPtr, int asyncID) {
        addresstable_t::iterator it = auxAddressTable.find(asyncID);
        addressmap_t::iterator it2 =	(it->second)->find(hostPtr);

        if(it2 != (it->second)->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            delete aet;
            (it->second)->erase(it2);
            return  HI_success;
        } else {
            fprintf(stderr, "[ERROR in remove_device_address_from_victim_cache()] No mapping found for the host pointer on async ID %d\n", asyncID);
            return HI_error;
        }
    }

    HI_error_t HI_reset_victim_cache ( int asyncID ) {
        addresstable_t::iterator it = auxAddressTable.find(asyncID);
        while(it != auxAddressTable.end()) {
			for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	delete aet;
			}
			(it->second)->clear();
            it++;
        }
		return  HI_success;
    }

    HI_error_t HI_reset_victim_cache_all ( ) {
		for( addresstable_t::iterator it = auxAddressTable.begin(); it != auxAddressTable.end(); ++it ) {
			for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	delete aet;
			}
			(it->second)->clear();
		}
		return  HI_success;
    }

    HI_error_t HI_get_device_mem_handle(const void *devPtr, HI_device_mem_handle_t *memHandle, int tid) {
    	addressmap_t *myHandleMap = masterHandleTable[tid];
#if PRESENT_TABLE_SEARCH_MODE == 0
		//Check whether devPtr exists as an entry to myHandleMap, 
		//which will be true if devPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	myHandleMap->find(devPtr);
        if(it2 != myHandleMap->end() ) {
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            memHandle->basePtr = aet->basePtr;
            memHandle->offset = 0;
            return  HI_success;
		}

		//Check whether devPtr is within the range of an allocated memory region 
		//in the addressTable.
		for (addressmap_t::iterator it2 = myHandleMap->begin(); it2 != myHandleMap->end(); ++it2) {
            const void* aet_devPtr = it2->first;
            addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            if (devPtr >= aet_devPtr && (size_t) devPtr < (size_t) aet_devPtr + aet->size) {
                memHandle->basePtr = aet->basePtr;
                memHandle->offset = (size_t) devPtr - (size_t) aet_devPtr;
                return  HI_success;
            }
        }
#else
		//Check whether devPtr exists as an entry to myHandleMap, 
		//which will be true if devPtr is a base address of the pointed memory.
        addressmap_t::iterator it2 =	myHandleMap->lower_bound(devPtr);
        if(it2 != myHandleMap->end() ) {
			if( it2->first == devPtr ) {
				//found the entry matching the key, devPtr.
            	addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            	memHandle->basePtr = aet->basePtr;
            	memHandle->offset = 0;
            	return  HI_success;
			} else {
				//devPtr may belong to an entry before the current one.
				if( it2 == myHandleMap->begin() ) {
					//There is no entry before the current one.
					memHandle->basePtr = NULL;
					memHandle->offset = 0;
					return HI_error;
				} else {
					--it2; 
            		const void* aet_devPtr = it2->first;
            		addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
            		if (devPtr >= aet_devPtr && (size_t) devPtr < (size_t) aet_devPtr + aet->size) {
                		memHandle->basePtr = aet->basePtr;
                		memHandle->offset = (size_t) devPtr - (size_t) aet_devPtr;
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
                return  HI_success;
            }
		}
#endif

        //fprintf(stderr, "[ERROR in get_device_mem_handle()] No mapping found for the device pointer\n");
		memHandle->basePtr = NULL;
		memHandle->offset = 0;
        return HI_error;
    }

    HI_error_t HI_set_device_mem_handle(const void *devPtr, void * handle, size_t size, int tid) {
    	addressmap_t *myHandleMap = masterHandleTable[tid];
        //fprintf(stderr, "[in set_device_mem_handle()] Setting address\n");
        addresstable_entity_t *aet = new addresstable_entity_t(handle, size);
        (*myHandleMap)[devPtr] = (void*) aet;
        return  HI_success;
    }

    HI_error_t HI_remove_device_mem_handle(const void *devPtr, int tid) {
    	addressmap_t *myHandleMap = masterHandleTable[tid];
        addressmap_t::iterator it2 = myHandleMap->find(devPtr);
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
            HI_free(hostPtrIter->second, asyncID);
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
extern HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value, int arg_type);
extern HI_error_t HI_kernel_call(std::string kernel_name, int gridSize[3], int blockSize[3], int async=DEFAULT_QUEUE);
extern HI_error_t HI_synchronize();

/////////////////////////////
//Device Memory Allocation //
/////////////////////////////
extern HI_error_t HI_malloc1D( const void *hostPtr, void** devPtr, size_t count, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE);
extern HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE);
extern HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE);
extern HI_error_t HI_free( const void *hostPtr, int asyncID);
extern HI_error_t HI_free_async( const void *hostPtr, int asyncID);
extern void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags=HI_MEM_READ_WRITE);
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
extern HI_error_t HI_malloc1D_unified( const void *hostPtr, void** devPtr, size_t count, int asyncID, HI_MallocKind_t flags=HI_MEM_READ_WRITE);
extern HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count,
                                  HI_MemcpyKind_t kind, int trType);
extern HI_error_t HI_free_unified( const void *hostPtr, int asyncID);

////////////////////////////
//Internal mapping tables //
////////////////////////////
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtr, int asyncID);
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtrBase, size_t * offset, int asyncID);
extern HI_error_t HI_get_device_address(const void * hostPtr, void ** devPtrBase, size_t * offset, size_t * size, int asyncID);
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
