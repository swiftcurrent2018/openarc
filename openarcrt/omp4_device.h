#ifndef __OPENMP4_DEV_HEADER__ 

#define __OPENMP4_DEV_HEADER__ 

#include <stddef.h>

//This header file contains a list of declarations of OpenMP runtime routines used in the OpenMP4-to-OpenACC translation.
//This can be used as a fake header file if a host OpenMP compiler does not support OpenMP4.

extern void *omp_target_alloc(size_t size, int device_num);

extern void omp_target_free(void * device_ptr, int device_num);

extern void omp_target_is_present(void * ptr, int device_num);

extern void omp_target_associate_ptr(void * host_ptr, void * device_ptr, size_t size, size_t device_offset, int device_num);

extern void omp_target_disassociate_ptr(void * ptr, int device_num);

extern void omp_target_memcpy(void * dst, void * src, size_t length, size_t dst_offset, size_t src_offset, int dst_device_num, int src_dev_num);

extern int omp_target_memcpy_rect( void * dst, void * src, size_t element_size, int num_dims, const size_t* volume, const size_t* dst_offsets, const size_t* src_offsets, const size_t* dst_dimensions, const size_t* src_dimensions, int dst_device_num, int src_device_num);

#endif


