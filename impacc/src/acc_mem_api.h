#ifndef __IMPACC_INTERNAL_RUNTIME_API_H__
#define __IMPACC_INTERNAL_RUNTIME_API_H__

#include <stdlib.h>
#include "impacc.h"
#include "openacc.h"

#if 1
typedef MPI_Datatype    acc_datatype;
typedef MPI_Op          acc_op;
#else
typedef enum {
    acc_float,
    acc_double
} acc_datatype;

typedef enum {
    acc_sum,
    acc_max,
    acc_min
} acc_op;
#endif

int acc_get_num_tasks();
int acc_get_task_num();

void acc_mem_send(int dst, void* buf, size_t size, int tag, int flags);
void acc_mem_send_async(int dst, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);
void acc_mem_send_from_host(int dst, h_void* buf, size_t size, int tag, int flags);
void acc_mem_send_from_host_async(int dst, h_void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);
void acc_mem_send_from_device(int dst, d_void* buf, size_t size, int tag, int flags);
void acc_mem_send_from_device_async(int dst, d_void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);

void acc_mem_recv(int src, void* buf, size_t size, int tag, int flags);
void acc_mem_recv_async(int src, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);
void acc_mem_recv_to_host(int src, h_void* buf, size_t size, int tag, int flags);
void acc_mem_recv_to_host_async(int src, h_void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);
void acc_mem_recv_to_device(int src, d_void* buf, size_t size, int flags, int tag);
void acc_mem_recv_to_device_async(int src, d_void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);

void acc_mem_wait(MPI_Request* mpi_req, MPI_Status* mpi_status);
void acc_mem_wait_host(int async);
void acc_mem_wait_device(int async);

void acc_mem_bcast(int root, void* buf, size_t size);
void acc_mem_reduce_from_host_to_host(void* sendbuf, void* recvbuf, size_t count, acc_datatype datatype, acc_op op, int root);
void acc_mem_allreduce_from_host_to_host(void* sendbuf, void* recvbuf, size_t count, acc_datatype datatype, acc_op op);

void acc_barrier();

void acc_type_commit(void* datatype);

void* acc_mem_malloc(size_t size, void** ptr);
void* acc_mem_calloc(size_t count, size_t size, void** ptr);
void acc_mem_free(void* ptr);

#endif /* __IMPACC_INTERNAL_RUNTIME_API_H__ */
