#include "impacc.h"
#include "acc_mem_api.h"
#include "debug.h"
#include "utils.h"

int IMPACC_API_FUNC(MPI_Init(int *argc, char ***argv)) {
    int ret;
    MPI_Initialized(&ret);
    if (!ret) printf("MPI_Init was not initialized. flag[%d]\n", ret);
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Finalize(void)) {
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Comm_size)(MPI_Comm comm, int *size) {
    *size = acc_get_num_tasks();
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Comm_rank)(MPI_Comm comm, int *rank) {
    *rank = acc_get_task_num();
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Send)(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, int flags) {
    size_t size = impacc::MPI_size(count, datatype);
    if (flags & IMPACC_MEM_S_DEV) {
        acc_mem_send_from_device(dest, acc_deviceptr(const_cast<void*>(buf)), size, tag, flags);
    } else {
        acc_mem_send(dest, const_cast<void*>(buf), size, tag, flags);
    }
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Recv)(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status, int flags) {
    size_t size = impacc::MPI_size(count, datatype);
    if (flags & IMPACC_MEM_R_DEV) {
        acc_mem_recv_to_device(source, acc_deviceptr(buf), size, tag, flags);
    } else {
        acc_mem_recv(source, buf, size, tag, flags);
    }
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Isend)(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request, int flags, int async) {
    size_t size = impacc::MPI_size(count, datatype);
    if (flags & IMPACC_MEM_S_DEV) {
        acc_mem_send_from_device_async(dest, acc_deviceptr(const_cast<void*>(buf)), size, tag, request, flags, async);
    } else {
        acc_mem_send_async(dest, const_cast<void*>(buf), size, tag, request, flags, async);
    }
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Irecv)(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request, int flags, int async) {
    size_t size = impacc::MPI_size(count, datatype);
    if (flags & IMPACC_MEM_R_DEV) {
        acc_mem_recv_to_device_async(source, acc_deviceptr(buf), size, tag, request, flags, async);
    } else {
        acc_mem_recv_async(source, buf, size, tag, request, flags, async);
    }
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Wait)(MPI_Request *request, MPI_Status *status) {
    acc_mem_wait(request, status);
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Waitall)(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) {
    for (int i = 0; i < count; i++) {
        acc_mem_wait(array_of_requests + i, array_of_statuses == MPI_STATUSES_IGNORE ? MPI_STATUS_IGNORE : array_of_statuses + i);
    }
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Barrier(MPI_Comm comm)) {
    MPI_Barrier(comm);
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Bcast)(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, int flags) {
    size_t size = impacc::MPI_size(count, datatype);
    acc_mem_bcast(root, buffer, size);
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Reduce)(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, int flags) {
    acc_mem_reduce_from_host_to_host(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, root);
    return MPI_SUCCESS;
}

int IMPACC_API_FUNC(MPI_Allreduce)(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, int flags) {
    acc_mem_allreduce_from_host_to_host(const_cast<void*>(sendbuf), recvbuf, count, datatype, op);
    return MPI_SUCCESS;
}

void *IMPACC_API_FUNC(malloc)(size_t size, void **ptr) {
    return acc_mem_malloc(size, ptr);
}

void IMPACC_API_FUNC(free)(void *ptr) {
    return acc_mem_free(ptr);
}

void *IMPACC_API_FUNC(calloc)(size_t nmemb, size_t size, void **ptr) {
    return acc_mem_calloc(nmemb, size, ptr);
}

