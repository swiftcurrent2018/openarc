#ifndef __IMPACC_H
#define __IMPACC_H

#include <mpi.h>

#define IMPACC_MEM_S_DEV        (1 << 0)
#define IMPACC_MEM_S_RO         (1 << 1)
#define IMPACC_MEM_R_DEV        (1 << 10)
#define IMPACC_MEM_R_RO         (1 << 11)

#define IMPACC_FLAGS_NONE       0

#define IMPACC_API_FUNC(f)  IMPACC_##f

#ifdef __cplusplus
extern "C" {
#endif

int IMPACC_API_FUNC(MPI_Init(int *argc, char ***argv));
int IMPACC_API_FUNC(MPI_Finalize(void));

int IMPACC_API_FUNC(MPI_Comm_size)(MPI_Comm comm, int *size);
int IMPACC_API_FUNC(MPI_Comm_rank)(MPI_Comm comm, int *rank);

int IMPACC_API_FUNC(MPI_Send)(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, int flags);
int IMPACC_API_FUNC(MPI_Recv)(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status, int flags);
int IMPACC_API_FUNC(MPI_Isend)(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request, int flags, int async);
int IMPACC_API_FUNC(MPI_Irecv)(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request, int flags, int async);

int IMPACC_API_FUNC(MPI_Wait)(MPI_Request *request, MPI_Status *status);
int IMPACC_API_FUNC(MPI_Waitall)(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);

int IMPACC_API_FUNC(MPI_Barrier(MPI_Comm comm));

int IMPACC_API_FUNC(MPI_Bcast)(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, int flags);
int IMPACC_API_FUNC(MPI_Reduce)(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, int flags);
int IMPACC_API_FUNC(MPI_Allreduce)(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, int flags);

void *IMPACC_API_FUNC(malloc)(size_t size, void **ptr);
void IMPACC_API_FUNC(free)(void *ptr);
void *IMPACC_API_FUNC(calloc)(size_t nmemb, size_t size, void **ptr);

#ifdef __cplusplus
}
#endif

#endif
