#ifndef __IMPACC_RUNTIME_H__
#define __IMPACC_RUNTIME_H__

#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <semaphore.h>
#include <set>
#include <vector>
#include "openacc.h"
#include "openaccrt.h"
#include "openaccrt_ext.h"
#include "queue.h"
#include "acc_mem_api.h"

#define DIST_TASK_TYPE_NULL             0
#define DIST_TASK_TYPE_SEND_H           1
#define DIST_TASK_TYPE_SEND_D           2
#define DIST_TASK_TYPE_RECV_H           3
#define DIST_TASK_TYPE_RECV_D           4
#define DIST_TASK_TYPE_BCAST            5
#define DIST_TASK_TYPE_REDUCE_HH        6
#define DIST_TASK_TYPE_ALLREDUCE_HH     7
#define DIST_TASK_TYPE_BARRIER          8
#define DIST_TASK_TYPE_TYPECOMMIT       9

#define DIST_TASK_INIT                  0
#define DIST_TASK_SUBMITTED             1
#define DIST_TASK_DEQUEUED              2
#define DIST_TASK_WAITFORREADY          3
#define DIST_TASK_READY                 4
#define DIST_TASK_RUNNING               5
#define DIST_TASK_COMPLETE              6

#define MAX_ACCS_PER_NODE               8

using namespace std;

class DistCommThread;
class DistExecutor;
class DistTask;
class NodeData;
class NodeDataManager;

extern DistExecutor* gDE;

class DistExecutor {
public:
    DistExecutor(int argc, char** argv, int numdevs);
    ~DistExecutor();

    int NumTasks();
    int TaskNum(int local_devnum);
    int ME();
    bool WithInNode(int dev1, int dev2);
    int NodeNum(int devid);

    void Wait(int async);
    void WaitMPIRequest(MPI_Request* mpi_req, MPI_Status* mpi_status);
    void Barrier();
    void SendFromHost(int dst, void* buf, size_t size, int tag, int flags);
    void SendFromHostAsync(int dst, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);
    void SendFromHostAsyncDatatype(int dst, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async, MPI_Datatype datatype);
    void SendFromDevice(int dst, void* buf, size_t size, int tag, int flags);
    void SendFromDeviceAsync(int dst, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);
    void RecvToHost(int src, void* buf, size_t size, int tag, int flags);
    void RecvToHostAsync(int src, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);
    void RecvToDevice(int src, void* buf, size_t size, int tag, int flags);
    void RecvToDeviceAsync(int src, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async);
    void ReduceFromHostToHost(void* sb, void* rb, size_t size, acc_datatype datatype, acc_op op, int root);
    void AllReduceFromHostToHost(void* sb, void* rb, size_t size, acc_datatype datatype, acc_op op);
    void Bcast(int root, void* buf, size_t size);
    void TypeCommit(void* datatype);

private:
    bool CanAccessPeer(Accelerator* driver_s, Accelerator* driver_r); 
    void AddAsync(DistTask* task);
    void AddMPIRequest(DistTask* task);

public:
    int rank;
    int size;
    char name[256];
    int numdevs;
    NodeDataManager* NDM;
    DistCommThread* DCT;

private:
    int mpi_ret;
    vector<DistTask*> asyncs[MAX_ACCS_PER_NODE];
    map<MPI_Request, DistTask*> mpi_reqs[MAX_ACCS_PER_NODE];
    int peeraccess[MAX_ACCS_PER_NODE][MAX_ACCS_PER_NODE];
};

class DistCommThread {
public:
    DistCommThread(DistExecutor* DE);
    ~DistCommThread();

    void EnqueueTask(DistTask* task);
    void ExecuteTask(DistTask* task);

    void Start();
    void Stop();

private:
    void Run();
    void HandleTask(DistTask* task);
    void HandleTaskSend(DistTask* task);
    void HandleTaskSendWin(DistTask* task);
    void HandleTaskSendAcn(DistTask* task);
    void HandleTaskRecv(DistTask* task);
    void HandleTaskRecvWin(DistTask* task);
    void HandleTaskRecvAcn(DistTask* task);
    void HandleTaskPairWin(DistTask* send, DistTask* recv);
    void HandleTaskBcast(DistTask* task);
    void HandleTaskReduce(DistTask* task);
    void HandleTaskAllReduce(DistTask* task);
    void HandleTaskBarrier(DistTask* task);
    void HandleTaskTypeCommit(DistTask* task);
    void CheckAsyncs();

    static void* ThreadFunc(void* argp);

private:
    DistExecutor* DE;
    pthread_t thread;
    bool running;
    LockFreeQueue<DistTask*>* queue;
    vector<DistTask*> pendings;
    pthread_mutex_t mutex_pendings;
    vector<DistTask*> asyncs;
    pthread_mutex_t mutex_asyncs;
    bool busy_waiting;
    sem_t sem_schedule;
    DistTask* last_c_task;
    DistTask* c_task[8];
    int c_task_idx;
};

class DistTask {
public:
    DistTask(unsigned long id);
    ~DistTask();

    void Reset(unsigned long id);
    void Clear();
    bool Async();
    void Submit();
    void Run();
    void Ready();
    void Wait();
    void WaitForReady();
    void Complete();

    static void Create(DistTask** newtask);
    static void CreateSend(DistTask** newtask, int type, int me, int dst, void* sb, size_t size, bool win, int tag, int flags);
    static void CreateSendAsync(DistTask** newtask, int type, int me, int dst, void* sb, size_t size, bool win, int tag, MPI_Request* mpi_req, int flags, int async);
    static void CreateRecv(DistTask** newtask, int type, int me, int src, void* rb, size_t size, bool win, int tag, int flags);
    static void CreateRecvAsync(DistTask** newtask, int type, int me, int src, void* rb, size_t size, bool win, int tag, MPI_Request* mpi_req, int flags, int async);
    static void CreateReduce(DistTask** newtask, int type, int me, int src, void* sb, void* rb, size_t count, acc_datatype datatype, acc_op op, int root);
    static void CreateAllReduce(DistTask** newtask, int type, int me, int src, void* sb, void* rb, size_t count, acc_datatype datatype, acc_op op);
    static void CreateBcast(DistTask** newtask, int type, int me, int root, void* buf, size_t size);
    static void CreateBarrier(DistTask** newtask);
    static void Free(DistTask* task);

public:
    unsigned long id;
    int type;
    int me;
    int src;
    int dst;
    void* sb;
    void* rb;
    size_t size;
    bool win;
    int tag;
    int flags;
    int async;
    int kind;
    int ci;
    void* cb;
    MPI_Datatype mpi_datatype;
    acc_datatype datatype;
    acc_op op;
    MPI_Request* mpi_req;
    DistTask* pair;
    Accelerator* dev;

private:
    int status;
    sem_t sem_ready;
    sem_t sem_complete;
};

class NodeData {
public:
    NodeData(void* addr, size_t size, void** ptr);
    ~NodeData();

public:
    void* addr;
    size_t size;
    void** ptr;
    int ref_cnt;
};

class NodeDataManager {
public:
    NodeDataManager(DistExecutor* DE);
    ~NodeDataManager();

    void AddData(void* addr, size_t size, void** ptr);
    void FreeData(void* addr);
    bool Alias(void* dst, void* src, size_t size);

private:
    DistExecutor* DE;
    map<void*, NodeData*> pool;
    pthread_mutex_t mutex_pool;
};

void IMPACC_init(int argc, char** argv, int numdevs);
void IMPACC_shutdown();

#endif /* __IMPACC_RUNTIME_H__ */
