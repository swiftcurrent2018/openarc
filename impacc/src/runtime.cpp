#include "runtime.h"
#include "debug.h"
#include "utils.h"

DistExecutor* gDE;

void IMPACC_init(int argc, char** argv, int numdevs) {
    gDE = new DistExecutor(argc, argv, numdevs);
    _check();
}

void IMPACC_shutdown() {
    if (!gDE) return;
    delete gDE;
    gDE = NULL;
}

DistExecutor::DistExecutor(int argc, char** argv, int numdevs) {
    this->numdevs = numdevs;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &mpi_ret);

    memset(peeraccess, 0, sizeof(peeraccess));

    NDM = new NodeDataManager(this);

    DCT = new DistCommThread(this);
    DCT->Start();
}

DistExecutor::~DistExecutor() {
    MPI_Finalize();
}

int DistExecutor::NumTasks() {
    return size * numdevs;
}

int DistExecutor::TaskNum(int local_devnum) {
    return rank * numdevs + local_devnum;
}

int DistExecutor::ME() {
    return TaskNum(get_thread_id());
}

bool DistExecutor::WithInNode(int dev1, int dev2) {
    return dev1 / numdevs == dev2 / numdevs;
}

int DistExecutor::NodeNum(int devid) {
    return devid / numdevs;
}

void DistExecutor::AddAsync(DistTask* task) {
    asyncs[get_thread_id()].push_back(task);
}

void DistExecutor::AddMPIRequest(DistTask* task) {
    *(task->mpi_req) = (int) task->id;
    mpi_reqs[get_thread_id()][*(task->mpi_req)] = task;
}

bool DistExecutor::CanAccessPeer(Accelerator* _driver_s, Accelerator* _driver_r) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
    return false;
#else
    int devnum_s = _driver_s->device_num;
    int devnum_r = _driver_r->device_num;

    if (peeraccess[devnum_r][devnum_s] == 0) {
        CudaDriver* driver_s = (CudaDriver*) _driver_s;
        CudaDriver* driver_r = (CudaDriver*) _driver_r;
        CUdevice dev_s = driver_s->cuDevice;
        CUdevice dev_r = driver_r->cuDevice;
        CUcontext ctx_s = driver_s->cuContext;
        CUcontext ctx_r = driver_r->cuContext;

        int canAccessPeer;

        CUresult cuerr = cuDeviceCanAccessPeer(&canAccessPeer, dev_r, dev_s);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);

        if (canAccessPeer) {
            cuerr = cuCtxEnablePeerAccess(ctx_s, 0);    
            if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
            peeraccess[devnum_r][devnum_s] = 1;
            _trace("DEV[%d] [%d] CAN ACCESS", devnum_r, devnum_s);
        } else {
            peeraccess[devnum_r][devnum_s] = -1;
            _trace("DEV[%d] [%d] CANNOT ACCESS", devnum_r, devnum_s);
        }
    }
    return peeraccess[devnum_r][devnum_s] == 1;
#endif
}

void DistExecutor::Wait(int async) {
    vector<DistTask*> v = asyncs[get_thread_id()];
    vector<DistTask*>::iterator it = v.begin();
    while (it != v.end()) {
        DistTask* dt = *it;
        if (async == dt->async) {
            if (dt->win && (dt->type == DIST_TASK_TYPE_RECV_D || dt->type == DIST_TASK_TYPE_SEND_D)) {
                dt->Complete();
                dt->pair->Complete();
            } else if (!dt->win && dt->type == DIST_TASK_TYPE_RECV_D) {
                dt->Complete();
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
                free(dt->sb);
#else
                CUresult cuerr = cuMemFreeHost(dt->sb);
                if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
#endif
                DistTask::Free(dt);
            } else {
                dt->Wait();
            }
            v.erase(it);
        } else ++it;
    }
}

void DistExecutor::WaitMPIRequest(MPI_Request* mpi_req, MPI_Status* mpi_status) {
    if (*mpi_req == MPI_REQUEST_NULL) return;
    if (mpi_reqs[get_thread_id()].count(*mpi_req)) {
        DistTask* dt = mpi_reqs[get_thread_id()][*mpi_req];
        dt->Wait();
        mpi_reqs[get_thread_id()].erase(*mpi_req);
        *mpi_req = MPI_REQUEST_NULL;
        //DistTask::Free(dt); TODO
    } else {
        _trace("mpi_req[%x] mpi_status[%x]", *mpi_req, mpi_status);
        MPI_Wait(mpi_req, mpi_status);
    }
}

void DistExecutor::Barrier() {
#pragma omp barrier
#pragma omp single
    {
        DistTask* dt;
        DistTask::CreateBarrier(&dt);
        DCT->ExecuteTask(dt);
        DistTask::Free(dt);
    }
}

void DistExecutor::SendFromHost(int dst, void* buf, size_t size, int tag, int flags) {
    int me = ME();
    bool win = WithInNode(me, dst);

    if (!win) {
        _trace("buf[%p] buf[%lf] tag[%d]", buf, ((double*) buf)[0], impacc::MPI_tag(tag, me, dst));
        MPI_Send(buf, (int) size, MPI_CHAR, NodeNum(dst), impacc::MPI_tag(tag, me, dst), MPI_COMM_WORLD);
        return;
    }

    DistTask* dt;
    DistTask::CreateSend(&dt, DIST_TASK_TYPE_SEND_H, me, dst, buf, size, win, tag, flags);
    DCT->ExecuteTask(dt);
    DistTask::Free(dt);
}

void DistExecutor::SendFromHostAsync(int dst, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async) {
    int me = ME();
    bool win = WithInNode(me, dst);

    if (!win) {
        _trace("buf[%p] buf[%lf] tag[%d]", buf, ((double*) buf)[0], impacc::MPI_tag(tag, me, dst));
        MPI_Isend(buf, (int) size, MPI_CHAR, NodeNum(dst), impacc::MPI_tag(tag, me, dst), MPI_COMM_WORLD, mpi_req);
        return;
    }

    DistTask* dt;
    DistTask::CreateSendAsync(&dt, DIST_TASK_TYPE_SEND_H, me, dst, buf, size, win, tag, mpi_req, flags, async);
    DCT->EnqueueTask(dt);
    AddAsync(dt);
    AddMPIRequest(dt);
}

void DistExecutor::SendFromDevice(int dst, void* buf, size_t size, int tag, int flags) {
    int me = ME();
    bool win = WithInNode(me, dst);

    DistTask* dt;
    DistTask::CreateSend(&dt, DIST_TASK_TYPE_SEND_D, me, dst, buf, size, win, tag, flags);

    if (win) {
        DCT->EnqueueTask(dt);
        dt->WaitForReady();
        if (dt->kind == HI_MemcpyDeviceToHost) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
            OpenCLDriver* driver = (OpenCLDriver*) dt->dev;
            if (driver == NULL) _error("invalid driver[%p]", driver);
            cl_command_queue queue = driver->getQueue(DEFAULT_QUEUE);
            if (queue == NULL) _error("invalid command queue[%p][%d]", queue, DEFAULT_QUEUE);
            cl_int clerr = clEnqueueReadBuffer(queue, (cl_mem) dt->sb, CL_TRUE, 0, dt->size, dt->rb, 0, NULL, NULL);
            if (clerr != CL_SUCCESS) _error("cl error[%d]", clerr);
#else
            CUresult cuerr = cuMemcpyDtoH(dt->rb, (CUdeviceptr) dt->sb, dt->size);
            if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
#endif
            dt->Complete();
            dt->pair->Complete();
        } else if (dt->kind == HI_MemcpyDeviceToDevice) {
            _check();
        } else _error("invalid receive type[%d]", dt->kind); 
    } else {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        dt->sb = malloc(size);
        OpenCLDriver* driver = (OpenCLDriver*) dt->dev;
        if (driver == NULL) _error("invalid driver[%p]", driver);
        cl_command_queue queue = driver->getQueue(DEFAULT_QUEUE);
        if (queue == NULL) _error("invalid command queue[%p][%d]", queue, DEFAULT_QUEUE);
        cl_int clerr = clEnqueueReadBuffer(queue, (cl_mem) buf, CL_TRUE, 0, dt->size, dt->sb, 0, NULL, NULL);
        if (clerr != CL_SUCCESS) _error("cl error[%d]", clerr);
        DCT->ExecuteTask(dt);
        free(dt->sb);
#else
        CUresult cuerr = cuMemAllocHost(&dt->sb, size);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
        cuerr = cuMemcpyDtoH(dt->sb, (CUdeviceptr) buf, size);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
        DCT->ExecuteTask(dt);
        cuerr = cuMemFreeHost(dt->sb);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
#endif
    }
    DistTask::Free(dt);
}

#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
#else
static void CUDA_CB cb_send_from_device_async(CUstream stream, CUresult status, void* userData) {
    DistTask* dt = (DistTask*) userData;
    gDE->DCT->EnqueueTask(dt);
}
#endif

void DistExecutor::SendFromDeviceAsync(int dst, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async) {
    int me = ME();
    bool win = WithInNode(me, dst);

    DistTask* dt;
    DistTask::CreateSendAsync(&dt, DIST_TASK_TYPE_SEND_D, me, dst, buf, size, win, tag, mpi_req, flags, async);

    if (win) {
        DCT->EnqueueTask(dt);
        dt->WaitForReady();
        if (dt->kind == HI_MemcpyDeviceToHost) {
            HI_set_async(async);
            HI_memcpy_async(dt->rb, dt->sb, size, HI_MemcpyDeviceToHost, 0, async);
            AddAsync(dt);
        } else if (dt->kind == HI_MemcpyDeviceToDevice) {
        } else _error("invalid receive type[%d]", dt->kind); 
    } else {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        dt->sb = malloc(size);
        OpenCLDriver* driver = (OpenCLDriver*) dt->dev;
        if (driver == NULL) _error("invalid driver[%p]", driver);
        cl_command_queue queue = driver->getQueue(DEFAULT_QUEUE);
        if (queue == NULL) _error("invalid command queue[%p][%d]", queue, DEFAULT_QUEUE);
        cl_int clerr = clEnqueueReadBuffer(queue, (cl_mem) buf, CL_TRUE, 0, size, dt->sb, 0, NULL, NULL);
        if (clerr != CL_SUCCESS) _error("cl error[%d]", clerr);
        DCT->ExecuteTask(dt);
        AddAsync(dt);
#else
        CUresult cuerr = cuMemAllocHost(&dt->sb, size);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
#if 1
        cuerr = cuMemcpyDtoH(dt->sb, (CUdeviceptr) buf, size);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
        DCT->EnqueueTask(dt);
#else
        HI_set_async(async);
        CudaDriver* driver = (CudaDriver*) dt->dev;
        CUstream stream = driver->getQueue(async);
        cuerr = cuMemcpyDtoHAsync(dt->sb, (CUdeviceptr) buf, size, stream);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
        CUstreamCallback callback = cb_send_from_device_async;
        cuerr = cuStreamAddCallback(stream, callback, (void*) dt, 0);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
#endif
        AddAsync(dt);
#endif
    }
}

void DistExecutor::RecvToHost(int src, void* buf, size_t size, int tag, int flags) {
    int me = ME();
    bool win = WithInNode(me, src);

    if (!win) {
        _trace("buf[%p] tag[%d]", buf, impacc::MPI_tag(tag, src, me));
        MPI_Recv(buf, (int) size, MPI_CHAR, NodeNum(src), impacc::MPI_tag(tag, src, me), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return;
    }

    DistTask* dt;
    DistTask::CreateRecv(&dt, DIST_TASK_TYPE_RECV_H, me, src, buf, size, win, tag, flags);
    DCT->ExecuteTask(dt);
    DistTask::Free(dt);
}

void DistExecutor::RecvToHostAsync(int src, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async) {
    int me = ME();
    bool win = WithInNode(me, src);

    if (!win) {
        MPI_Irecv(buf, (int) size, MPI_CHAR, NodeNum(src), impacc::MPI_tag(tag, src, me), MPI_COMM_WORLD, mpi_req);
        return;
    }

    DistTask* dt;
    DistTask::CreateRecvAsync(&dt, DIST_TASK_TYPE_RECV_H, me, src, buf, size, win, tag, mpi_req, flags, async);
    DCT->EnqueueTask(dt);
    AddAsync(dt);
}

void DistExecutor::RecvToDevice(int src, void* buf, size_t size, int tag, int flags) {
    int me = ME();
    bool win = WithInNode(me, src);

    DistTask* dt;
    DistTask::CreateRecv(&dt, DIST_TASK_TYPE_RECV_D, me, src, buf, size, win, tag, flags);

    if (!win) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        dt->sb = malloc(size);
#else
        CUresult cuerr = cuMemAllocHost(&dt->sb, size);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
#endif
    }

    DCT->EnqueueTask(dt);
    dt->WaitForReady();
    if (dt->kind == HI_MemcpyHostToDevice) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        OpenCLDriver* driver = (OpenCLDriver*) dt->dev;
        if (driver == NULL) _error("invalid driver[%p]", driver);
        cl_command_queue queue = driver->getQueue(DEFAULT_QUEUE);
        if (queue == NULL) _error("invalid command queue[%p][%d]", queue, DEFAULT_QUEUE);
        cl_int clerr = clEnqueueWriteBuffer(queue, (cl_mem) dt->rb, CL_TRUE, 0, dt->size, dt->sb, 0, NULL, NULL);
        if (clerr != CL_SUCCESS) _error("cl error[%d]", clerr);
        if (!win) {
            free(dt->sb);
        }
#else
        CUresult cuerr = cuMemcpyHtoD((CUdeviceptr) dt->rb, dt->sb, dt->size);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
        if (!win) {
            CUresult cuerr = cuMemFreeHost(dt->sb);
            if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
        }
#endif
    } else if (dt->kind == HI_MemcpyDeviceToDevice) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        OpenCLDriver* driver = (OpenCLDriver*) dt->dev;
        if (driver == NULL) _error("invalid driver[%p]", driver);
        cl_command_queue queue = driver->getQueue(DEFAULT_QUEUE);
        if (queue == NULL) _error("invalid command queue[%p][%d]", queue, DEFAULT_QUEUE);

        cl_int clerr = clEnqueueCopyBuffer(queue, (cl_mem) dt->sb, (cl_mem) dt->rb, 0, 0, dt->size, 0, NULL, NULL);
        if (clerr != CL_SUCCESS) _error("cl error[%d]", clerr);
        clerr = clFinish(queue);
        if (clerr != CL_SUCCESS) _error("cl error[%d]", clerr);
#else
        _check();
        CUresult cuerr;

        CudaDriver* driver_s = (CudaDriver*) dt->pair->dev;
        CudaDriver* driver_r = (CudaDriver*) dt->dev;

        CUdevice dev_s = driver_s->cuDevice;
        CUdevice dev_r = driver_r->cuDevice;

        CUcontext ctx_s = driver_s->cuContext;
        CUcontext ctx_r = driver_r->cuContext;

        bool canAccessPeer = CanAccessPeer(driver_s, driver_r);

        if (canAccessPeer) {
        _check();
            cuerr = cuMemcpyPeer((CUdeviceptr) dt->rb, ctx_r, (CUdeviceptr) dt->sb, ctx_s, dt->size);
            if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
        } else {
        _check();
            cuerr = cuMemcpy((CUdeviceptr) dt->rb, (CUdeviceptr) dt->sb, dt->size);
            if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
        }
        _check();
#endif
    } else _error("invalid receive type[%d]", dt->kind); 
    dt->Complete();
    if (dt->pair) dt->pair->Complete();
    DistTask::Free(dt);
}

void DistExecutor::RecvToDeviceAsync(int src, void* buf, size_t size, int tag, MPI_Request* mpi_req, int flags, int async) {
    int me = ME();
    bool win = WithInNode(me, src);

    DistTask* dt;
    DistTask::CreateRecvAsync(&dt, DIST_TASK_TYPE_RECV_D, me, src, buf, size, win, tag, mpi_req, flags, async);

    if (win) {
        DCT->EnqueueTask(dt);
        dt->WaitForReady();
        if (dt->kind == HI_MemcpyHostToDevice) {
            HI_set_async(async);
            HI_memcpy_async(dt->rb, dt->sb, dt->size, HI_MemcpyHostToDevice, 0, async);
            AddAsync(dt);
        } else if (dt->kind == HI_MemcpyDeviceToDevice) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
            HI_set_async(async);
            HI_memcpy_async(dt->rb, dt->sb, dt->size, HI_MemcpyDeviceToDevice, 0, async);
            AddAsync(dt);
#else
            CUresult cuerr;

            CudaDriver* driver_s = (CudaDriver*) dt->pair->dev;
            CudaDriver* driver_r = (CudaDriver*) dt->dev;

            CUdevice dev_s = driver_s->cuDevice;
            CUdevice dev_r = driver_r->cuDevice;

            CUcontext ctx_s = driver_s->cuContext;
            CUcontext ctx_r = driver_r->cuContext;

            bool canAccessPeer = CanAccessPeer(driver_s, driver_r);

            if (canAccessPeer) {
                HI_set_async(async);
                cuerr = cuMemcpyPeerAsync((CUdeviceptr) dt->rb, ctx_r, (CUdeviceptr) dt->sb, ctx_s, dt->size, driver_r->getQueue(async));
                if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
            } else {
                HI_set_async(async);
                HI_memcpy_async(dt->rb, dt->sb, dt->size, HI_MemcpyDeviceToDevice, 0, async);
            }
            AddAsync(dt);
#endif
        } else _error("invalid receive type[%d]", dt->kind); 
    } else {
#if defined(OPENARC_ARCH) && OPENARC_ARCH != 0
        dt->sb = malloc(size);
#else
        CUresult cuerr = cuMemAllocHost(&dt->sb, size);
        if (cuerr != CUDA_SUCCESS) _error("cuda error[%d]", cuerr);
#endif

        DCT->EnqueueTask(dt);
        dt->WaitForReady();
        HI_set_async(async);
        HI_memcpy_async(dt->rb, dt->sb, dt->size, HI_MemcpyHostToDevice, 0, async);
        AddAsync(dt);
        //HERE
    }
}

void DistExecutor::ReduceFromHostToHost(void* sb, void* rb, size_t size, acc_datatype datatype, acc_op op, int root) {
    int me = ME();
    int tid = get_thread_id();

    DistTask* dt;
    DistTask::CreateReduce(&dt, DIST_TASK_TYPE_REDUCE_HH, me, tid, sb, rb, size, datatype, op, root);

    DCT->ExecuteTask(dt);
    DistTask::Free(dt);
}

void DistExecutor::AllReduceFromHostToHost(void* sb, void* rb, size_t size, acc_datatype datatype, acc_op op) {
    int me = ME();
    int tid = get_thread_id();

    DistTask* dt;
    DistTask::CreateAllReduce(&dt, DIST_TASK_TYPE_ALLREDUCE_HH, me, tid, sb, rb, size, datatype, op);

    DCT->ExecuteTask(dt);
    DistTask::Free(dt);
}

void DistExecutor::Bcast(int root, void* buf, size_t size) {
    int me = ME();
    int tid = get_thread_id();

    DistTask* dt;
    DistTask::CreateBcast(&dt, DIST_TASK_TYPE_BCAST, me, root, buf, size);

    DCT->ExecuteTask(dt);
    DistTask::Free(dt);
}

void DistExecutor::TypeCommit(void* datatype) {
    DistTask* dt;
    DistTask::Create(&dt);
    dt->type = DIST_TASK_TYPE_TYPECOMMIT;
    dt->rb = datatype;
    DCT->ExecuteTask(dt);
    DistTask::Free(dt);
}

DistCommThread::DistCommThread(DistExecutor* DE) {
    this->DE = DE;
    thread = (pthread_t) NULL;
    running = false;
    pthread_mutex_init(&mutex_pendings, NULL);
    pthread_mutex_init(&mutex_asyncs, NULL);
    sem_init(&sem_schedule, 0, 0);
    busy_waiting = true;

    queue = new LockFreeQueueMS<DistTask*>(1024);

    last_c_task = NULL;
    c_task_idx = 0;
}

DistCommThread::~DistCommThread() {
    Stop();
    pthread_mutex_destroy(&mutex_pendings);
    pthread_mutex_destroy(&mutex_asyncs);
    sem_destroy(&sem_schedule);
    delete queue;
}

void DistCommThread::EnqueueTask(DistTask* task) {
    task->Submit();
    while (!queue->Enqueue(task)) {}
    if (!busy_waiting) sem_post(&sem_schedule);
}

void DistCommThread::ExecuteTask(DistTask* task) {
    EnqueueTask(task);
    task->Wait();
}

void DistCommThread::Start() {
    if (!thread) {
        running = true;
        pthread_create(&thread, NULL, &DistCommThread::ThreadFunc, this);
#if 0
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int i = 10; i < 16; i++) {
            CPU_SET(i, &cpuset);
        }
        pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#endif
    }
}

void DistCommThread::Stop() {
    if (thread) {
        running = false;
        pthread_join(thread, NULL);
        thread = (pthread_t) NULL;
    }
}

void DistCommThread::Run() {
    while (running) {
        DistTask* task;
        if (!busy_waiting) sem_wait(&sem_schedule);
        while (queue->Dequeue(&task)) HandleTask(task);
        if (asyncs.empty()) {
            busy_waiting = false;
        } else {
            busy_waiting = true;
            CheckAsyncs();
        }
    }
}

void DistCommThread::CheckAsyncs() {
    vector<DistTask*>::iterator it = asyncs.begin();
    while (it != asyncs.end()) {
        DistTask* task = *it;
        int flag;
        MPI_Test(task->mpi_req, &flag, MPI_STATUS_IGNORE);
        if (flag) {
            if (task->type == DIST_TASK_TYPE_RECV_D) task->Ready();
            else task->Complete();
            asyncs.erase(it);
        } else ++it;
    }
}

void DistCommThread::HandleTask(DistTask* task) {
    _trace("id[%lu] type[%d] src[%d] dst[%d] sb[%p] rb[%p] size[%lu] tag[%d] win[%d]", task->id, task->type, task->src, task->dst, task->sb, task->rb, task->size, task->tag, task->win);
    switch (task->type) {
        case DIST_TASK_TYPE_SEND_H:
        case DIST_TASK_TYPE_SEND_D:         HandleTaskSend(task);       break;
        case DIST_TASK_TYPE_RECV_H:
        case DIST_TASK_TYPE_RECV_D:         HandleTaskRecv(task);       break;
        case DIST_TASK_TYPE_BCAST:          HandleTaskBcast(task);      break;
        case DIST_TASK_TYPE_REDUCE_HH:      HandleTaskReduce(task);     break;
        case DIST_TASK_TYPE_ALLREDUCE_HH:   HandleTaskAllReduce(task);  break;
        case DIST_TASK_TYPE_BARRIER:        HandleTaskBarrier(task);    break;
        case DIST_TASK_TYPE_TYPECOMMIT:     HandleTaskTypeCommit(task); break;
        default: _error("INVALID DIST TASK TYPE[%d]", task->type);
    }
}

void DistCommThread::HandleTaskSend(DistTask* task) {
    if (task->win) HandleTaskSendWin(task);
    else HandleTaskSendAcn(task);
}

void DistCommThread::HandleTaskPairWin(DistTask* send, DistTask* recv) {
    if (send->type == DIST_TASK_TYPE_SEND_H && recv->type == DIST_TASK_TYPE_RECV_H) {
        if ((send->flags & IMPACC_MEM_S_RO) && (recv->flags & IMPACC_MEM_R_RO)) {
            if (!DE->NDM->Alias(recv->rb, send->sb, recv->size)) {
                memcpy(recv->rb, send->sb, recv->size);
            }
        } else {
            memcpy(recv->rb, send->sb, recv->size);
        }
        send->Complete();
        recv->Complete();
    } else if (send->type == DIST_TASK_TYPE_SEND_H && recv->type == DIST_TASK_TYPE_RECV_D) {
        recv->sb = send->sb;
        recv->kind = HI_MemcpyHostToDevice;
        recv->pair = send;
        recv->Ready();
    } else if (send->type == DIST_TASK_TYPE_SEND_D && recv->type == DIST_TASK_TYPE_RECV_H) {
        send->rb = recv->rb;
        send->kind = HI_MemcpyDeviceToHost;
        send->size = recv->size;
        send->pair = recv;
        send->Ready();
    } else if (send->type == DIST_TASK_TYPE_SEND_D && recv->type == DIST_TASK_TYPE_RECV_D) {
        recv->sb = send->sb;
        recv->kind = send->kind = HI_MemcpyDeviceToDevice;
        recv->pair = send;
        recv->Ready();
    } else {
        _error("INVALID TYPE SEND[%d] RECV[%d]", send->type, recv->type);
    }
}

void DistCommThread::HandleTaskSendWin(DistTask* task) {
    bool found = false;
    pthread_mutex_lock(&mutex_pendings);
    for (vector<DistTask*>::iterator it = pendings.begin(); it != pendings.end(); ++it) {
        DistTask* pdt = *it;
        if (task->tag == pdt->tag && task->me == pdt->src && task->dst == pdt->me) {
            HandleTaskPairWin(task, pdt);
            found = true;
            pendings.erase(it);
            break;
        }
    }
    if (!found) pendings.push_back(task);
    pthread_mutex_unlock(&mutex_pendings);
}

void DistCommThread::HandleTaskSendAcn(DistTask* task) {
    if (task->Async()) {
        MPI_Isend(task->sb, (int) task->size, MPI_CHAR, DE->NodeNum(task->dst), task->tag, MPI_COMM_WORLD, task->mpi_req);
        asyncs.push_back(task);
    } else {
        _trace("sb[%p] size[%lu] dst[%d] rank[%d] tag[%d] ttag[%d]", task->sb, task->size, task->dst, DE->NodeNum(task->dst), task->tag, impacc::MPI_tag(task->tag, task->me, task->dst));
        MPI_Send(task->sb, (int) task->size, MPI_CHAR, DE->NodeNum(task->dst), impacc::MPI_tag(task->tag, task->me, task->dst), MPI_COMM_WORLD);
        task->Complete();
    }
}

void DistCommThread::HandleTaskRecv(DistTask* task) {
    if (task->win) HandleTaskRecvWin(task);
    else HandleTaskRecvAcn(task);
}

void DistCommThread::HandleTaskRecvWin(DistTask* task) {
    bool found = false;
    pthread_mutex_lock(&mutex_pendings);
    for (vector<DistTask*>::iterator it = pendings.begin(); it != pendings.end(); ++it) {
        DistTask* pdt = *it;
        if (task->tag == pdt->tag && task->me == pdt->dst && task->src == pdt->me) {
            HandleTaskPairWin(pdt, task);
            found = true;
            pendings.erase(it);
            break;
        }
    }
    if (!found) pendings.push_back(task);
    pthread_mutex_unlock(&mutex_pendings);
}

void DistCommThread::HandleTaskRecvAcn(DistTask* task) {
    if (task->Async()) {
        if (task->type == DIST_TASK_TYPE_RECV_H) {
            MPI_Irecv(task->rb, (int) task->size, MPI_CHAR, DE->NodeNum(task->src), task->tag, MPI_COMM_WORLD, task->mpi_req);
            asyncs.push_back(task);
        } else if (task->type == DIST_TASK_TYPE_RECV_D) {
            MPI_Irecv(task->sb, (int) task->size, MPI_CHAR, DE->NodeNum(task->src), task->tag, MPI_COMM_WORLD, task->mpi_req);
            asyncs.push_back(task);
        } else _error("INVALID TASK TYPE[%d]", task->type);
        asyncs.push_back(task);
    } else {
#ifdef GPU_DIRECT
        MPI_Recv(task->rb, (int) task->size, MPI_CHAR, DE->NodeNum(task->src), task->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        task->Complete();
#else
        if (task->type == DIST_TASK_TYPE_RECV_H) {
            _trace("rb[%p] size[%lu] src[%d] rank[%d] tag[%d] ttag[%d]", task->rb, task->size, task->src, DE->NodeNum(task->src), task->tag, impacc::MPI_tag(task->tag, task->src, task->me));
            MPI_Recv(task->rb, (int) task->size, MPI_CHAR, DE->NodeNum(task->src), impacc::MPI_tag(task->tag, task->src, task->me), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            task->Complete();
        } else if (task->type == DIST_TASK_TYPE_RECV_D) {
            MPI_Recv(task->sb, (int) task->size, MPI_CHAR, DE->NodeNum(task->src), task->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            task->kind = HI_MemcpyHostToDevice;
            task->Ready();
        } else _error("INVALID TASK TYPE[%d]", task->type);
#endif
    }
}

void DistCommThread::HandleTaskBcast(DistTask* task) {
    if (last_c_task == NULL) {
        last_c_task = task;
        last_c_task->ci = 0;
    }

    last_c_task->ci++;

    if (last_c_task->ci > 1 && last_c_task->ci <= DE->numdevs) {
        DistTask* p = last_c_task;
        while (p->pair) {
            p = p->pair;
        }
        p->pair = task;
    }
    if (last_c_task->ci == DE->numdevs) {
        bool contains_root = gDE->WithInNode(last_c_task->me, task->src);
        if (contains_root) {
            void* src = NULL;
            for (DistTask* p = last_c_task; p; p = p->pair) {
                if (p->me == p->src) {
                    src = p->sb;
                    break;
                }
            }
            if (src == NULL) _error("root[%d] is not in the node", task->src);
            for (DistTask* p = last_c_task; p; p = p->pair) {
                if (p->me != p->src) memcpy(p->rb, src, p->size);
            }
        }

        MPI_Bcast(last_c_task->sb, (int) task->size, MPI_CHAR, DE->NodeNum(task->src), MPI_COMM_WORLD);

        DistTask* p = last_c_task->pair;
        while (p) {
            if (!contains_root) memcpy(p->rb, last_c_task->rb, last_c_task->size);
            p->Complete();
            p = p->pair;
        }

        last_c_task->Complete();
        last_c_task = NULL;
    }
}

void DistCommThread::HandleTaskReduce(DistTask* task) {
    if (last_c_task == NULL) {
        last_c_task = task;
        last_c_task->ci = 0;
    }

    last_c_task->ci++;

    if (last_c_task->ci > 1 && last_c_task->ci <= DE->numdevs) {
        DistTask* p = last_c_task;
        while (p->pair) {
            p = p->pair;
        }
        p->pair = task;
    }
    if (last_c_task->ci == DE->numdevs) {
#if 1
        MPI_Datatype mpi_datatype = task->datatype;
        MPI_Op mpi_op = task->op;
#else
        MPI_Datatype mpi_datatype;
        MPI_Op mpi_op;
        switch (task->datatype) {
            case acc_float:     mpi_datatype = MPI_FLOAT;   break;
            case acc_double:    mpi_datatype = MPI_DOUBLE;  break;
        }
        switch (task->op) {
            case acc_sum:       mpi_op = MPI_SUM;           break;
            case acc_max:       mpi_op = MPI_MAX;           break;
            case acc_min:       mpi_op = MPI_MIN;           break;
        }
#endif

        DistTask* p = last_c_task;
        void* rb = last_c_task->rb;
        while (p->pair) {
            p = p->pair;
            if (mpi_datatype == MPI_DOUBLE && mpi_op == MPI_SUM) {
                double* rbuf = (double*) last_c_task->sb;
                double* pbuf = (double*) p->sb;
                for (int i = 0; i < task->size; i++) {
                    rbuf[i] += pbuf[i];
                }
            } else if (mpi_datatype == MPI_DOUBLE && mpi_op == MPI_MAX) {
                double* rbuf = (double*) last_c_task->sb;
                double* pbuf = (double*) p->sb;
                for (int i = 0; i < task->size; i++) {
                    if (rbuf[i] < pbuf[i]) rbuf[i] = pbuf[i];
                }
            } else if (mpi_datatype == MPI_DOUBLE && mpi_op == MPI_MIN) {
                double* rbuf = (double*) last_c_task->sb;
                double* pbuf = (double*) p->sb;
                for (int i = 0; i < task->size; i++) {
                    if (rbuf[i] > pbuf[i]) rbuf[i] = pbuf[i];
                }
            } else {
                _error("IMPLEMENT THIS CASE datatype[%d] op[%d]", task->datatype, task->op);
            }
            if (p->me == p->dst) rb = p->rb;
        }

        MPI_Reduce(last_c_task->sb, rb, (int) task->size, mpi_datatype, mpi_op, DE->NodeNum(last_c_task->dst), MPI_COMM_WORLD);

        p = last_c_task->pair;
        while (p) {
            p->Complete();
            p = p->pair;
        }
        last_c_task->Complete();
        last_c_task = NULL;
    }
}

void DistCommThread::HandleTaskAllReduce(DistTask* task) {
#if 1
    size_t tsize = task->datatype == MPI_DOUBLE ? sizeof(double) : sizeof(float);
    size_t bsize = tsize * task->size;
#else
    size_t tsize = task->datatype == acc_double ? sizeof(double) : sizeof(float);
    size_t bsize = tsize * task->size;
#endif

    if (last_c_task == NULL) {
        last_c_task = task;
        last_c_task->ci = 0;
    }

    last_c_task->ci++;

    if (last_c_task->ci > 1 && last_c_task->ci <= DE->numdevs) {
        DistTask* p = last_c_task;
        while (p->pair) {
            p = p->pair;
        }
        p->pair = task;
    }
    if (last_c_task->ci == DE->numdevs) {
#if 1
        MPI_Datatype mpi_datatype = task->datatype;
        MPI_Op mpi_op = task->op;
#else
        MPI_Datatype mpi_datatype;
        MPI_Op mpi_op;
        switch (task->datatype) {
            case acc_float:     mpi_datatype = MPI_FLOAT;   break;
            case acc_double:    mpi_datatype = MPI_DOUBLE;  break;
        }
        switch (task->op) {
            case acc_sum:       mpi_op = MPI_SUM;           break;
            case acc_max:       mpi_op = MPI_MAX;           break;
            case acc_min:       mpi_op = MPI_MIN;           break;
        }
#endif

        DistTask* p = last_c_task;
        while (p->pair) {
            p = p->pair;
            if (mpi_datatype == MPI_DOUBLE && mpi_op == MPI_SUM) {
                double* rbuf = (double*) last_c_task->sb;
                double* pbuf = (double*) p->sb;
                for (int i = 0; i < task->size; i++) {
                    rbuf[i] += pbuf[i];
                }
            } else if (mpi_datatype == MPI_DOUBLE && mpi_op == MPI_MAX) {
                double* rbuf = (double*) last_c_task->sb;
                double* pbuf = (double*) p->sb;
                for (int i = 0; i < task->size; i++) {
                    if (rbuf[i] < pbuf[i]) rbuf[i] = pbuf[i];
                }
            } else if (mpi_datatype == MPI_DOUBLE && mpi_op == MPI_MIN) {
                double* rbuf = (double*) last_c_task->sb;
                double* pbuf = (double*) p->sb;
                for (int i = 0; i < task->size; i++) {
                    if (rbuf[i] > pbuf[i]) rbuf[i] = pbuf[i];
                }
            } else {
                _error("IMPLEMENT THIS CASE datatype[%d] op[%d]", task->datatype, task->op);
            }
        }

        MPI_Allreduce(last_c_task->sb, last_c_task->rb, (int) task->size, mpi_datatype, mpi_op, MPI_COMM_WORLD);

        p = last_c_task->pair;
        while (p) {
            memcpy(p->rb, last_c_task->rb, bsize);
            p->Complete();
            p = p->pair;
        }
        last_c_task->Complete();
        last_c_task = NULL;
    }
}

void DistCommThread::HandleTaskBarrier(DistTask* task) {
    MPI_Barrier(MPI_COMM_WORLD);
    task->Complete();
}

void DistCommThread::HandleTaskTypeCommit(DistTask* task) {
    MPI_Datatype* datatype = (MPI_Datatype*) task->rb;
    MPI_Type_commit(datatype);
    task->Complete();
}

void* DistCommThread::ThreadFunc(void* argp) {
    ((DistCommThread*) argp)->Run();
    return NULL;
}

unsigned long _NewID() {
    static unsigned long uid = 0;
    unsigned long new_uid;
    do {
        new_uid = uid + 1;
    } while (!__sync_bool_compare_and_swap(&uid, uid, new_uid));
    return new_uid;
}

DistTask::DistTask(unsigned long id) {
    this->id = id;
    status = DIST_TASK_INIT;
}

DistTask::~DistTask() {
    if (status == DIST_TASK_INIT) return;
    sem_destroy(&sem_ready);
    sem_destroy(&sem_complete);
}

void DistTask::Create(DistTask** newtask) {
    *newtask = new DistTask(_NewID());
    (*newtask)->Clear();
}

void DistTask::CreateSend(DistTask** newtask, int type, int me, int dst, void* sb, size_t size, bool win, int tag, int flags) {
    Create(newtask);
    (*newtask)->type = type;
    (*newtask)->me = me;
    (*newtask)->dst = dst;
    (*newtask)->sb = sb;
    (*newtask)->size = size;
    (*newtask)->win = win;
    (*newtask)->tag = tag;
    (*newtask)->flags = flags;
    if (type == DIST_TASK_TYPE_SEND_D) {
        (*newtask)->dev = getHostConf()->device;
    }
}

void DistTask::CreateSendAsync(DistTask** newtask, int type, int me, int dst, void* sb, size_t size, bool win, int tag, MPI_Request* mpi_req, int flags, int async) {
    CreateSend(newtask, type, me, dst, sb, size, win, tag, flags);
    (*newtask)->mpi_req = mpi_req;
    (*newtask)->async = async;
}

void DistTask::CreateRecv(DistTask** newtask, int type, int me, int src, void* rb, size_t size, bool win, int tag, int flags) {
    Create(newtask);
    (*newtask)->type = type;
    (*newtask)->me = me;
    (*newtask)->src = src;
    (*newtask)->rb = rb;
    (*newtask)->size = size;
    (*newtask)->win = win;
    (*newtask)->tag = tag;
    (*newtask)->flags = flags;
    if (type == DIST_TASK_TYPE_RECV_D) {
        (*newtask)->dev = getHostConf()->device;
    }
}

void DistTask::CreateRecvAsync(DistTask** newtask, int type, int me, int src, void* rb, size_t size, bool win, int tag, MPI_Request* mpi_req, int flags, int async) {
    CreateRecv(newtask, type, me, src, rb, size, win, tag, flags);
    (*newtask)->mpi_req = mpi_req;
    (*newtask)->async = async;
}

void DistTask::CreateReduce(DistTask** newtask, int type, int me, int src, void* sb, void* rb, size_t size, acc_datatype datatype, acc_op op, int root) {
    Create(newtask);
    (*newtask)->type = type;
    (*newtask)->me = me;
    (*newtask)->src = src;
    (*newtask)->sb = sb;
    (*newtask)->rb = rb;
    (*newtask)->size = size;
    (*newtask)->datatype = datatype;
    (*newtask)->op = op;
    (*newtask)->dst = root;
}

void DistTask::CreateAllReduce(DistTask** newtask, int type, int me, int src, void* sb, void* rb, size_t size, acc_datatype datatype, acc_op op) {
    Create(newtask);
    (*newtask)->type = type;
    (*newtask)->me = me;
    (*newtask)->src = src;
    (*newtask)->sb = sb;
    (*newtask)->rb = rb;
    (*newtask)->size = size;
    (*newtask)->datatype = datatype;
    (*newtask)->op = op;
}

void DistTask::CreateBcast(DistTask** newtask, int type, int me, int root, void* buf, size_t size) {
    Create(newtask);
    (*newtask)->type = type;
    (*newtask)->me = me;
    (*newtask)->src = root;
    (*newtask)->sb = buf;
    (*newtask)->rb = buf;
    (*newtask)->size = size;
}

void DistTask::CreateBarrier(DistTask** newtask) {
    Create(newtask);
    (*newtask)->type = DIST_TASK_TYPE_BARRIER;
}

void DistTask::Free(DistTask* task) {
    delete task;
    task = NULL;
}

void DistTask::Clear() {
    status = DIST_TASK_INIT;
    type = DIST_TASK_TYPE_NULL;
    src = -1;
    dst = -1;
    sb = NULL;
    rb = NULL;
    size = 0UL;
    tag = -1;
    flags = -1;
    async = -1;
    kind = -1;
    mpi_req = NULL;
    pair = NULL;
    ci = -1;
    cb = NULL;
    mpi_datatype = 0;
}

bool DistTask::Async() {
    return async != -1;
}

void DistTask::Submit() {
    if (status != DIST_TASK_INIT) return;
    sem_init(&sem_ready, 0, 0);
    sem_init(&sem_complete, 0, 0);
    status = DIST_TASK_SUBMITTED;
}

void DistTask::Run() {
    status = DIST_TASK_RUNNING;
}

void DistTask::Ready() {
    status = DIST_TASK_READY;
    sem_post(&sem_ready);
}

void DistTask::Wait() {
    if (status == DIST_TASK_COMPLETE) return;
    sem_wait(&sem_complete);
}

void DistTask::WaitForReady() {
    if (status >= DIST_TASK_READY) return;
    status = DIST_TASK_WAITFORREADY;
    sem_wait(&sem_ready);
}

void DistTask::Complete() {
    if (status == DIST_TASK_COMPLETE) return;
    if (status == DIST_TASK_WAITFORREADY) Ready();
    status = DIST_TASK_COMPLETE;
    sem_post(&sem_complete);
}

int acc_get_num_tasks() {
    return gDE->NumTasks();
}

int acc_get_task_num() {
    return gDE->ME();
}

void acc_mem_send_from_host(int dst, h_void* buf, size_t size, int tag, int flags) {
    _trace("dst[%d] buf[%p] size[%lu] tag[%d] flags[%d] ", dst, buf, size, tag, flags);
    gDE->SendFromHost(dst, buf, size, tag, flags);
}

void acc_mem_send_from_host_async(int dst, h_void* buf, size_t size, int tag, MPI_Request *mpi_req, int flags, int async) {
    _trace("dst[%d] buf[%p] size[%lu] tag[%d] mpi_req[%p] async[%d]", dst, buf, size, tag, mpi_req, flags, async);
    gDE->SendFromHostAsync(dst, buf, size, tag, mpi_req, flags, async);
}

void acc_mem_send_from_device(int dst, d_void* buf, size_t size, int tag, int flags) {
    _trace("dst[%d] buf[%p] size[%lu] tag[%d] flags[%d]", dst, buf, size, tag, flags);
    gDE->SendFromDevice(dst, buf, size, tag, flags);
}

void acc_mem_send_from_device_async(int dst, d_void* buf, size_t size, int tag, MPI_Request *mpi_req, int flags, int async) {
    _trace("dst[%d] buf[%p] size[%lu] tag[%d] mpi_req[%p] flags[%d] async[%d]", dst, buf, size, tag, mpi_req, flags, async);
    gDE->SendFromDeviceAsync(dst, buf, size, tag, mpi_req, flags, async);
}

void acc_mem_recv_to_host(int src, h_void* buf, size_t size, int tag, int flags) {
    _trace("src[%d] buf[%p] size[%lu] tag[%d] flags[%d]", src, buf, size, tag, flags);
    gDE->RecvToHost(src, buf, size, tag, flags);
}

void acc_mem_recv_to_host_async(int src, h_void* buf, size_t size, int tag, MPI_Request *mpi_req, int flags, int async) {
    _trace("src[%d] buf[%p] size[%lu] tag[%d] mpi_req[%p] flags[%d]  async[%d]", src, buf, size, tag, mpi_req, flags, async);
    gDE->RecvToHostAsync(src, buf, size, tag, mpi_req, flags, async);
}

void acc_mem_recv_to_device(int src, d_void* buf, size_t size, int tag, int flags) {
    _trace("src[%d] buf[%p] size[%lu] tag[%d] flags[%d]", src, buf, size, tag, flags);
    gDE->RecvToDevice(src, buf, size, tag, flags);
}

void acc_mem_recv_to_device_async(int src, d_void* buf, size_t size, int tag, MPI_Request *mpi_req, int flags, int async) {
    _trace("src[%d] buf[%p] size[%lu] tag[%d] mpi_req[%p] flags[%d]  async[%d]", src, buf, size, tag, mpi_req, flags, async);
    gDE->RecvToDeviceAsync(src, buf, size, tag, mpi_req, flags, async);
}

void acc_mem_send(int dst, void* buf, size_t size, int tag, int flags) {
    if (acc_hostptr(buf) == NULL) acc_mem_send_from_host(dst, (h_void*) buf, size, tag, flags);
    else acc_mem_send_from_device(dst, (d_void*) buf, size, tag, flags);
}

void acc_mem_send_async(int dst, void* buf, size_t size, int tag, MPI_Request *mpi_req, int flags, int async) {
    if (acc_hostptr(buf) == NULL) acc_mem_send_from_host_async(dst, (h_void*) buf, size, tag, mpi_req, flags, async);
    else acc_mem_send_from_device_async(dst, (d_void*) buf, size, tag, mpi_req, flags, async);
}

void acc_mem_recv(int src, void* buf, size_t size, int tag, int flags) {
    if (acc_hostptr(buf) == NULL) acc_mem_recv_to_host(src, (h_void*) buf, size, tag, flags);
    else acc_mem_recv_to_device(src, (d_void*) buf, size, tag, flags);
}

void acc_mem_recv_async(int src, void* buf, size_t size, int tag, MPI_Request *mpi_req, int flags, int async) {
    if (acc_hostptr(buf) == NULL) acc_mem_recv_to_host_async(src, (h_void*) buf, size, tag, mpi_req, flags, async);
    else acc_mem_recv_to_device_async(src, (d_void*) buf, size, tag, mpi_req, flags, async);
}

void acc_mem_wait(MPI_Request* mpi_req, MPI_Status* mpi_status) {
    gDE->WaitMPIRequest(mpi_req, mpi_status);
}

void acc_mem_wait_host(int async) {
    gDE->Wait(async);
}

void acc_mem_wait_device(int async) {
    acc_wait(async);
    gDE->Wait(async);
}

void acc_mem_bcast(int root, void* buf, size_t size) {
    _trace("root[%d] buf[%p] size[%lu]", root, buf, size);
    gDE->Bcast(root, buf, size);
}

void acc_mem_reduce_from_host_to_host(void* sb, void* rb, size_t size, acc_datatype datatype, acc_op op, int root) {
    _trace("sb[%p] rb[%p] size[%lu] type[%d] op[%d] root[%d]", sb, rb, size, datatype, op, root);
    gDE->ReduceFromHostToHost(sb, rb, size, datatype, op, root);
}

void acc_mem_allreduce_from_host_to_host(void* sb, void* rb, size_t size, acc_datatype datatype, acc_op op) {
    _trace("sb[%p] rb[%p] size[%lu] type[%d] op[%d]", sb, rb, size, datatype, op);
    gDE->AllReduceFromHostToHost(sb, rb, size, datatype, op);
}

void acc_barrier() {
    gDE->Barrier();
}

void acc_type_commit(void* datatype) {
    gDE->TypeCommit(datatype);
}

void* acc_mem_malloc(size_t size, void** ptr) {
    void* p = malloc(size);
    if (gDE->numdevs < 2) return p;
    gDE->NDM->AddData(p, size, ptr);

    return p;
}

void* acc_mem_calloc(size_t count, size_t size, void** ptr) {
    void* p = calloc(count, size);
    if (gDE->numdevs < 2) return p;
    gDE->NDM->AddData(p, count * size, ptr);

    return p;
}

void acc_mem_free(void* ptr) {
    if (gDE->numdevs < 2) return free(ptr);
    gDE->NDM->FreeData(ptr);
}

NodeData::NodeData(void* addr, size_t size, void** ptr) {
    this->addr = addr;
    this->size = size;
    this->ptr = ptr;
    ref_cnt = 1;
}

NodeData::~NodeData() {

}

NodeDataManager::NodeDataManager(DistExecutor* DE) {
    this->DE = DE;
    pthread_mutex_init(&mutex_pool, NULL);
}

NodeDataManager::~NodeDataManager() {
    pthread_mutex_destroy(&mutex_pool);
}

void NodeDataManager::AddData(void* addr, size_t size, void** ptr) {
    pthread_mutex_lock(&mutex_pool);
    NodeData* data = new NodeData(addr, size, ptr);
    pool[addr] = data; 
    pthread_mutex_unlock(&mutex_pool);
}

void NodeDataManager::FreeData(void* addr) {
    pthread_mutex_lock(&mutex_pool);
    NodeData* fd = NULL;
    if (pool.find(addr) != pool.end()) {
        fd = pool[addr];
    } else  {
        for (map<void*, NodeData*>::iterator it = pool.begin(); it != pool.end(); ++it) {
            NodeData* data = it->second;
            if (data->addr <= addr && (size_t) data->addr + data->size > (size_t) addr) {
                fd = data;
                break;
            }
        }
    }

    if (fd != NULL && --fd->ref_cnt == 0) {
        free(fd->addr);
    }
    pthread_mutex_unlock(&mutex_pool);
}

bool NodeDataManager::Alias(void* dst, void* src, size_t size) {
    pthread_mutex_lock(&mutex_pool);
    if (pool.find(dst) == pool.end()) {
        pthread_mutex_unlock(&mutex_pool);
        return false;
    }
    NodeData* a_dst = pool[dst];
    if (a_dst->size != size) {
        pthread_mutex_unlock(&mutex_pool);
        return false;
    }

    NodeData* a_src = NULL;
    for (map<void*, NodeData*>::iterator it = pool.begin(); it != pool.end(); ++it) {
        NodeData* data = it->second;
        if (data->addr <= src && (size_t) data->addr + data->size > (size_t) src) {
            a_src = data;
            break;
        }
    }

    if (a_src == NULL) {
        pthread_mutex_unlock(&mutex_pool);
        return false;
    }

    _error("dst[%p, %lu, %d, %p] src[%p, %lu, %d, %p]", a_dst->addr, a_dst->size, a_dst->ref_cnt, a_dst->ptr, a_src->addr, a_src->size, a_src->ref_cnt, a_src->ptr);

    *a_dst->ptr = src;

    a_src->ref_cnt++;

    free(a_dst->addr);
    pool.erase(dst);

    pthread_mutex_unlock(&mutex_pool);
    return true;
}

