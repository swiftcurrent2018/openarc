#ifndef __IMPACC_QUEUE_H__
#define __IMPACC_QUEUE_H__

template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue(unsigned long size) {
        this->size = size;
        idx_r = 0;
        idx_w = 0;
        elements = (volatile T*)(new T[size]);
    }

    virtual ~LockFreeQueue() {
        delete[] elements;
    }

    virtual bool Enqueue(T element) {
        unsigned long next_idx_w = (idx_w + 1) % size;
        if (next_idx_w == idx_r) return false;
        elements[idx_w] = element;
        __sync_synchronize();
        idx_w = next_idx_w;
        return true;
    }

    bool Dequeue(T* element) {
        if (idx_r == idx_w) return false;
        unsigned long next_idx_r = (idx_r + 1) % size;
        *element = (T) elements[idx_r];
        idx_r = next_idx_r;
        return true;
    }

    bool Peek(T* element) {
        if (idx_r == idx_w) return false;
        *element = (T) elements[idx_r];
        return true;
    }

    unsigned long Size() {
        if (idx_w >= idx_r) return idx_w - idx_r;
        return size - idx_r + idx_w;
    }

    bool Empty() {
        return Size() == 0UL;
    }

protected:
    unsigned long size;
    volatile T* elements;
    volatile unsigned long idx_r;
    volatile unsigned long idx_w;
};

// Multiple Producers & Single Consumer
template<typename T>
class LockFreeQueueMS: public LockFreeQueue<T> {
public:
    LockFreeQueueMS(unsigned long size) : LockFreeQueue<T>::LockFreeQueue(size) {
        idx_w_cas = 0;
    }
    ~LockFreeQueueMS() {}

    bool Enqueue(T element) {
        while (true) {
            unsigned long prev_idx_w = idx_w_cas;
            unsigned long next_idx_w = (prev_idx_w + 1) % this->size;
            if (next_idx_w == this->idx_r) return false;
            if (__sync_bool_compare_and_swap(&idx_w_cas, prev_idx_w, next_idx_w)) {
                this->elements[prev_idx_w] = element;
                while (!__sync_bool_compare_and_swap(&this->idx_w, prev_idx_w, next_idx_w)) {}
                break;
            }
        }
        return true;
    }

private:
    volatile unsigned long idx_w_cas;
};

#endif /* __IMPACC_QUEUE_H__ */
