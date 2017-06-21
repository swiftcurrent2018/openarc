#include "omp_helper.h"
#include "omp_helper_ext.h"

int q_max = OMP_HELPER_QUEUE_MAX;
int q_off = OMP_HELPER_QUEUE_OFF;

omp_depend* depends[OMP_HELPER_QUEUE_DEPTH];
int depends_args[OMP_HELPER_QUEUE_DEPTH];
std::vector<omp_depend*> queues[OMP_HELPER_QUEUE_DEPTH][OMP_HELPER_QUEUE_MAX];
int depth;

int q_wait_bits[OMP_HELPER_QUEUE_DEPTH];
int q_last_used[OMP_HELPER_QUEUE_DEPTH];
int q_cur;

#ifdef _OPENMP
#pragma omp threadprivate(depends, depends_args, queues, depth, q_wait_bits, q_last_used, q_cur)
#endif

void omp_helper_set_queue_max(int max) {
    q_max = max;
}

void omp_helper_set_queue_off(int off) {
    q_off = off;
}

void omp_helper_task_depend_check(omp_depend* d, int* wait_bits) {
    for (int i = 0; i < q_max; i++) {
        std::vector<omp_depend*> *q = queues[depth] + i;
        for (std::vector<omp_depend*>::iterator it = q->begin(); it != q->end(); ++it) {
            omp_depend* qr = *it;
            //printf("[%s:%d] qr[%llx:%llx] d[%llx,%llx]\n", __FILE__, __LINE__, qr->s, qr->e, d->s, d->e);
            if (qr->e < d->s || d->e < qr->s) continue;
            *wait_bits |= (1 << i);
            break;
        }
    }
}

void omp_helper_task_enter(int args, int* types, void** values) {
    int wait_bits = 0;
    omp_depend* d = new omp_depend[args];
    for (int i = 0; i < args; i++) {
        d[i].t = types[i];
        d[i].s = (size_t) values[i * 2 + 0];
        d[i].e = (size_t) values[i * 2 + 1];
        if (types[i] & oh_in) omp_helper_task_depend_check(d + i, &wait_bits);
    }

    //printf("[%s:%d] wait_bits[%x]\n", __FILE__, __LINE__, wait_bits);

    q_wait_bits[depth] = wait_bits;

    if (depends[depth]) delete[] depends[depth];
    depends[depth] = d;
    depends_args[depth] = args;

    depth++;
}

void omp_helper_task_exit() {
    depth--;

    omp_depend* d = depends[depth];

    for (int i = 0; i < depends_args[depth]; i++) {
        if (d[i].t & oh_out) {
            for (int j = 0; j < q_max; j++) {
                if (q_last_used[depth + 1] & (1 << j)) 
                    queues[depth][j].push_back(d + i);
            }
        }
    }

    q_last_used[depth + 1] = 0;
}

void omp_helper_task_exec(int args, int* types, void** values, int* async, int* waits) {
    int q = q_cur;
    int wait_bits = 0;
    omp_depend* d = new omp_depend[args];
    for (int i = 0; i < args; i++) {
        d[i].t = types[i];
        d[i].s = (size_t) values[i * 2 + 0];
        d[i].e = (size_t) values[i * 2 + 1];
        //printf("[%s:%d] R[%d] [%llx:%llx]\n", __FILE__, __LINE__, i, d[i].s, d[i].e);

        if (types[i] & oh_in) omp_helper_task_depend_check(d + i, &wait_bits);
    }

//    printf("[%s:%d] wait_bits[%x] q_cur[%d]\n", __FILE__, __LINE__, wait_bits, q_cur); 

    for (int i = q_cur, j = 0; j < q_max; i++, j++) {
        if (wait_bits & (1 << i)) {
            q = i;
            wait_bits &= ~(1 << i);
            break;
        }
        if (i == q_max - 1) i = -1;
        if (j == q_max - 1) q_cur = q_cur + 1 == q_max ? 0 : q_cur + 1;
    }

    for (int i = 0; i < args; i++) {
        if (types[i] & oh_out) queues[depth][q].push_back(d + i);
    }

    q_last_used[depth] |= (1 << q);

    q += q_off;

    if (async) *async = q;

    if (waits) {
        if (depth > 0) wait_bits |= q_wait_bits[depth - 1];
        for (int i = 0; i < q_max; i++) {
            waits[i] = wait_bits & (1 << i) ? i + q_off : q;
        }
    }
}

