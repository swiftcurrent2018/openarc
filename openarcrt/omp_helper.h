#ifndef _OMP_HELPER_H_
#define _OMP_HELPER_H_

#include <stdio.h>

#define oh_in       (1 << 0)
#define oh_out      (1 << 1)
#define oh_inout    (oh_in | oh_out)

typedef struct {
    int t;
    size_t s;
    size_t e;
} omp_depend;

#ifdef __cplusplus
extern "C" {
#endif

extern void omp_helper_set_queue_max(int max);

extern void omp_helper_set_queue_off(int off);

extern void omp_helper_task_depend_check(omp_depend* d, int* wait_bits);

extern void omp_helper_task_enter(int args, int* types, void** values);

extern void omp_helper_task_exit();

extern void omp_helper_task_exec(int args, int* types, void** values, int* async, int* waits);

#ifdef __cplusplus
}
#endif

#endif /* _OMP_HELPER_H_ */
