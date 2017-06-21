#ifndef _OMP_HELPER_EXT_H_
#define _OMP_HELPER_EXT_H_

#define OMP_HELPER_QUEUE_MAX    32
#define OMP_HELPER_QUEUE_OFF    2
#define OMP_HELPER_QUEUE_DEPTH  10

#include <vector>

extern int q_max;
extern int q_off;

extern omp_depend* depends[OMP_HELPER_QUEUE_DEPTH];
extern int depends_args[OMP_HELPER_QUEUE_DEPTH];
extern std::vector<omp_depend*> queues[OMP_HELPER_QUEUE_DEPTH][OMP_HELPER_QUEUE_MAX];
extern int depth;

extern int q_wait_bits[OMP_HELPER_QUEUE_DEPTH];
extern int q_last_used[OMP_HELPER_QUEUE_DEPTH];
extern int q_cur;

#endif /* _OMP_HELPER_EXT_H_ */
