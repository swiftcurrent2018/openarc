/*
  NAS Parallel Benchmarks 2.3 OpenMP C Versions
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if defined(_OPENMP)
#include <omp.h>
#endif /* _OPENMP */

typedef int boolean;
typedef struct { float real; float imag; } scomplex;

#define TRUE	1
#define FALSE	0

//#define max(a,b) (((a) > (b)) ? (a) : (b))
//#define min(a,b) (((a) < (b)) ? (a) : (b))
#define	pow2(a) ((a)*(a))

#define get_real(c) c.real
#define get_imag(c) c.imag
#define cadd(c_r, c_i,a_r, a_i,b_r, b_i) (c_r = a_r + b_r, c_i = a_i + b_i)
#define csub(c_r, c_i,a_r, a_i,b_r, b_i) (c_r = a_r - b_r, c_i= a_i - b_i)
#define cmul(c_r, c_i ,a_r, a_i, b_r, b_i) (c_r = a_r * b_r - a_i * b_i, \
                     c_i = a_r * b_i + a_i * b_r)
#define crmul(c_r, c_i,a_r, a_i,b) (c_r = a_r * b, c_i = a_i * b)

extern float randlc(float *, float);
extern void vranlc(int, float *, float, float *);
extern void timer_clear(int);
extern void timer_start(int);
extern void timer_stop(int);
extern double timer_read(int);

extern void c_print_results(char *name, char classT, int n1, int n2,
			    int n3, int niter, int nthreads, double t,
			    double mops, char *optype, int passed_verification,
			    char *npbversion, char *compiletime, char *cc,
			    char *clink, char *c_lib, char *c_inc,
			    char *cflags, char *clinkflags, char *rand);
