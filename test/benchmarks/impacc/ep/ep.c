//---------------------------------------------------------------------
//      program EMBAR
//---------------------------------------------------------------------
//
//   This is the MPI version of the APP Benchmark 1,
//   the "embarassingly parallel" benchmark.
//
//
//   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
//   numbers.  MK is the Log_2 of the size of each batch of uniform random
//   numbers.  MK can be set for convenience on a given system, since it does
//   not affect the results.

#define DEVS_PER_NODE   4
#define MAX_GANG        32768

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "npbparams.h"
#include "type.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"
#include <sys/time.h>
#include <unistd.h>
//#include <openacc.h>

//#define MK        16
#define MK        10
#define MM        (M - MK)
#define NN        (1LL << MM)
#define NK        (1 << MK)
#define NQ        10
#define EPSILON   1.0e-8
#define A         1220703125.0
#define S         271828183.0

#define t_total   0
#define t_gpairs  1
#define t_randn   2
#define t_rcomm   3
#define t_last    4

/* common/storage/ */
static double x[2*NK];
static double q[NQ];
static double q0, q1, q2, q3, q4, q5, q6, q7, q8, q9;

#pragma openarc impacc ignoreglobal(stderr,__stderrp)

//typedef long long INT_TYPE;
typedef int INT_TYPE;

int main(int argc, char *argv[])
{
  double Mops, t1, t2, t3, t4, x1, x2;
  double sx, sy, tm, an, tt, gc, dum[3];
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  INT_TYPE nn, np, node, no_nodes; 
  INT_TYPE i, ik, kk, l, k, nit, ierrcode, no_large_nodes;
  INT_TYPE _k, _np, _nnp;
  INT_TYPE np_add, k_offset, j;
  logical verified;
  char size[16];
  struct timeval tv0, tv1, tv2;
  double dtv0, dtv1, dtv2;
  char hostname[64];
  int root = 0;
  MPI_Datatype dp_type = MPI_DOUBLE;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &node);
  MPI_Comm_size(MPI_COMM_WORLD, &no_nodes);
  //acc_set_device_num(node % DEVS_PER_NODE, acc_device_xeonphi);

  if (node == root)  {

    // Because the size of the problem is too large to store in a 32-bit
    // integer for some classes, we put it into a string (for printing).
    // Have to strip off the decimal point put in there by the floating
    // point print statement (internal file)
    printf("\n NAS Parallel Benchmarks (NPB3.3-MPI-C) - EP Benchmark\n\n");
    sprintf(size, "%15.0lf", pow(2.0, M+1));
    j = 14;
    if (size[j] == '.') j = j - 1;
    size[j+1] = '\0';
    printf(" Number of random numbers generated: %15s\n", size);
    printf(" Number of active processes:           %13lld\n", no_nodes);
  }

  verified = false;

  // Compute the number of "batches" of random number pairs generated 
  // per processor. Adjust if the number of processors does not evenly 
  // divide the total number

  nn = NN;
  np = nn / no_nodes;
  no_large_nodes = nn % no_nodes;
  if (node < no_large_nodes) {
    np_add = 1;
  } else {
    np_add = 0;
  }
  np = np + np_add;


  if (np == 0) {
    fprintf(stderr, "Too many nodes:%6lld%6d\n", no_nodes, nn);
    ierrcode = 1;
    MPI_Abort(MPI_COMM_WORLD, ierrcode);
    exit(EXIT_FAILURE);
  }

  // Call the random number generator functions and initialize
  // the x-array to reduce the effects of paging on the timings.
  // Also, all mathematical functions that are used. Make
  // sure these initializations cannot be eliminated as dead code.

  vranlc(0, &dum[0], dum[1], &dum[2]);
  dum[0] = randlc(&dum[1], dum[2]);
  for (i = 0; i < 2*NK; i++) {
    x[i] = -1.0e99;
  }
  Mops = log(sqrt(fabs(max(1.0, 1.0))));

  //---------------------------------------------------------------------
  // Synchronize before placing time stamp
  //---------------------------------------------------------------------
  for (i = 0; i < t_last; i++) {
    timer_clear(i);
  }
  timer_start(t_total);

  t1 = A;
  vranlc(0, &t1, A, x);

  // Compute AN = A ^ (2 * NK) (mod 2^46).

  t1 = A;

  for (i = 0; i < MK + 1; i++) {
    t2 = randlc(&t1, t1);
  }

  an = t1;
  tt = S;
  gc = 0.0;
  sx = 0.0;
  sy = 0.0;

  for (i = 0; i < NQ; i++) {
    q[i] = 0.0;
  }
  q0 = q1 = q2 = q3 = q4 = q5 = q6 = q7 = q8 = q9 = 0.0;

  // Each instance of this loop may be performed independently. We compute
  // the k offsets separately to take into account the fact that some nodes
  // have more numbers to generate than others

  if (np_add == 1) {
    k_offset = node * np -1;
  } else {
    k_offset = no_large_nodes*(np+1) + (node-no_large_nodes)*np -1;
  }

  _np = np;
  while (_np > MAX_GANG * 128) {
    _np >>= 1;
  }
  _nnp = np / _np;
  if (np % _np != 0) {
      _nnp++;
  }
  //printf("[%s:%d] nn[%d] no_nodes[%d] np[%d] _np[%d] _nnp[%d]\n", __FILE__, __LINE__, nn, no_nodes, np, _np, _nnp);

  gethostname(hostname, sizeof(hostname));
  gettimeofday(&tv0, NULL);
  dtv0 = tv0.tv_sec + 1.e-6 * tv0.tv_usec;

#pragma acc kernels loop gang worker(16) independent private(x, t1, t2, t3, t4, x1, x2, kk, i, ik, l, k) reduction(+:sx, sy, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9)
  for (_k = 0; _k < _np; _k++)
  for (k = 0; k < _nnp; k++) {
    kk = k_offset + k + 1 + (_k * _nnp);
    if (_k * _nnp + k < np) {
    t1 = S;
    t2 = an;

    // Find starting seed t1 for this kk.

    for (i = 1; i <= 100; i++) {
      ik = kk / 2;
      if (2 * ik != kk) {
          //t3 = randlc(&t1, t2);
          double* _x = &t1;
          double _a = t2;

          const double _r23 = 1.1920928955078125e-07;
          const double _r46 = _r23 * _r23;
          const double _t23 = 8.388608e+06;
          const double _t46 = _t23 * _t23;

          double _t1, _t2, _t3, _t4, _a1, _a2, _x1, _x2, _z;
          double _r;

          _t1 = _r23 * _a;
          _a1 = (int) _t1;
          _a2 = _a - _t23 * _a1;

          _t1 = _r23 * (*_x);
          _x1 = (int) _t1;
          _x2 = *_x - _t23 * _x1;
          _t1 = _a1 * _x2 + _a2 * _x1;
          _t2 = (int) (_r23 * _t1);
          _z = _t1 - _t23 * _t2;
          _t3 = _t23 * _z + _a2 * _x2;
          _t4 = (int) (_r46 * _t3);
          *_x = _t3 - _t46 * _t4;
          _r = _r46 * (*_x);

          t3 = _r;
      }
      if (ik == 0) break;
      {
          //t3 = randlc(&t2, t2);
          double* _x = &t2;
          double _a = t2;

          const double _r23 = 1.1920928955078125e-07;
          const double _r46 = _r23 * _r23;
          const double _t23 = 8.388608e+06;
          const double _t46 = _t23 * _t23;

          double _t1, _t2, _t3, _t4, _a1, _a2, _x1, _x2, _z;
          double _r;

          _t1 = _r23 * _a;
          _a1 = (int) _t1;
          _a2 = _a - _t23 * _a1;

          _t1 = _r23 * (*_x);
          _x1 = (int) _t1;
          _x2 = *_x - _t23 * _x1;
          _t1 = _a1 * _x2 + _a2 * _x1;
          _t2 = (int) (_r23 * _t1);
          _z = _t1 - _t23 * _t2;
          _t3 = _t23 * _z + _a2 * _x2;
          _t4 = (int) (_r46 * _t3);
          *_x = _t3 - _t46 * _t4;
          _r = _r46 * (*_x);

          t3 = _r;
      }
      kk = ik;
    }

    // Compute uniform pseudorandom numbers.
    {
      //vranlc(2 * NK, &t1, A, x);
      INT_TYPE _n = 2 * NK;
      double* _x = &t1;
      double _a = A;
      double *_y = x;

      const double _r23 = 1.1920928955078125e-07;
      const double _r46 = _r23 * _r23;
      const double _t23 = 8.388608e+06;
      const double _t46 = _t23 * _t23;

      double _t1, _t2, _t3, _t4, _a1, _a2, _x1, _x2, _z;

      INT_TYPE _i;

      _t1 = _r23 * _a;
      _a1 = (int) _t1;
      _a2 = _a - _t23 * _a1;

      for ( _i = 0; _i < _n; _i++ ) {
        _t1 = _r23 * (*_x);
        _x1 = (int) _t1;
        _x2 = *_x - _t23 * _x1;
        _t1 = _a1 * _x2 + _a2 * _x1;
        _t2 = (int) (_r23 * _t1);
        _z = _t1 - _t23 * _t2;
        _t3 = _t23 * _z + _a2 * _x2;
        _t4 = (int) (_r46 * _t3) ;
        *_x = _t3 - _t46 * _t4;
        _y[_i] = _r46 * (*_x);
      }
    }

    // Compute Gaussian deviates by acceptance-rejection method and 
    // tally counts in concentric square annuli.  This loop is not 
    // vectorizable. 
    for (i = 0; i < NK; i++) {
      x1 = 2.0 * x[2*i] - 1.0;
      x2 = 2.0 * x[2*i+1] - 1.0;
      t1 = x1 * x1 + x2 * x2;
      if (t1 <= 1.0) {
        t2   = sqrt(-2.0 * log(t1) / t1);
        t3   = (x1 * t2);
        t4   = (x2 * t2);
        l    = max(fabs(t3), fabs(t4));
        //q[l] = q[l] + 1.0;
        if (l == 0) q0 += 1.0;
        else if (l == 1) q1 += 1.0;
        else if (l == 2) q2 += 1.0;
        else if (l == 3) q3 += 1.0;
        else if (l == 4) q4 += 1.0;
        else if (l == 5) q5 += 1.0;
        else if (l == 6) q6 += 1.0;
        else if (l == 7) q7 += 1.0;
        else if (l == 8) q8 += 1.0;
        else q9 += 1.0;
        sx   = sx + t3;
        sy   = sy + t4;
      }
    }
  }

  }

  gettimeofday(&tv1, NULL);
  dtv1 = tv1.tv_sec + 1.e-6 * tv1.tv_usec;

  MPI_Allreduce(&sx, x, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  sx = x[0];
  MPI_Allreduce(&sy, x, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  sy = x[0];

  /*
  MPI_Allreduce(q, x, NQ, dp_type, MPI_SUM, MPI_COMM_WORLD);
  for (i = 0; i < NQ; i++) {
    q[i] = x[i];
  }
  */

  MPI_Allreduce(&q0, q + 0, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q1, q + 1, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q2, q + 2, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q3, q + 3, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q4, q + 4, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q5, q + 5, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q6, q + 6, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q7, q + 7, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q8, q + 8, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&q9, q + 9, 1, dp_type, MPI_SUM, MPI_COMM_WORLD);

  for (i = 0; i < NQ; i++) {
    gc = gc + q[i];
  }

  timer_stop(t_total);
  tm = timer_read(t_total);

  MPI_Allreduce(&tm, x, 1, dp_type, MPI_MAX, MPI_COMM_WORLD);
  tm = x[0];

  gettimeofday(&tv2, NULL);
  dtv2 = tv2.tv_sec + 1.e-6 * tv2.tv_usec;

  if (node == root) {
    nit = 0;
    verified = true;
    if (M == 24) {
      sx_verify_value = -3.247834652034740e+3;
      sy_verify_value = -6.958407078382297e+3;
    } else if (M == 25) {
      sx_verify_value = -2.863319731645753e+3;
      sy_verify_value = -6.320053679109499e+3;
    } else if (M == 28) {
      sx_verify_value = -4.295875165629892e+3;
      sy_verify_value = -1.580732573678431e+4;
    } else if (M == 30) {
      sx_verify_value =  4.033815542441498e+4;
      sy_verify_value = -2.660669192809235e+4;
    } else if (M == 32) {
      sx_verify_value =  4.764367927995374e+4;
      sy_verify_value = -8.084072988043731e+4;
    } else if (M == 36) {
      sx_verify_value =  1.982481200946593e+5;
      sy_verify_value = -1.020596636361769e+5;
    } else if (M == 40) {
      sx_verify_value = -5.319717441530e+05;
      sy_verify_value = -3.688834557731e+05;
    } else if (M == 42) {
      sx_verify_value = -2.267329467773412e+06;
      sy_verify_value = -9.532963461793256e+05;
    } else if (M == 44) {
      sx_verify_value = -1.986768799833953e+01;
      sy_verify_value = -2.164442253974266e+01;
    } else if (M == 46) {
      sx_verify_value = -7.947075305879116e+01;
      sy_verify_value = -8.657769014406949e+01;
    } else {
      verified = false;
    }
    if (verified) {
      sx_err = fabs((sx - sx_verify_value)/sx_verify_value);
      sy_err = fabs((sy - sy_verify_value)/sy_verify_value);
      verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
      printf("[sx_err,sy_err] %.15e, %.15e\n", sx_err, sy_err);
    }
    Mops = pow(2.0, M+1)/tm/1000000.0;

    printf("EP Benchmark Results:\n\n"
        "CPU Time =%10.4lf\n"
        "N = 2^%5d\n"
        "No. Gaussian Pairs =%14.0lf\n"
        "Sums = %25.15lE%25.15lE\n"
        "Counts:\n",
        tm, M, gc, sx, sy);
    for (i = 0; i < NQ; i++) {
      printf("%3lld%14.0lf\n", i, q[i]);
    }

    c_print_results("EP", CLASS, M+1, 0, 0, nit, NPM, 
        no_nodes, tm, Mops, 
        "Random numbers generated", 
        verified, NPBVERSION, COMPILETIME, CS1,
        CS2, CS3, CS4, CS5, CS6, CS7);
  }

  printf("0-2 start[%lf] end [%lf] elapsed[%lf] hostname[%s]\n", dtv0, dtv2, dtv2 - dtv0, hostname);
  printf("0-1 start[%lf] end [%lf] elapsed[%lf] hostname[%s]\n", dtv0, dtv1, dtv1 - dtv0, hostname);
  printf("1-2 start[%lf] end [%lf] elapsed[%lf] hostname[%s]\n", dtv1, dtv2, dtv2 - dtv1, hostname);

  MPI_Finalize();

  return 0;
}

