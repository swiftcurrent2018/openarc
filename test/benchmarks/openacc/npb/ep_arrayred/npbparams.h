/* CLASS = S */
/*
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
*/
#define	CLASS	 'S'
#define	M	24
#define	CONVERTDOUBLE	FALSE
#define COMPILETIME "30 Apr 2008"
#define NPBVERSION "2.3"
#define CS1 "cc"
#define CS2 "cc"
#define CS3 "-lm -lgomp"
#define CS4 "-I../common"
#define CS5 "-O3 -fopenmp"
#define CS6 "(none)"
#define CS7 "randdp"
//# of iteratins (NN) = 4096
//Loops with 4096 iterations will be excuted in GPU
//====> # of threads = 256 
//====> # of blocks = 16 
//Org version: allocated memory for x = 134217728
//Unrolled version: allocated memory for x = 134217728
//Unrolling factor = 1
//#define BLOCK_SIZE      256
//#define NBLOCKS 16
//Serial version: allocated memory for x + q = 32808
#ifdef _OPENARC_
#pragma openarc #define NN 4096
#endif
#ifndef _UNROLLFAC_
#define _UNROLLFAC_ 1
#endif
