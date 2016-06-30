/*

                 Copyright (c) 2010.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 1.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//#define _OPENACCM
#ifdef _OPENACCM
#include <openacc.h>
#endif

#define LULESH_SHOW_PROGRESS 0
#define LULESH_MEASURE_TIME 1
#define LULESH_STORE_OUTPUT 1
#define	LULESH_PRINT_SIZE	0
#define	LULESH_CHECK_SIZE	0

#define ALLOW_ASPENIFSTMT 0

#ifndef T_EDGEELEM
#define T_EDGEELEM 45
#endif
#define T_EDGENODES (T_EDGEELEM+1)

#if T_EDGEELEM == 10
#pragma openarc #define T_EDGEELEM 10
#elif T_EDGEELEM == 30
#pragma openarc #define T_EDGEELEM 30
#elif T_EDGEELEM == 45
#pragma openarc #define T_EDGEELEM 45
#elif T_EDGEELEM == 60
#pragma openarc #define T_EDGEELEM 60
#endif
#pragma openarc #define T_EDGENODES (T_EDGEELEM+1)

#define T_NUMELEM	(T_EDGEELEM*T_EDGEELEM*T_EDGEELEM)
#pragma openarc #define T_NUMELEM (T_EDGEELEM*T_EDGEELEM*T_EDGEELEM)
#define T_LENGTH	(T_EDGEELEM*T_EDGEELEM*T_EDGEELEM)
#define T_NUMELEM8	(T_NUMELEM*8)
#define T_NUMNODE	(T_EDGENODES*T_EDGENODES*T_EDGENODES)
#pragma openarc #define T_NUMNODE	(T_EDGENODES*T_EDGENODES*T_EDGENODES)
#define T_NUMNODESETS	(T_EDGENODES*T_EDGENODES)
#define T_NODEELEMCORNERLIST	T_NUMELEM8

#if LULESH_MEASURE_TIME
double my_timer()
{
                struct timeval time;

                gettimeofday (&time, 0); 

                return time.tv_sec + time.tv_usec / 1000000.0;
}
#endif

#define std_max(a,b)	(((a) > (b)) ? (a) : (b))

#define Real_t	double
#define Index_t int
#define Int_t	int

enum { VolumeError = -1, QStopError = -2 } ;

/****************************************************/
/* Allow flexibility for arithmetic representations */
/****************************************************/

/* Could also support fixed point and interval arithmetic types */
//typedef float        real4 ;
//typedef double       real8 ;
//typedef long double  real10 ;  /* 10 bytes on x86 */

//typedef int    Index_t ; /* array subscript and loop index */
//typedef real8  Real_t ;  /* floating point representation */
//typedef int    Int_t ;   /* integer representation */

inline float  SQRT4(float  arg) { return sqrtf(arg) ; }
inline double  SQRT8(double  arg) { return sqrt(arg) ; }
inline long double SQRT10(long double arg) { return sqrtl(arg) ; }

inline float  CBRT4(float  arg) { return cbrtf(arg) ; }
inline double  CBRT8(double  arg) { return cbrt(arg) ; }
inline long double CBRT10(long double arg) { return cbrtl(arg) ; }

inline float  FABS4(float  arg) { return fabsf(arg) ; }
inline double  FABS8(double  arg) { return fabs(arg) ; }
inline long double FABS10(long double arg) { return fabsl(arg) ; }


/************************************************************/
/* Allow for flexible data layout experiments by separating */
/* array interface from underlying implementation.          */
/************************************************************/

//struct Domain {

/* This first implementation allows for runnable code */
/* and is not meant to be optimal. Final implementation */
/* should separate declaration and allocation phases */
/* so that allocation can be scheduled in a cache conscious */
/* manner. */

//private:

   /******************/
   /* Implementation */
   /******************/

   /* Node-centered */

   Real_t* m_x ;  /* coordinates */
   Real_t* m_y ;
   Real_t* m_z ;

   Real_t* m_xd ; /* velocities */
   Real_t* m_yd ;
   Real_t* m_zd ;

   Real_t* m_xdd ; /* accelerations */
   Real_t* m_ydd ;
   Real_t* m_zdd ;

   Real_t* m_fx ;  /* forces */
   Real_t* m_fy ;
   Real_t* m_fz ;

   Real_t* m_nodalMass ;  /* mass */

   Index_t* m_symmX ;  /* symmetry plane nodesets */
   Index_t* m_symmY ;
   Index_t* m_symmZ ;

   Index_t* m_nodeElemCount ;
   Index_t* m_nodeElemStart ;
//   Index_t* m_nodeElemList ;
   Index_t* m_nodeElemCornerList ;

   /* Element-centered */

   Index_t*  m_matElemlist ;  /* material indexset */
   Index_t*  m_nodelist ;     /* elemToNode connectivity */

   Index_t*  m_lxim ;  /* element connectivity across each face */
   Index_t*  m_lxip ;
   Index_t*  m_letam ;
   Index_t*  m_letap ;
   Index_t*  m_lzetam ;
   Index_t*  m_lzetap ;

   Int_t*    m_elemBC ;  /* symmetry/free-surface flags for each elem face */

   Real_t* m_dxx ;  /* principal strains -- temporary */
   Real_t* m_dyy ;
   Real_t* m_dzz ;

   Real_t* m_delv_xi ;    /* velocity gradient -- temporary */
   Real_t* m_delv_eta ;
   Real_t* m_delv_zeta ;

   Real_t* m_delx_xi ;    /* coordinate gradient -- temporary */
   Real_t* m_delx_eta ;
   Real_t* m_delx_zeta ;
   
   Real_t* m_e ;   /* energy */

   Real_t* m_p ;   /* pressure */
   Real_t* m_q ;   /* q */
   Real_t* m_ql ;  /* linear term for q */
   Real_t* m_qq ;  /* quadratic term for q */

   Real_t* m_v ;     /* relative volume */
   Real_t* m_volo ;  /* reference volume */
   Real_t* m_vnew ;  /* new relative volume -- temporary */
   Real_t* m_delv ;  /* m_vnew - m_v */
   Real_t* m_vdov ;  /* volume derivative over volume */

   Real_t* m_arealg ;  /* characteristic length of an element */
   
   Real_t* m_ss ;      /* "sound speed" */

   Real_t* m_elemMass ;  /* mass */

   /* Temporary variables from IntegrateStressForElems() and CalcFBHourglassForceForElems()*/
   Real_t *fx_elem;
   Real_t *fy_elem;
   Real_t *fz_elem;
   /* Temporary variables from CalcHourglassControlForElems() */
   Real_t *dvdx;
   Real_t *dvdy;
   Real_t *dvdz;
   Real_t *x8n;
   Real_t *y8n;
   Real_t *z8n;
   /* Temporary variables from EvalEOSForElems() */
   Real_t *e_old;
   Real_t *delvc;
   Real_t *p_old;
   Real_t *q_old;
   Real_t *compression;
   Real_t *compHalfStep;
   Real_t *qq;
   Real_t *ql;
   Real_t *work;
   Real_t *p_new;
   Real_t *e_new;
   Real_t *q_new;
   Real_t *bvc;
   Real_t *pbvc;
   /* Temporary variables from ApplyMaterialPropertiesForElems() */
   Real_t *vnewc;
   /* Temporary variables from CalcVolumeForceForElems() */
   Real_t *sigxx;
   Real_t *sigyy;
   Real_t *sigzz;
   Real_t *determ;

   /* Parameters */

   Real_t  m_dtfixed ;           /* fixed time increment */
   Real_t  m_time ;              /* current time */
   Real_t  m_deltatime ;         /* variable time increment */
   Real_t  m_deltatimemultlb ;
   Real_t  m_deltatimemultub ;
   Real_t  m_stoptime ;          /* end time for simulation */

   Real_t  m_u_cut ;             /* velocity tolerance */
   Real_t  m_hgcoef ;            /* hourglass control */
   Real_t  m_qstop ;             /* excessive q indicator */
   Real_t  m_monoq_max_slope ;
   Real_t  m_monoq_limiter_mult ;
   Real_t  m_e_cut ;             /* energy tolerance */
   Real_t  m_p_cut ;             /* pressure tolerance */
   Real_t  m_ss4o3 ;
   Real_t  m_q_cut ;             /* q tolerance */
   Real_t  m_v_cut ;             /* relative volume tolerance */
   Real_t  m_qlc_monoq ;         /* linear term coef for q */
   Real_t  m_qqc_monoq ;         /* quadratic term coef for q */
   Real_t  m_qqc ;
   Real_t  m_eosvmax ;
   Real_t  m_eosvmin ;
   Real_t  m_pmin ;              /* pressure floor */
   Real_t  m_emin ;              /* energy floor */
   Real_t  m_dvovmax ;           /* maximum allowable volume change */
   Real_t  m_refdens ;           /* reference density */

   Real_t  m_dtcourant ;         /* courant constraint */
   Real_t  m_dthydro ;           /* volume change constraint */
   Real_t  m_dtmax ;             /* maximum allowable time increment */

   Int_t   m_cycle ;             /* iteration count for simulation */

//[DEBUG] Moved from main() for easy model generation.
   Index_t edgeElems ;
   Index_t edgeNodes ;

//#pragma aspen declare param(m_sizeX:edgeElems)
   Index_t   m_sizeX ;           /* X,Y,Z extent of this block */
   Index_t   m_sizeY ;
   Index_t   m_sizeZ ;

//#pragma aspen declare param(m_numElem:T_NUMELEM)
   Index_t   m_numElem ;         /* Elements/Nodes in this domain */
//#pragma aspen declare param(m_numElem8:m_numElem*8)
   Index_t m_numElem8;
//#pragma aspen declare param(m_numNode:T_NUMNODE)
   Index_t   m_numNode ;

#pragma aspen declare param(m_nCorner:m_numElem8)
   Index_t   m_nCorner ;


//public:

   /**************/
   /* Allocation */
   /**************/

   void AllocateNodalPersistent(size_t size)
   {
      m_x = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_y = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_z = (Real_t*)malloc(size*sizeof(Real_t)) ;

      m_xd = (Real_t*)calloc(size, sizeof(Real_t)) ;
      m_yd = (Real_t*)calloc(size, sizeof(Real_t)) ;
      m_zd = (Real_t*)calloc(size, sizeof(Real_t)) ;

      m_xdd = (Real_t*)calloc(size, sizeof(Real_t)) ;
      m_ydd = (Real_t*)calloc(size, sizeof(Real_t)) ;
      m_zdd = (Real_t*)calloc(size, sizeof(Real_t)) ;

      m_fx = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_fy = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_fz = (Real_t*)malloc(size*sizeof(Real_t)) ;

      m_nodalMass = (Real_t*)calloc(size, sizeof(Real_t)) ;
   }

   void AllocateElemPersistent(size_t size)
   {
      Index_t i;
      m_matElemlist = (Index_t*)malloc(size*sizeof(Index_t)) ;
      m_nodelist = (Index_t*)malloc(8*size*sizeof(Index_t)) ;

      m_lxim = (Index_t*)malloc(size*sizeof(Index_t)) ;
      m_lxip = (Index_t*)malloc(size*sizeof(Index_t)) ;
      m_letam = (Index_t*)malloc(size*sizeof(Index_t)) ;
      m_letap = (Index_t*)malloc(size*sizeof(Index_t)) ;
      m_lzetam = (Index_t*)malloc(size*sizeof(Index_t)) ;
      m_lzetap = (Index_t*)malloc(size*sizeof(Index_t)) ;

      m_elemBC = (Int_t*)malloc(size*sizeof(Int_t)) ;

      m_e = (Real_t*)calloc(size, sizeof(Real_t)) ;

      m_p = (Real_t*)calloc(size, sizeof(Real_t)) ;
      m_q = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_ql = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_qq = (Real_t*)malloc(size*sizeof(Real_t)) ;

      m_v = (Real_t*)malloc(size*sizeof(Real_t)) ;
      for (i=0;i<size;++i) {
         m_v[i] = 1.0;
      }
      m_volo = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_delv = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_vdov = (Real_t*)malloc(size*sizeof(Real_t)) ;

      m_arealg = (Real_t*)malloc(size*sizeof(Real_t)) ;
   
      m_ss = (Real_t*)malloc(size*sizeof(Real_t)) ;

      m_elemMass = (Real_t*)malloc(size*sizeof(Real_t)) ;
   }

   /* Temporaries should not be initialized in bulk but */
   /* this is a runnable placeholder for now */
   void AllocateElemTemporary(size_t size)
   {
      m_dxx = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_dyy = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_dzz = (Real_t*)malloc(size*sizeof(Real_t)) ;

      m_delv_xi = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_delv_eta = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_delv_zeta = (Real_t*)malloc(size*sizeof(Real_t)) ;

      m_delx_xi = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_delx_eta = (Real_t*)malloc(size*sizeof(Real_t)) ;
      m_delx_zeta = (Real_t*)malloc(size*sizeof(Real_t)) ;

      m_vnew = (Real_t*)malloc(size*sizeof(Real_t)) ;
   }

   void AllocateNodesets(size_t size)
   {
#pragma aspen declare param(size:(edgeElems+1)*(edgeElems+1))
      m_symmX = (Index_t*)malloc(size*sizeof(Index_t)) ;
      m_symmY = (Index_t*)malloc(size*sizeof(Index_t)) ;
      m_symmZ = (Index_t*)malloc(size*sizeof(Index_t)) ;
   }

   void AllocateNodeElemIndexes()
   {
       Index_t m, i, j;
       Index_t clSize;

#if LULESH_PRINT_SIZE
       printf("T_NUMNODE\t%d\n", m_numNode);
#endif
#if LULESH_CHECK_SIZE
       if (m_numNode != T_NUMNODE) {
          printf("T_NUMNODE should be %d\n", m_numNode);
          exit(1);
       }
#endif
       /* set up node-centered indexing of elements */
       m_nodeElemCount = (Index_t*)malloc(m_numNode*sizeof(Index_t)) ;

       for (i=0;i<m_numNode;++i) {
          m_nodeElemCount[i]=0;
       }

       for (i=0; i<m_numElem; ++i) {
          Index_t *nl = &m_nodelist[8*i] ;
          for (j=0; j < 8; ++j) {
             ++m_nodeElemCount[nl[j]];
          }
       }

       m_nodeElemStart = (Index_t*)malloc(m_numNode*sizeof(Index_t)) ;

       m_nodeElemStart[0]=0;

       for (i=1; i < m_numNode; ++i) {
          m_nodeElemStart[i] = m_nodeElemStart[i-1] + m_nodeElemCount[i-1] ;
       }

       m_nCorner = m_nodeElemStart[m_numNode-1] + m_nodeElemCount[m_numNode-1];

//       m_nodeElemList.resize(nodeElemStart(m_numNode-1) +
//                             nodeElemCount(m_numNode-1));

#if LULESH_PRINT_SIZE
       printf("T_NODEELEMCORNERLIST\t%d\n", (m_nodeElemStart[m_numNode-1]+m_nodeElemCount[m_numNode-1]));
#endif
#if LULESH_CHECK_SIZE
       if ((m_nodeElemStart[m_numNode-1]+m_nodeElemCount[m_numNode-1])!= T_NODEELEMCORNERLIST) {
          printf("T_NODEELEMCORNERLIST should be %d\n", (m_nodeElemStart[m_numNode-1]+m_nodeElemCount[m_numNode-1]));
          exit(1);
       }
#endif
#pragma aspen declare data(m_nodeElemCornerList:traits(Array(m_nCorner,aspen_param_int)))
       m_nodeElemCornerList = (Index_t*)malloc((m_nodeElemStart[m_numNode-1]+m_nodeElemCount[m_numNode-1])*sizeof(Index_t)) ;

       for (i=0; i < m_numNode; ++i) {
          m_nodeElemCount[i]=0;
       }

       for (i=0; i < m_numElem; ++i) {
          Index_t *nl = &m_nodelist[8*i] ;
          for (j=0; j < 8; ++j) {
             Index_t m = nl[j];
             Index_t k = i*8 + j ;
             Index_t offset = m_nodeElemStart[m]+m_nodeElemCount[m] ;
//             nodeElemList(offset) = i;
             m_nodeElemCornerList[offset] = k;
             ++m_nodeElemCount[m];
          }
       }

       clSize = (m_nodeElemStart[m_numNode-1]+m_nodeElemCount[m_numNode-1]);
#pragma aspen control ignore
       for (i=0; i < clSize; ++i) {
          Index_t clv = m_nodeElemCornerList[i] ;
          if ((clv < 0) || (clv > m_numElem*8)) {
               fprintf(stderr,
        "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
               exit(1);
          }
      }
   }

   
//} domain ;


Real_t *Allocate(size_t size)
{
   return (Real_t *)(malloc(sizeof(Real_t)*size)) ;
}

void Release(Real_t **ptr)
{
#pragma aspen control probability(1)
   if (*ptr != NULL) {
      free(*ptr) ;
      *ptr = NULL ;
   }
}

void AllocateTemporary(size_t numElem8) {
#if LULESH_PRINT_SIZE
       printf("T_NUMELEM8\t%d\n", numElem8);
#endif
#if LULESH_CHECK_SIZE
       if (numElem8 != T_NUMELEM8) {
          printf("T_NUMELEM8 should be %d\n", numElem8);
          exit(1);
       }
#endif
   fx_elem = Allocate(numElem8) ;
   fy_elem = Allocate(numElem8) ;
   fz_elem = Allocate(numElem8) ;
	dvdx = Allocate(numElem8) ;
	dvdy = Allocate(numElem8) ;
	dvdz = Allocate(numElem8) ;
	x8n  = Allocate(numElem8) ;
	y8n  = Allocate(numElem8) ;
	z8n  = Allocate(numElem8) ;
}

void AllocateTemporary2(size_t length) {
   e_old = Allocate(length) ;
   delvc = Allocate(length) ;
   p_old = Allocate(length) ;
   q_old = Allocate(length) ;
   compression = Allocate(length) ;
   compHalfStep = Allocate(length) ;
   qq = Allocate(length) ;
   ql = Allocate(length) ;
   work = Allocate(length) ;
   p_new = Allocate(length) ;
   e_new = Allocate(length) ;
   q_new = Allocate(length) ;
   bvc = Allocate(length) ;
   pbvc = Allocate(length) ;
   vnewc = Allocate(length) ;
   sigxx  = Allocate(length) ;
   sigyy  = Allocate(length) ;
   sigzz  = Allocate(length) ;
   determ = Allocate(length) ;
}


/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
#define XI_M        0x003
#define XI_M_SYMM   0x001
#define XI_M_FREE   0x002

#define XI_P        0x00c
#define XI_P_SYMM   0x004
#define XI_P_FREE   0x008

#define ETA_M       0x030
#define ETA_M_SYMM  0x010
#define ETA_M_FREE  0x020

#define ETA_P       0x0c0
#define ETA_P_SYMM  0x040
#define ETA_P_FREE  0x080

#define ZETA_M      0x300
#define ZETA_M_SYMM 0x100
#define ZETA_M_FREE 0x200

#define ZETA_P      0xc00
#define ZETA_P_SYMM 0x400
#define ZETA_P_FREE 0x800


static inline
void TimeIncrement()
{
   Real_t targetdt = m_stoptime - m_time;

//[DEBUG] Current ASPEN does not accept "<=" operator.
#pragma aspen control probability(1)
   if ((m_dtfixed <= 0.0) && (m_cycle != 0)) {
      Real_t ratio ;
      Real_t olddt = m_deltatime;

      /* This will require a reduction in parallel */
      Real_t newdt = 1.0e+20 ;
#pragma aspen control probability(1)
      if (m_dtcourant < newdt) {
         newdt = m_dtcourant / 2.0 ;
      }
#pragma aspen control probability(1)
      if (m_dthydro < newdt) {
         newdt = m_dthydro * 2.0 / 3.0 ;
      }

      ratio = newdt / olddt ;
#pragma aspen control execute flops(3:traits(dp))
      if (ratio >= 1.0) {
         if (ratio < m_deltatimemultlb) {
            newdt = olddt ;
         }
         else if (ratio > m_deltatimemultub) {
            newdt = olddt*m_deltatimemultub;
         }
      }

      if (newdt > m_dtmax) {
         newdt = m_dtmax;
      }
      m_deltatime = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
#pragma aspen control execute flops(6:traits(dp))
   if ((targetdt > m_deltatime) &&
       (targetdt < (4.0 * m_deltatime / 3.0)) ) {
      targetdt = 2.0 * m_deltatime / 3.0 ;
   }

#pragma aspen control ignore
   if (targetdt < m_deltatime) {
      m_deltatime = targetdt ;
   }

   m_time += m_deltatime;

   ++m_cycle;
}

static inline
void InitStressTermsForElems(Index_t numElem, 
                             Real_t sigxx[T_NUMELEM], Real_t sigyy[T_NUMELEM], 
                             Real_t sigzz[T_NUMELEM], Real_t p_p[T_NUMELEM],
                             Real_t p_q[T_NUMELEM])
{
   //
   // pull in the stresses appropriate to the hydro integration
   //
   Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(p_p, p_q, sigxx, sigyy, sigzz)
#else
#pragma omp parallel for private(i) firstprivate(numElem)
#endif
   for (i = 0 ; i < numElem ; ++i){
      sigxx[i] =  sigyy[i] = sigzz[i] =  - p_p[i] - p_q[i];
   }
}

static inline
void CalcElemShapeFunctionDerivatives( const Real_t* const x,
                                       const Real_t* const y,
                                       const Real_t* const z,
                                       Real_t b[][8],
                                       Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = .125 * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = .125 * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = .125 * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = .125 * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = .125 * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = .125 * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = .125 * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = .125 * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = .125 * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = 8. * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

static inline
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = 0.5 * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = 0.5 * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = 0.5 * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = 0.5 * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = 0.5 * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = 0.5 * (z2 + z1 - z3 - z0);
   Real_t areaX = 0.25 * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = 0.25 * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = 0.25 * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

static inline
void CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8])
{
   Index_t i;
   for (i = 0 ; i < 8 ; ++i) {
      pfx[i] = 0.0;
      pfy[i] = 0.0;
      pfz[i] = 0.0;
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}

static inline
void SumElemStressesToNodeForces( const Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  Real_t* const fx,
                                  Real_t* const fy,
                                  Real_t* const fz )
{
  Real_t pfx0 = B[0][0] ;   Real_t pfx1 = B[0][1] ;
  Real_t pfx2 = B[0][2] ;   Real_t pfx3 = B[0][3] ;
  Real_t pfx4 = B[0][4] ;   Real_t pfx5 = B[0][5] ;
  Real_t pfx6 = B[0][6] ;   Real_t pfx7 = B[0][7] ;

  Real_t pfy0 = B[1][0] ;   Real_t pfy1 = B[1][1] ;
  Real_t pfy2 = B[1][2] ;   Real_t pfy3 = B[1][3] ;
  Real_t pfy4 = B[1][4] ;   Real_t pfy5 = B[1][5] ;
  Real_t pfy6 = B[1][6] ;   Real_t pfy7 = B[1][7] ;

  Real_t pfz0 = B[2][0] ;   Real_t pfz1 = B[2][1] ;
  Real_t pfz2 = B[2][2] ;   Real_t pfz3 = B[2][3] ;
  Real_t pfz4 = B[2][4] ;   Real_t pfz5 = B[2][5] ;
  Real_t pfz6 = B[2][6] ;   Real_t pfz7 = B[2][7] ;

  fx[0] = -( stress_xx * pfx0 );
  fx[1] = -( stress_xx * pfx1 );
  fx[2] = -( stress_xx * pfx2 );
  fx[3] = -( stress_xx * pfx3 );
  fx[4] = -( stress_xx * pfx4 );
  fx[5] = -( stress_xx * pfx5 );
  fx[6] = -( stress_xx * pfx6 );
  fx[7] = -( stress_xx * pfx7 );

  fy[0] = -( stress_yy * pfy0  );
  fy[1] = -( stress_yy * pfy1  );
  fy[2] = -( stress_yy * pfy2  );
  fy[3] = -( stress_yy * pfy3  );
  fy[4] = -( stress_yy * pfy4  );
  fy[5] = -( stress_yy * pfy5  );
  fy[6] = -( stress_yy * pfy6  );
  fy[7] = -( stress_yy * pfy7  );

  fz[0] = -( stress_zz * pfz0 );
  fz[1] = -( stress_zz * pfz1 );
  fz[2] = -( stress_zz * pfz2 );
  fz[3] = -( stress_zz * pfz3 );
  fz[4] = -( stress_zz * pfz4 );
  fz[5] = -( stress_zz * pfz5 );
  fz[6] = -( stress_zz * pfz6 );
  fz[7] = -( stress_zz * pfz7 );
}

static inline
void IntegrateStressForElems( Index_t numElem,
                              Real_t sigxx[T_NUMELEM], Real_t sigyy[T_NUMELEM], 
                              Real_t sigzz[T_NUMELEM], Real_t determ[T_NUMELEM],
                              Index_t p_nodelist[T_NUMELEM8], Real_t p_x[T_NUMNODE],
                              Real_t p_y[T_NUMNODE], Real_t p_z[T_NUMNODE],
                              Index_t p_nodeElemCount[T_NUMNODE], Index_t p_nodeElemStart[T_NUMNODE],
                              Index_t p_nodeElemCornerList[T_NODEELEMCORNERLIST],
                              Real_t p_fx[T_NUMNODE], Real_t p_fy[T_NUMNODE],
                              Real_t p_fz[T_NUMNODE])
{
   Index_t k, lnode, gnode, i;
/*
   Real_t *fx_elem = Allocate(numElem8) ;
   Real_t *fy_elem = Allocate(numElem8) ;
   Real_t *fz_elem = Allocate(numElem8) ;
*/

  // loop over all elements
#ifdef _OPENACC
#pragma acc parallel loop independent present(p_x, p_y, p_z, determ, p_nodelist, sigxx,\
sigyy, sigzz, fx_elem, fy_elem, fz_elem)
#else
#pragma omp parallel for private(k) firstprivate(numElem)
#endif
  for( k=0 ; k<numElem ; ++k )
  {
    Real_t B[3][8] ;// shape function derivatives
    Real_t x_local[8] ;
    Real_t y_local[8] ;
    Real_t z_local[8] ;

#pragma aspen declare data(elemNodes:traits(Array(8,aspen_param_int)))
    const Index_t* const elemNodes = &p_nodelist[8*k];

    // get nodal coordinates from global arrays and copy into local arrays.
    for( lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemNodes[lnode];
      x_local[lnode] = p_x[gnode];
      y_local[lnode] = p_y[gnode];
      z_local[lnode] = p_z[gnode];
    }

    /* Volume calculation involves extra work for numerical consistency. */
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                         B, &determ[k]);

    CalcElemNodeNormals( B[0] , B[1], B[2],
                          x_local, y_local, z_local );

    SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                 &fx_elem[k*8], &fy_elem[k*8], &fz_elem[k*8] ) ;

#if 0
    // copy nodal force contributions to global force arrray.
    for( lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemNodes[lnode];
      p_fx(gnode) += fx_local[lnode];
      p_fy(gnode) += fy_local[lnode];
      p_fz(gnode) += fz_local[lnode];
    }
#endif
  }

  {
     Index_t numNode = m_numNode;

#ifdef _OPENACC
#pragma acc kernels loop independent present(fx_elem, fy_elem, fz_elem, \
p_fx, p_fy, p_fz, p_nodeElemCount, p_nodeElemCornerList, \
p_nodeElemStart)
#else
#pragma omp parallel for private(gnode) firstprivate(numNode)
#endif
     for( gnode=0 ; gnode<numNode ; ++gnode )
     {
        Index_t count = p_nodeElemCount[gnode];
        Index_t start = p_nodeElemStart[gnode];
        Real_t fx = 0.0 ;
        Real_t fy = 0.0 ;
        Real_t fz = 0.0 ;
#pragma aspen declare param(aspen_param_elemCount:1)
#pragma aspen control loop(aspen_param_elemCount)
        for (i=0 ; i < count ; ++i) {
           Index_t elem = p_nodeElemCornerList[start+i];
           fx += fx_elem[elem] ;
           fy += fy_elem[elem] ;
           fz += fz_elem[elem] ;
        }
        p_fx[gnode] = fx ;
        p_fy[gnode] = fy ;
        p_fz[gnode] = fz ;
     }
  }

/*
  Release(&fz_elem) ;
  Release(&fy_elem) ;
  Release(&fx_elem) ;
*/
}


static inline
void CollectDomainNodesToElemNodes(const Index_t* elemToNode,
                                   Real_t elemX[8],
                                   Real_t elemY[8],
                                   Real_t elemZ[8])
{
   Index_t nd0i = elemToNode[0] ;
   Index_t nd1i = elemToNode[1] ;
   Index_t nd2i = elemToNode[2] ;
   Index_t nd3i = elemToNode[3] ;
   Index_t nd4i = elemToNode[4] ;
   Index_t nd5i = elemToNode[5] ;
   Index_t nd6i = elemToNode[6] ;
   Index_t nd7i = elemToNode[7] ;

   elemX[0] = m_x[nd0i];
   elemX[1] = m_x[nd1i];
   elemX[2] = m_x[nd2i];
   elemX[3] = m_x[nd3i];
   elemX[4] = m_x[nd4i];
   elemX[5] = m_x[nd5i];
   elemX[6] = m_x[nd6i];
   elemX[7] = m_x[nd7i];

   elemY[0] = m_y[nd0i];
   elemY[1] = m_y[nd1i];
   elemY[2] = m_y[nd2i];
   elemY[3] = m_y[nd3i];
   elemY[4] = m_y[nd4i];
   elemY[5] = m_y[nd5i];
   elemY[6] = m_y[nd6i];
   elemY[7] = m_y[nd7i];

   elemZ[0] = m_z[nd0i];
   elemZ[1] = m_z[nd1i];
   elemZ[2] = m_z[nd2i];
   elemZ[3] = m_z[nd3i];
   elemZ[4] = m_z[nd4i];
   elemZ[5] = m_z[nd5i];
   elemZ[6] = m_z[nd6i];
   elemZ[7] = m_z[nd7i];

}

static inline
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = 1.0 / 12.0 ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

static inline
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

static inline
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t *hourgam0,
                              Real_t *hourgam1, Real_t *hourgam2, Real_t *hourgam3,
                              Real_t *hourgam4, Real_t *hourgam5, Real_t *hourgam6,
                              Real_t *hourgam7, Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Index_t i00=0;
   Index_t i01=1;
   Index_t i02=2;
   Index_t i03=3;

   Real_t h00 =
      hourgam0[i00] * xd[0] + hourgam1[i00] * xd[1] +
      hourgam2[i00] * xd[2] + hourgam3[i00] * xd[3] +
      hourgam4[i00] * xd[4] + hourgam5[i00] * xd[5] +
      hourgam6[i00] * xd[6] + hourgam7[i00] * xd[7];

   Real_t h01 =
      hourgam0[i01] * xd[0] + hourgam1[i01] * xd[1] +
      hourgam2[i01] * xd[2] + hourgam3[i01] * xd[3] +
      hourgam4[i01] * xd[4] + hourgam5[i01] * xd[5] +
      hourgam6[i01] * xd[6] + hourgam7[i01] * xd[7];

   Real_t h02 =
      hourgam0[i02] * xd[0] + hourgam1[i02] * xd[1]+
      hourgam2[i02] * xd[2] + hourgam3[i02] * xd[3]+
      hourgam4[i02] * xd[4] + hourgam5[i02] * xd[5]+
      hourgam6[i02] * xd[6] + hourgam7[i02] * xd[7];

   Real_t h03 =
      hourgam0[i03] * xd[0] + hourgam1[i03] * xd[1] +
      hourgam2[i03] * xd[2] + hourgam3[i03] * xd[3] +
      hourgam4[i03] * xd[4] + hourgam5[i03] * xd[5] +
      hourgam6[i03] * xd[6] + hourgam7[i03] * xd[7];

   hgfx[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfx[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfx[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfx[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfx[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfx[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfx[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfx[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * yd[0] + hourgam1[i00] * yd[1] +
      hourgam2[i00] * yd[2] + hourgam3[i00] * yd[3] +
      hourgam4[i00] * yd[4] + hourgam5[i00] * yd[5] +
      hourgam6[i00] * yd[6] + hourgam7[i00] * yd[7];

   h01 =
      hourgam0[i01] * yd[0] + hourgam1[i01] * yd[1] +
      hourgam2[i01] * yd[2] + hourgam3[i01] * yd[3] +
      hourgam4[i01] * yd[4] + hourgam5[i01] * yd[5] +
      hourgam6[i01] * yd[6] + hourgam7[i01] * yd[7];

   h02 =
      hourgam0[i02] * yd[0] + hourgam1[i02] * yd[1]+
      hourgam2[i02] * yd[2] + hourgam3[i02] * yd[3]+
      hourgam4[i02] * yd[4] + hourgam5[i02] * yd[5]+
      hourgam6[i02] * yd[6] + hourgam7[i02] * yd[7];

   h03 =
      hourgam0[i03] * yd[0] + hourgam1[i03] * yd[1] +
      hourgam2[i03] * yd[2] + hourgam3[i03] * yd[3] +
      hourgam4[i03] * yd[4] + hourgam5[i03] * yd[5] +
      hourgam6[i03] * yd[6] + hourgam7[i03] * yd[7];


   hgfy[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfy[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfy[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfy[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfy[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfy[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfy[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfy[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * zd[0] + hourgam1[i00] * zd[1] +
      hourgam2[i00] * zd[2] + hourgam3[i00] * zd[3] +
      hourgam4[i00] * zd[4] + hourgam5[i00] * zd[5] +
      hourgam6[i00] * zd[6] + hourgam7[i00] * zd[7];

   h01 =
      hourgam0[i01] * zd[0] + hourgam1[i01] * zd[1] +
      hourgam2[i01] * zd[2] + hourgam3[i01] * zd[3] +
      hourgam4[i01] * zd[4] + hourgam5[i01] * zd[5] +
      hourgam6[i01] * zd[6] + hourgam7[i01] * zd[7];

   h02 =
      hourgam0[i02] * zd[0] + hourgam1[i02] * zd[1]+
      hourgam2[i02] * zd[2] + hourgam3[i02] * zd[3]+
      hourgam4[i02] * zd[4] + hourgam5[i02] * zd[5]+
      hourgam6[i02] * zd[6] + hourgam7[i02] * zd[7];

   h03 =
      hourgam0[i03] * zd[0] + hourgam1[i03] * zd[1] +
      hourgam2[i03] * zd[2] + hourgam3[i03] * zd[3] +
      hourgam4[i03] * zd[4] + hourgam5[i03] * zd[5] +
      hourgam6[i03] * zd[6] + hourgam7[i03] * zd[7];


   hgfz[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfz[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfz[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfz[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfz[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfz[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfz[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfz[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);
}

static inline
void CalcFBHourglassForceForElems(Real_t determ[T_NUMELEM],
            Real_t x8n[T_NUMELEM8], Real_t y8n[T_NUMELEM8], Real_t z8n[T_NUMELEM8],
            Real_t dvdx[T_NUMELEM8], Real_t dvdy[T_NUMELEM8], Real_t dvdz[T_NUMELEM8],
            Real_t hourg, Real_t p_ss[T_NUMELEM], Index_t p_nodelist[T_NUMELEM8],
            Real_t p_elemMass[T_NUMELEM], Real_t p_xd[T_NUMNODE], Real_t p_yd[T_NUMNODE],
            Real_t p_zd[T_NUMNODE], Index_t p_nodeElemCount[T_NUMNODE], 
            Index_t p_nodeElemStart[T_NUMNODE], 
            Index_t p_nodeElemCornerList[T_NODEELEMCORNERLIST],
            Real_t p_fx[T_NUMNODE], Real_t p_fy[T_NUMNODE],
            Real_t p_fz[T_NUMNODE])

{
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/

   Index_t i1, i2, gnode, i;
   Index_t numElem = m_numElem;
#pragma aspen declare param(numElem8:m_numElem*8)
   Index_t numElem8 = numElem * 8 ;
#if LULESH_PRINT_SIZE
       printf("T_NUMELEM8\t%d\n", numElem8);
#endif
#if LULESH_CHECK_SIZE
       if (numElem8 != T_NUMELEM8) {
          printf("T_NUMELEM8 should be %d\n", numElem8);
          exit(1);
       }
#endif
/*
   Real_t *fx_elem = Allocate(numElem8) ;
   Real_t *fy_elem = Allocate(numElem8) ;
   Real_t *fz_elem = Allocate(numElem8) ;
*/

   Real_t  gamma[4][8];

   gamma[0][0] =  1.;
   gamma[0][1] =  1.;
   gamma[0][2] = -1.;
   gamma[0][3] = -1.;
   gamma[0][4] = -1.;
   gamma[0][5] = -1.;
   gamma[0][6] =  1.;
   gamma[0][7] =  1.;
   gamma[1][0] =  1.;
   gamma[1][1] = -1.;
   gamma[1][2] = -1.;
   gamma[1][3] =  1.;
   gamma[1][4] = -1.;
   gamma[1][5] =  1.;
   gamma[1][6] =  1.;
   gamma[1][7] = -1.;
   gamma[2][0] =  1.;
   gamma[2][1] = -1.;
   gamma[2][2] =  1.;
   gamma[2][3] = -1.;
   gamma[2][4] =  1.;
   gamma[2][5] = -1.;
   gamma[2][6] =  1.;
   gamma[2][7] = -1.;
   gamma[3][0] = -1.;
   gamma[3][1] =  1.;
   gamma[3][2] = -1.;
   gamma[3][3] =  1.;
   gamma[3][4] =  1.;
   gamma[3][5] = -1.;
   gamma[3][6] =  1.;
   gamma[3][7] = -1.;

/*************************************************/
/*    compute the hourglass modes */

#ifdef _OPENACC
#pragma acc kernels loop independent copyin(gamma[0:4][0:8]) present(fx_elem, fy_elem, fz_elem, \
p_xd, p_yd, p_zd, dvdx, dvdy, dvdz, x8n, y8n, z8n, p_nodelist, determ, \
p_ss, p_elemMass)
#else
#pragma omp parallel for private(i2, i1) firstprivate(numElem, hourg) 
#endif
   for(i2=0; i2<numElem; ++i2){
      Real_t *fx_local, *fy_local, *fz_local ;
      Real_t hgfx[8], hgfy[8], hgfz[8] ;

      Real_t coefficient;

      Real_t hourgam0[4], hourgam1[4], hourgam2[4], hourgam3[4] ;
      Real_t hourgam4[4], hourgam5[4], hourgam6[4], hourgam7[4];
      Real_t xd1[8], yd1[8], zd1[8] ;

#pragma aspen declare data(elemToNode:traits(Array(8,aspen_param_int)))
      const Index_t *elemToNode = &p_nodelist[8*i2];
      Index_t i3=8*i2;
      Real_t volinv=1.0/determ[i2];
      Real_t ss1, mass1, volume13 ;
      Index_t n0si2;
      Index_t n1si2;
      Index_t n2si2;
      Index_t n3si2;
      Index_t n4si2;
      Index_t n5si2;
      Index_t n6si2;
      Index_t n7si2;
      for(i1=0;i1<4;++i1){

         Real_t hourmodx =
            x8n[i3] * gamma[i1][0] + x8n[i3+1] * gamma[i1][1] +
            x8n[i3+2] * gamma[i1][2] + x8n[i3+3] * gamma[i1][3] +
            x8n[i3+4] * gamma[i1][4] + x8n[i3+5] * gamma[i1][5] +
            x8n[i3+6] * gamma[i1][6] + x8n[i3+7] * gamma[i1][7];

         Real_t hourmody =
            y8n[i3] * gamma[i1][0] + y8n[i3+1] * gamma[i1][1] +
            y8n[i3+2] * gamma[i1][2] + y8n[i3+3] * gamma[i1][3] +
            y8n[i3+4] * gamma[i1][4] + y8n[i3+5] * gamma[i1][5] +
            y8n[i3+6] * gamma[i1][6] + y8n[i3+7] * gamma[i1][7];

         Real_t hourmodz =
            z8n[i3] * gamma[i1][0] + z8n[i3+1] * gamma[i1][1] +
            z8n[i3+2] * gamma[i1][2] + z8n[i3+3] * gamma[i1][3] +
            z8n[i3+4] * gamma[i1][4] + z8n[i3+5] * gamma[i1][5] +
            z8n[i3+6] * gamma[i1][6] + z8n[i3+7] * gamma[i1][7];

         hourgam0[i1] = gamma[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
                                                  dvdy[i3  ] * hourmody +
                                                  dvdz[i3  ] * hourmodz );

         hourgam1[i1] = gamma[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
                                                  dvdy[i3+1] * hourmody +
                                                  dvdz[i3+1] * hourmodz );

         hourgam2[i1] = gamma[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
                                                  dvdy[i3+2] * hourmody +
                                                  dvdz[i3+2] * hourmodz );

         hourgam3[i1] = gamma[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
                                                  dvdy[i3+3] * hourmody +
                                                  dvdz[i3+3] * hourmodz );

         hourgam4[i1] = gamma[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
                                                  dvdy[i3+4] * hourmody +
                                                  dvdz[i3+4] * hourmodz );

         hourgam5[i1] = gamma[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
                                                  dvdy[i3+5] * hourmody +
                                                  dvdz[i3+5] * hourmodz );

         hourgam6[i1] = gamma[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
                                                  dvdy[i3+6] * hourmody +
                                                  dvdz[i3+6] * hourmodz );

         hourgam7[i1] = gamma[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
                                                  dvdy[i3+7] * hourmody +
                                                  dvdz[i3+7] * hourmodz );

      }

      /* compute forces */
      /* store forces into h arrays (force arrays) */

      ss1=p_ss[i2];
      mass1=p_elemMass[i2];
      volume13=CBRT8(determ[i2]);

      n0si2 = elemToNode[0];
      n1si2 = elemToNode[1];
      n2si2 = elemToNode[2];
      n3si2 = elemToNode[3];
      n4si2 = elemToNode[4];
      n5si2 = elemToNode[5];
      n6si2 = elemToNode[6];
      n7si2 = elemToNode[7];

      xd1[0] = p_xd[n0si2];
      xd1[1] = p_xd[n1si2];
      xd1[2] = p_xd[n2si2];
      xd1[3] = p_xd[n3si2];
      xd1[4] = p_xd[n4si2];
      xd1[5] = p_xd[n5si2];
      xd1[6] = p_xd[n6si2];
      xd1[7] = p_xd[n7si2];

      yd1[0] = p_yd[n0si2];
      yd1[1] = p_yd[n1si2];
      yd1[2] = p_yd[n2si2];
      yd1[3] = p_yd[n3si2];
      yd1[4] = p_yd[n4si2];
      yd1[5] = p_yd[n5si2];
      yd1[6] = p_yd[n6si2];
      yd1[7] = p_yd[n7si2];

      zd1[0] = p_zd[n0si2];
      zd1[1] = p_zd[n1si2];
      zd1[2] = p_zd[n2si2];
      zd1[3] = p_zd[n3si2];
      zd1[4] = p_zd[n4si2];
      zd1[5] = p_zd[n5si2];
      zd1[6] = p_zd[n6si2];
      zd1[7] = p_zd[n7si2];

      coefficient = - hourg * 0.01 * ss1 * mass1 / volume13;

      CalcElemFBHourglassForce(xd1,yd1,zd1,
                      hourgam0,hourgam1,hourgam2,hourgam3,
                      hourgam4,hourgam5,hourgam6,hourgam7,
                      coefficient, hgfx, hgfy, hgfz);

#pragma aspen declare data(fx_local:traits(Array(8, aspen_param_double)))
      fx_local = &fx_elem[i3] ;
      fx_local[0] = hgfx[0];
      fx_local[1] = hgfx[1];
      fx_local[2] = hgfx[2];
      fx_local[3] = hgfx[3];
      fx_local[4] = hgfx[4];
      fx_local[5] = hgfx[5];
      fx_local[6] = hgfx[6];
      fx_local[7] = hgfx[7];

#pragma aspen declare data(fy_local:traits(Array(8, aspen_param_double)))
      fy_local = &fy_elem[i3] ;
      fy_local[0] = hgfy[0];
      fy_local[1] = hgfy[1];
      fy_local[2] = hgfy[2];
      fy_local[3] = hgfy[3];
      fy_local[4] = hgfy[4];
      fy_local[5] = hgfy[5];
      fy_local[6] = hgfy[6];
      fy_local[7] = hgfy[7];

#pragma aspen declare data(fz_local:traits(Array(8, aspen_param_double)))
      fz_local = &fz_elem[i3] ;
      fz_local[0] = hgfz[0];
      fz_local[1] = hgfz[1];
      fz_local[2] = hgfz[2];
      fz_local[3] = hgfz[3];
      fz_local[4] = hgfz[4];
      fz_local[5] = hgfz[5];
      fz_local[6] = hgfz[6];
      fz_local[7] = hgfz[7];

#if 0
      p_fx[n0si2] += hgfx[0];
      p_fy[n0si2] += hgfy[0];
      p_fz[n0si2] += hgfz[0];

      p_fx[n1si2] += hgfx[1];
      p_fy[n1si2] += hgfy[1];
      p_fz[n1si2] += hgfz[1];

      p_fx[n2si2] += hgfx[2];
      p_fy[n2si2] += hgfy[2];
      p_fz[n2si2] += hgfz[2];

      p_fx[n3si2] += hgfx[3];
      p_fy[n3si2] += hgfy[3];
      p_fz[n3si2] += hgfz[3];

      p_fx[n4si2] += hgfx[4];
      p_fy[n4si2] += hgfy[4];
      p_fz[n4si2] += hgfz[4];

      p_fx[n5si2] += hgfx[5];
      p_fy[n5si2] += hgfy[5];
      p_fz[n5si2] += hgfz[5];

      p_fx[n6si2] += hgfx[6];
      p_fy[n6si2] += hgfy[6];
      p_fz[n6si2] += hgfz[6];

      p_fx[n7si2] += hgfx[7];
      p_fy[n7si2] += hgfy[7];
      p_fz[n7si2] += hgfz[7];
#endif
   }

  {
     Index_t numNode = m_numNode;
#ifdef _OPENACC
#pragma acc kernels loop independent present(p_nodeElemCount, p_nodeElemStart, \
p_nodeElemCornerList, fx_elem, fy_elem, fz_elem, p_fx, p_fy, p_fz)
#else
#pragma omp parallel for private(gnode, i) firstprivate(numNode)
#endif
     for( gnode=0 ; gnode<numNode ; ++gnode )
     {
        Index_t count = p_nodeElemCount[gnode];
        Index_t start = p_nodeElemStart[gnode];
        Real_t fx = 0.0 ;
        Real_t fy = 0.0 ;
        Real_t fz = 0.0 ;
#pragma aspen declare param(aspen_param_elemCount:1)
#pragma aspen control loop(aspen_param_elemCount)
        for (i=0 ; i < count ; ++i) {
           Index_t elem = p_nodeElemCornerList[start+i];
           fx += fx_elem[elem] ;
           fy += fy_elem[elem] ;
           fz += fz_elem[elem] ;
        }
        p_fx[gnode] += fx ;
        p_fy[gnode] += fy ;
        p_fz[gnode] += fz ;
     }
  }

/*
  Release(&fz_elem) ;
  Release(&fy_elem) ;
  Release(&fx_elem) ;
*/
}

static inline
void CalcHourglassControlForElems(Real_t determ[T_NUMELEM], Real_t hgcoef,
         Index_t p_nodelist[T_NUMELEM8], Real_t p_volo[T_NUMELEM], Real_t p_v[T_NUMELEM])
{
   Index_t i, ii;
   Index_t numElem = m_numElem;
#pragma aspen declare param(numElem8:m_numElem*8)
   Index_t numElem8 = numElem * 8 ;
#if LULESH_PRINT_SIZE
       printf("T_NUMELEM8\t%d\n", numElem8);
#endif
#if LULESH_CHECK_SIZE
       if (numElem8 != T_NUMELEM8) {
          printf("T_NUMELEM8 should be %d\n", numElem8);
          exit(1);
       }
#endif
/*
   Real_t *dvdx = Allocate(numElem8) ;
   Real_t *dvdy = Allocate(numElem8) ;
   Real_t *dvdz = Allocate(numElem8) ;
   Real_t *x8n  = Allocate(numElem8) ;
   Real_t *y8n  = Allocate(numElem8) ;
   Real_t *z8n  = Allocate(numElem8) ;
*/

   /* start loop over elements */

int abort = 0;

#ifdef _OPENACC
#pragma acc parallel loop present(dvdx, dvdy, dvdz, x8n, y8n, z8n, \
m_x, m_y, m_z, p_volo, p_v, determ, p_nodelist) reduction(||:abort)
#else
#pragma omp parallel for private(i, ii) firstprivate(numElem) reduction(||:abort)
#endif
   for (i=0 ; i<numElem ; ++i){
      Real_t  x1[8],  y1[8],  z1[8] ;
      Real_t pfx[8], pfy[8], pfz[8] ;

#pragma aspen declare data(elemToNode:traits(Array(8,aspen_param_int)))
      Index_t* elemToNode = &p_nodelist[8*i];
      CollectDomainNodesToElemNodes(elemToNode, x1, y1, z1);

      CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

      /* load into temporary storage for FB Hour Glass control */
      for(ii=0;ii<8;++ii){
         Index_t jj=8*i+ii;

         dvdx[jj] = pfx[ii];
         dvdy[jj] = pfy[ii];
         dvdz[jj] = pfz[ii];
         x8n[jj]  = x1[ii];
         y8n[jj]  = y1[ii];
         z8n[jj]  = z1[ii];
      }

      determ[i] = p_volo[i] * p_v[i];

      /* Do a check for negative volumes */
#pragma aspen control ignore
      if ( p_v[i] <= 0.0 ) {
         abort = 1;
      }
   }
   if ( abort ) {
      fprintf(stderr, "VolumeError in CalcHourglassControlForElems(); exit\n");
      exit(VolumeError) ;
   }

#pragma aspen control probability(1)
   if ( hgcoef > 0. ) {
      CalcFBHourglassForceForElems(determ,x8n,y8n,z8n,dvdx,dvdy,dvdz,hgcoef,m_ss,m_nodelist,
              m_elemMass,m_xd,m_yd,m_zd,m_nodeElemCount,m_nodeElemStart,m_nodeElemCornerList,
              m_fx,m_fy,m_fz) ;
   }

/*
   Release(&z8n) ;
   Release(&y8n) ;
   Release(&x8n) ;
   Release(&dvdz) ;
   Release(&dvdy) ;
   Release(&dvdx) ;
*/

   return ;
}

static inline
void CalcVolumeForceForElems()
{
   Index_t k;
   Index_t numElem = m_numElem;
   int abort = 0;
#pragma aspen control probability(1)
   if (numElem != 0) {
      Real_t  hgcoef = m_hgcoef;
#if LULESH_PRINT_SIZE
       printf("T_NUMELEM\t%d\n", numElem);
#endif
#if LULESH_CHECK_SIZE
       if (numElem != T_NUMELEM) {
          printf("T_NUMELEM should be %d\n", numElem);
          exit(1);
       }
#endif
/*
      Real_t *sigxx  = Allocate(numElem) ;
      Real_t *sigyy  = Allocate(numElem) ;
      Real_t *sigzz  = Allocate(numElem) ;
      Real_t *determ = Allocate(numElem) ;
*/

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(numElem, sigxx, sigyy, sigzz, m_p, m_q);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( numElem, sigxx, sigyy, sigzz, determ, m_nodelist,
                               m_x, m_y, m_z, m_nodeElemCount, m_nodeElemStart,
                               m_nodeElemCornerList, m_fx, m_fy, m_fz) ;

      // check for negative element volume

#pragma aspen control ignore
#ifdef _OPENACC
#pragma acc parallel loop independent present(determ) reduction(||: abort)
#else
#pragma omp parallel for private(k) firstprivate(numElem) reduction(||: abort)
#endif
      for ( k=0 ; k<numElem ; ++k ) {
         if (determ[k] <= 0.0) {
            abort = 1;
         }
      }
      if (abort == 1) {
         fprintf(stderr, "VolumeError in CalcVolumeForceForElems(); exit\n");
         exit(VolumeError) ;
      }

      CalcHourglassControlForElems(determ,hgcoef,m_nodelist,m_volo,m_v) ;
/*
      Release(&determ) ;
      Release(&sigzz) ;
      Release(&sigyy) ;
      Release(&sigxx) ;
*/
   }
}

static inline void CalcForceForNodes(Real_t p_fx[T_NUMNODE], Real_t p_fy[T_NUMNODE],
                              Real_t p_fz[T_NUMNODE])

{
  Index_t i;
  Index_t numNode = m_numNode;
#ifdef _OPENACC
#pragma acc parallel loop independent present(p_fx, p_fy, p_fz)
#else
#pragma omp parallel for private(i) firstprivate(numNode)
#endif
  for (i=0; i<numNode; ++i) {
     p_fx[i] = 0.0 ;
     p_fy[i] = 0.0 ;
     p_fz[i] = 0.0 ;
  }

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems() ;

  /* Calculate Nodal Forces at domain boundaries */
  /* problem->commSBN->Transfer(CommSBN::forces); */

}

static inline
void CalcAccelerationForNodes(Real_t p_fx[T_NUMNODE], Real_t p_fy[T_NUMNODE],
                              Real_t p_fz[T_NUMNODE], Real_t p_xdd[T_NUMNODE],
                              Real_t p_ydd[T_NUMNODE], Real_t p_zdd[T_NUMNODE],
                              Real_t p_nodalMass[T_NUMNODE])

{
   Index_t i;
   Index_t numNode = m_numNode;
#ifdef _OPENACC
#pragma acc parallel loop present(p_fx, p_fy, p_fz, p_xdd, p_ydd, p_zdd, \
p_nodalMass)
#else
#pragma omp parallel for private(i) firstprivate(numNode)
#endif
   for (i = 0; i < numNode; ++i) {
      p_xdd[i] = p_fx[i] / p_nodalMass[i];
      p_ydd[i] = p_fy[i] / p_nodalMass[i];
      p_zdd[i] = p_fz[i] / p_nodalMass[i];
   }
}


static inline
void ApplyAccelerationBoundaryConditionsForNodes(Real_t p_xdd[T_NUMNODE], 
          Real_t p_ydd[T_NUMNODE], Real_t p_zdd[T_NUMNODE],
          Index_t p_symmX[T_NUMNODESETS], Index_t p_symmY[T_NUMNODESETS],
          Index_t p_symmZ[T_NUMNODESETS])
{
  Index_t i;
  Index_t numNodeBC = (m_sizeX+1)*(m_sizeX+1) ;
 
#ifdef _OPENACC
#pragma acc parallel firstprivate(numNodeBC) present(p_xdd, p_ydd, p_zdd, \
p_symmX, p_symmY, p_symmZ)
#else
#pragma omp parallel firstprivate(numNodeBC)
#endif
{
#ifdef _OPENACC
#pragma acc loop independent
#else
#pragma omp for nowait private(i)
#endif
  for(i=0 ; i<numNodeBC ; ++i)
    p_xdd[p_symmX[i]] = 0.0 ;

#ifdef _OPENACC
#pragma acc loop independent
#else
#pragma omp for nowait private(i)
#endif
  for(i=0 ; i<numNodeBC ; ++i)
    p_ydd[p_symmY[i]] = 0.0 ;

#ifdef _OPENACC
#pragma acc loop independent
#else
#pragma omp for nowait private(i)
#endif
  for(i=0 ; i<numNodeBC ; ++i)
    p_zdd[p_symmZ[i]] = 0.0 ;
}
}

static inline
void CalcVelocityForNodes(const Real_t dt, const Real_t u_cut,
          Real_t p_xd[T_NUMNODE], Real_t p_yd[T_NUMNODE], Real_t p_zd[T_NUMNODE],
          Real_t p_xdd[T_NUMNODE], Real_t p_ydd[T_NUMNODE], Real_t p_zdd[T_NUMNODE])
{
   Index_t i;
   Index_t numNode = m_numNode;

#ifdef _OPENACC
#pragma acc parallel loop independent present(p_xd, p_yd, p_zd, p_xdd, p_ydd, p_zdd)
#else
#pragma omp parallel for private(i) firstprivate(numNode)
#endif
   for ( i = 0 ; i < numNode ; ++i )
   {
     Real_t xdtmp, ydtmp, zdtmp ;

     xdtmp = p_xd[i] + p_xdd[i] * dt ;
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_xdtmp:0.1)
#pragma aspen control probability(aspen_param_xdtmp)
#else
#pragma aspen control probability(1)
#endif
     if( FABS8(xdtmp) < u_cut ) xdtmp = 0.0;
     p_xd[i] = xdtmp ;

     ydtmp = p_yd[i] + p_ydd[i] * dt ;
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_ydtmp:0.1)
#pragma aspen control probability(aspen_param_ydtmp)
#else
#pragma aspen control probability(1)
#endif
     if( FABS8(ydtmp) < u_cut ) ydtmp = 0.0;
     p_yd[i] = ydtmp ;

     zdtmp = p_zd[i] + p_zdd[i] * dt ;
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_zdtmp:0.1)
#pragma aspen control probability(aspen_param_zdtmp)
#else
#pragma aspen control probability(1)
#endif
     if( FABS8(zdtmp) < u_cut ) zdtmp = 0.0;
     p_zd[i] = zdtmp ;
   }
}

static inline
void CalcPositionForNodes(const Real_t dt,
          Real_t p_x[T_NUMNODE], Real_t p_y[T_NUMNODE], Real_t p_z[T_NUMNODE],
          Real_t p_xd[T_NUMNODE], Real_t p_yd[T_NUMNODE], Real_t p_zd[T_NUMNODE])
{
   Index_t i;
   Index_t numNode = m_numNode;

#ifdef _OPENACC
#pragma acc parallel loop independent present(p_x, p_y, p_z, p_xd, p_yd, p_zd)
#else
#pragma omp parallel for private(i) firstprivate(numNode)
#endif
   for ( i = 0 ; i < numNode ; ++i )
   {
     p_x[i] += p_xd[i] * dt ;
     p_y[i] += p_yd[i] * dt ;
     p_z[i] += p_zd[i] * dt ;
   }
}

static inline
void LagrangeNodal()
{
  const Real_t delt = m_deltatime;
  Real_t u_cut = m_u_cut;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(m_fx,m_fy,m_fz);

  CalcAccelerationForNodes(m_fx,m_fy,m_fz,m_xdd,m_ydd,m_zdd,m_nodalMass);

  ApplyAccelerationBoundaryConditionsForNodes(m_xdd,m_ydd,m_zdd,m_symmX,m_symmY,m_symmZ);

  CalcVelocityForNodes(delt,u_cut,m_xd,m_yd,m_zd,m_xdd,m_ydd,m_zdd);

  CalcPositionForNodes(delt,m_x,m_y,m_z,m_xd,m_yd,m_zd);

  return;
}

static inline
Real_t CalcElemVolumeI( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = 1.0/12.0;

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

static inline
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolumeI( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

static inline
Real_t AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) *
      (fx * gx + fy * gy + fz * gz);
   return area ;
}

static inline
Real_t CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume)
{
   Real_t a, charLength = 0.0;

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ;
   charLength = std_max(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = std_max(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = std_max(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = std_max(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = std_max(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = std_max(a,charLength) ;

   charLength = 4.0 * volume / SQRT8(charLength);

   return charLength;
}

static inline
void CalcElemVelocityGrandient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = 1.0 / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
#pragma aspen declare data(pfx:traits(Array(8, aspen_param_double)))
  const Real_t* const pfx = b[0];
#pragma aspen declare data(pfy:traits(Array(8, aspen_param_double)))
  const Real_t* const pfy = b[1];
#pragma aspen declare data(pfz:traits(Array(8, aspen_param_double)))
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  =  .5 * ( dxddy + dyddx );
  d[4]  =  .5 * ( dxddz + dzddx );
  d[3]  =  .5 * ( dzddy + dyddz );
}

static inline
void CalcKinematicsForElems( Index_t numElem,Real_t dt, 
                             Index_t p_nodelist[T_NUMELEM8], Real_t p_x[T_NUMNODE],
                             Real_t p_y[T_NUMNODE], Real_t p_z[T_NUMNODE],
                             Real_t p_volo[T_NUMELEM], Real_t p_v[T_NUMELEM],
                             Real_t p_vnew[T_NUMELEM], Real_t p_delv[T_NUMELEM],
                             Real_t p_arealg[T_NUMELEM], Real_t p_xd[T_NUMNODE], 
                             Real_t p_yd[T_NUMNODE], Real_t p_zd[T_NUMNODE],
                             Real_t p_dxx[T_NUMELEM], Real_t p_dyy[T_NUMELEM],
                             Real_t p_dzz[T_NUMELEM])

{
  Index_t k, lnode, j;
  // loop over all elements
#ifdef _OPENACC
#pragma acc parallel loop independent present(p_dxx, p_dyy, p_dzz, p_x, p_y, p_z, p_xd, \
p_yd, p_zd, p_v, p_volo, p_vnew, p_delv, p_arealg, p_nodelist)
#else
#pragma omp parallel for private(k, lnode, j) firstprivate(numElem, dt)
#endif
  for( k=0 ; k<numElem ; ++k )
  {
     Real_t B[3][8] ; /** shape function derivatives */
     Real_t D[6] ;
     Real_t x_local[8] ;
     Real_t y_local[8] ;
     Real_t z_local[8] ;
     Real_t xd_local[8] ;
     Real_t yd_local[8] ;
     Real_t zd_local[8] ;
     Real_t detJ = 0.0 ;

    Real_t volume ;
    Real_t relativeVolume ;
#pragma aspen declare data(elemToNode:traits(Array(8,aspen_param_int)))
    const Index_t* const elemToNode = &p_nodelist[8*k] ;
    Real_t dt2;

    // get nodal coordinates from global arrays and copy into local arrays.
    for( lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemToNode[lnode];
      x_local[lnode] = p_x[gnode];
      y_local[lnode] = p_y[gnode];
      z_local[lnode] = p_z[gnode];
    }

    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local );
    relativeVolume = volume / p_volo[k] ;
    p_vnew[k] = relativeVolume ;
    p_delv[k] = relativeVolume - p_v[k] ;

    // set characteristic length
    p_arealg[k] = CalcElemCharacteristicLength(x_local,
                                                  y_local,
                                                  z_local,
                                                  volume);

    // get nodal velocities from global array and copy into local arrays.
    for( lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemToNode[lnode];
      xd_local[lnode] = p_xd[gnode];
      yd_local[lnode] = p_yd[gnode];
      zd_local[lnode] = p_zd[gnode];
    }

    dt2 = 0.5 * dt;
    for ( j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j];
    }

    CalcElemShapeFunctionDerivatives( x_local,
                                          y_local,
                                          z_local,
                                          B, &detJ );

    CalcElemVelocityGrandient( xd_local,
                               yd_local,
                               zd_local,
                               B, detJ, D );

    // put velocity gradient quantities into their global arrays.
    p_dxx[k] = D[0];
    p_dyy[k] = D[1];
    p_dzz[k] = D[2];
  }
}

static inline
void CalcLagrangeElements(Real_t deltatime,
                          Real_t p_vnew[T_NUMELEM], Real_t p_vdov[T_NUMELEM],
                          Real_t p_dxx[T_NUMELEM], Real_t p_dyy[T_NUMELEM],
                          Real_t p_dzz[T_NUMELEM])

{
   Index_t k;
   Index_t numElem = m_numElem;
   int abort = 0;
#pragma aspen control probability(1)
   if (numElem > 0) {
      CalcKinematicsForElems(numElem,deltatime,m_nodelist,m_x,m_y,m_z,m_volo,m_v,p_vnew,
          m_delv,m_arealg,m_xd,m_yd,m_zd,p_dxx,p_dyy,p_dzz);

      // element loop to do some stuff not included in the elemlib function.
#ifdef _OPENACC
#pragma acc parallel loop independent present(p_vdov, p_dxx, p_dyy, p_dzz, p_vnew) \
reduction(||: abort)
#else
#pragma omp parallel for private(k) firstprivate(numElem) reduction(||:abort)
#endif
      for ( k=0 ; k<numElem ; ++k )
      {
        // calc strain rate and apply as constraint (only done in FB element)
        Real_t vdov = p_dxx[k] + p_dyy[k] + p_dzz[k] ;
        Real_t vdovthird = vdov/3.0 ;
        
        // make the rate of deformation tensor deviatoric
        p_vdov[k] = vdov ;
        p_dxx[k] -= vdovthird ;
        p_dyy[k] -= vdovthird ;
        p_dzz[k] -= vdovthird ;

        // See if any volumes are negative, and take appropriate action.
#pragma aspen control ignore
        if (p_vnew[k] <= 0.0)
        {
           abort = 1;
        }
      }
      if (abort == 1)
      {
         fprintf(stderr, "VolumeError in CalcLagrangeElements(); exit\n");
         exit(VolumeError) ;
      }
   }
}

static inline
void CalcMonotonicQGradientsForElems(Index_t p_nodelist[T_NUMELEM8], 
        Real_t p_x[T_NUMNODE], Real_t p_y[T_NUMNODE], Real_t p_z[T_NUMNODE],
        Real_t p_xd[T_NUMNODE], Real_t p_yd[T_NUMNODE],Real_t p_zd[T_NUMNODE],
        Real_t p_volo[T_NUMELEM], Real_t p_vnew[T_NUMELEM],
        Real_t p_delx_zeta[T_NUMELEM], Real_t p_delv_zeta[T_NUMELEM],
        Real_t p_delx_xi[T_NUMELEM], Real_t p_delv_xi[T_NUMELEM],
        Real_t p_delx_eta[T_NUMELEM], Real_t p_delv_eta[T_NUMELEM])
{
   Index_t i;
#define SUM4(a,b,c,d) (a + b + c + d)
   Index_t numElem = m_numElem;

#ifdef _OPENACC
#pragma acc parallel loop independent present(p_vnew, p_nodelist, p_x, p_y, p_z, p_xd, \
p_yd, p_zd, p_volo, p_delx_xi, p_delx_eta, p_delx_zeta, p_delv_xi, p_delv_eta,\
p_delv_zeta)
#else
#pragma omp parallel for private(i) firstprivate(numElem)
#endif
   for (i = 0 ; i < numElem ; ++i ) {
      const Real_t ptiny = 1.e-36 ;
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

#pragma aspen declare data(elemToNode:traits(Array(8,aspen_param_int)))
      const Index_t *elemToNode = &p_nodelist[8*i];
      Index_t n0 = elemToNode[0] ;
      Index_t n1 = elemToNode[1] ;
      Index_t n2 = elemToNode[2] ;
      Index_t n3 = elemToNode[3] ;
      Index_t n4 = elemToNode[4] ;
      Index_t n5 = elemToNode[5] ;
      Index_t n6 = elemToNode[6] ;
      Index_t n7 = elemToNode[7] ;

      Real_t x0 = p_x[n0] ;
      Real_t x1 = p_x[n1] ;
      Real_t x2 = p_x[n2] ;
      Real_t x3 = p_x[n3] ;
      Real_t x4 = p_x[n4] ;
      Real_t x5 = p_x[n5] ;
      Real_t x6 = p_x[n6] ;
      Real_t x7 = p_x[n7] ;

      Real_t y0 = p_y[n0] ;
      Real_t y1 = p_y[n1] ;
      Real_t y2 = p_y[n2] ;
      Real_t y3 = p_y[n3] ;
      Real_t y4 = p_y[n4] ;
      Real_t y5 = p_y[n5] ;
      Real_t y6 = p_y[n6] ;
      Real_t y7 = p_y[n7] ;

      Real_t z0 = p_z[n0] ;
      Real_t z1 = p_z[n1] ;
      Real_t z2 = p_z[n2] ;
      Real_t z3 = p_z[n3] ;
      Real_t z4 = p_z[n4] ;
      Real_t z5 = p_z[n5] ;
      Real_t z6 = p_z[n6] ;
      Real_t z7 = p_z[n7] ;

      Real_t xv0 = p_xd[n0] ;
      Real_t xv1 = p_xd[n1] ;
      Real_t xv2 = p_xd[n2] ;
      Real_t xv3 = p_xd[n3] ;
      Real_t xv4 = p_xd[n4] ;
      Real_t xv5 = p_xd[n5] ;
      Real_t xv6 = p_xd[n6] ;
      Real_t xv7 = p_xd[n7] ;

      Real_t yv0 = p_yd[n0] ;
      Real_t yv1 = p_yd[n1] ;
      Real_t yv2 = p_yd[n2] ;
      Real_t yv3 = p_yd[n3] ;
      Real_t yv4 = p_yd[n4] ;
      Real_t yv5 = p_yd[n5] ;
      Real_t yv6 = p_yd[n6] ;
      Real_t yv7 = p_yd[n7] ;

      Real_t zv0 = p_zd[n0] ;
      Real_t zv1 = p_zd[n1] ;
      Real_t zv2 = p_zd[n2] ;
      Real_t zv3 = p_zd[n3] ;
      Real_t zv4 = p_zd[n4] ;
      Real_t zv5 = p_zd[n5] ;
      Real_t zv6 = p_zd[n6] ;
      Real_t zv7 = p_zd[n7] ;

      Real_t vol = p_volo[i]*p_vnew[i] ;
      Real_t norm = 1.0 / ( vol + ptiny ) ;

      Real_t dxj = -0.25*(SUM4(x0,x1,x5,x4) - SUM4(x3,x2,x6,x7)) ;
      Real_t dyj = -0.25*(SUM4(y0,y1,y5,y4) - SUM4(y3,y2,y6,y7)) ;
      Real_t dzj = -0.25*(SUM4(z0,z1,z5,z4) - SUM4(z3,z2,z6,z7)) ;

      Real_t dxi =  0.25*(SUM4(x1,x2,x6,x5) - SUM4(x0,x3,x7,x4)) ;
      Real_t dyi =  0.25*(SUM4(y1,y2,y6,y5) - SUM4(y0,y3,y7,y4)) ;
      Real_t dzi =  0.25*(SUM4(z1,z2,z6,z5) - SUM4(z0,z3,z7,z4)) ;

      Real_t dxk =  0.25*(SUM4(x4,x5,x6,x7) - SUM4(x0,x1,x2,x3)) ;
      Real_t dyk =  0.25*(SUM4(y4,y5,y6,y7) - SUM4(y0,y1,y2,y3)) ;
      Real_t dzk =  0.25*(SUM4(z4,z5,z6,z7) - SUM4(z0,z1,z2,z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      p_delx_zeta[i] = vol / SQRT8(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = 0.25*(SUM4(xv4,xv5,xv6,xv7) - SUM4(xv0,xv1,xv2,xv3)) ;
      dyv = 0.25*(SUM4(yv4,yv5,yv6,yv7) - SUM4(yv0,yv1,yv2,yv3)) ;
      dzv = 0.25*(SUM4(zv4,zv5,zv6,zv7) - SUM4(zv0,zv1,zv2,zv3)) ;

      p_delv_zeta[i] = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      p_delx_xi[i] = vol / SQRT8(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = 0.25*(SUM4(xv1,xv2,xv6,xv5) - SUM4(xv0,xv3,xv7,xv4)) ;
      dyv = 0.25*(SUM4(yv1,yv2,yv6,yv5) - SUM4(yv0,yv3,yv7,yv4)) ;
      dzv = 0.25*(SUM4(zv1,zv2,zv6,zv5) - SUM4(zv0,zv3,zv7,zv4)) ;

      p_delv_xi[i] = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      p_delx_eta[i] = vol / SQRT8(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = -0.25*(SUM4(xv0,xv1,xv5,xv4) - SUM4(xv3,xv2,xv6,xv7)) ;
      dyv = -0.25*(SUM4(yv0,yv1,yv5,yv4) - SUM4(yv3,yv2,yv6,yv7)) ;
      dzv = -0.25*(SUM4(zv0,zv1,zv5,zv4) - SUM4(zv3,zv2,zv6,zv7)) ;

      p_delv_eta[i] = ax*dxv + ay*dyv + az*dzv ;
   }
#undef SUM4
}

static inline
void CalcMonotonicQRegionForElems(// parameters
                          Real_t qlc_monoq,
                          Real_t qqc_monoq,
                          Real_t monoq_limiter_mult,
                          Real_t monoq_max_slope,
                          Real_t ptiny,
                          // the elementset length
                          Index_t elength,
                          Index_t p_matElemlist[T_NUMELEM], Int_t p_elemBC[T_NUMELEM],
                          Real_t p_delx_xi[T_NUMELEM], Real_t p_delx_eta[T_NUMELEM],
                          Real_t p_delx_zeta[T_NUMELEM],
                          Real_t p_delv_xi[T_NUMELEM], Real_t p_delv_eta[T_NUMELEM],
                          Real_t p_delv_zeta[T_NUMELEM],
                          Index_t p_lxim[T_NUMELEM],Index_t p_lxip[T_NUMELEM],
                          Index_t p_letam[T_NUMELEM],Index_t p_letap[T_NUMELEM],
                          Index_t p_lzetam[T_NUMELEM],Index_t p_lzetap[T_NUMELEM],
                          Real_t p_vnew[T_NUMELEM], Real_t p_vdov[T_NUMELEM],
                          Real_t p_volo[T_NUMELEM], Real_t p_elemMass[T_NUMELEM],
                          Real_t p_qq[T_NUMELEM], Real_t p_ql[T_NUMELEM])
{
   Index_t ielem;
#ifdef _OPENACC
#pragma acc parallel loop independent firstprivate(qlc_monoq, qqc_monoq, monoq_limiter_mult,\
monoq_max_slope, ptiny) present(p_matElemlist) present(p_vnew, p_vdov, p_delx_xi,\
p_delx_eta, p_delx_zeta, p_delv_xi, p_delv_eta, p_delv_zeta, p_elemMass, \
p_volo, p_lxip, p_lxim, p_letam, p_letap, p_lzetam, p_lzetap, p_ql, p_qq, \
p_elemBC)
#else 
#pragma omp parallel for private(ielem) firstprivate(elength, qlc_monoq, qqc_monoq, monoq_limiter_mult, monoq_max_slope, ptiny)
#endif
   for ( ielem = 0 ; ielem < elength; ++ielem ) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Index_t i = p_matElemlist[ielem];
      Int_t bcMask = p_elemBC[i] ;
      Real_t delvm, delvp ;

      /*  phixi     */
      Real_t norm = 1. / ( p_delv_xi[i] + ptiny ) ;

      switch (bcMask & XI_M) {
         case 0:         delvm = p_delv_xi[p_lxim[i]] ; break ;
         case XI_M_SYMM: delvm = p_delv_xi[i] ;            break ;
         case XI_M_FREE: delvm = 0.0 ;                break ;
         default:        /* ERROR */ ;                        break ;
      }
      switch (bcMask & XI_P) {
         case 0:         delvp = p_delv_xi[p_lxip[i]] ; break ;
         case XI_P_SYMM: delvp = p_delv_xi[i] ;            break ;
         case XI_P_FREE: delvp = 0.0 ;                break ;
         default:        /* ERROR */ ;                        break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = .5 * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < 0.) phixi = 0. ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = 1. / ( p_delv_eta[i] + ptiny ) ;

      switch (bcMask & ETA_M) {
         case 0:          delvm = p_delv_eta[p_letam[i]] ; break ;
         case ETA_M_SYMM: delvm = p_delv_eta[i] ;             break ;
         case ETA_M_FREE: delvm = 0.0 ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }
      switch (bcMask & ETA_P) {
         case 0:          delvp = p_delv_eta[p_letap[i]] ; break ;
         case ETA_P_SYMM: delvp = p_delv_eta[i] ;             break ;
         case ETA_P_FREE: delvp = 0.0 ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = .5 * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < 0.) phieta = 0. ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = 1. / ( p_delv_zeta[i] + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case 0:           delvm = p_delv_zeta[p_lzetam[i]] ; break ;
         case ZETA_M_SYMM: delvm = p_delv_zeta[i] ;              break ;
         case ZETA_M_FREE: delvm = 0.0 ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }
      switch (bcMask & ZETA_P) {
         case 0:           delvp = p_delv_zeta[p_lzetap[i]] ; break ;
         case ZETA_P_SYMM: delvp = p_delv_zeta[i] ;              break ;
         case ZETA_P_FREE: delvp = 0.0 ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = .5 * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < 0.) phizeta = 0.;
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_vdov2:0.1)
#pragma aspen control probability(aspen_param_vdov2)
//#pragma aspen control probability(aspen_param_vdov2) flops(1:traits(dp)) loads(1*aspen_param_double:from(p_vdov):traits(stride(1))) 
#else
#pragma aspen control probability(1)
#endif
      if ( p_vdov[i] > 0. )  {
         qlin  = 0. ;
         qquad = 0. ;
      }
      else {
         Real_t delvxxi   = p_delv_xi[i]   * p_delx_xi[i]   ;
         Real_t delvxeta  = p_delv_eta[i]  * p_delx_eta[i]  ;
         Real_t delvxzeta = p_delv_zeta[i] * p_delx_zeta[i] ;
         Real_t rho;

#pragma aspen control ignore
         if ( delvxxi   > 0. ) delvxxi   = 0. ;
#pragma aspen control ignore
         if ( delvxeta  > 0. ) delvxeta  = 0. ;
#pragma aspen control ignore
         if ( delvxzeta > 0. ) delvxzeta = 0. ;

         rho = p_elemMass[i] / (p_volo[i] * p_vnew[i]) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (1. - phixi) +
               delvxeta  * (1. - phieta) +
               delvxzeta * (1. - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (1. - phixi*phixi) +
               delvxeta*delvxeta   * (1. - phieta*phieta) +
               delvxzeta*delvxzeta * (1. - phizeta*phizeta)  ) ;
      }

      p_qq[i] = qquad ;
      p_ql[i] = qlin  ;
   }
}

static inline
void CalcMonotonicQForElems()
{  
   //
   // initialize parameters
   // 
   const Real_t ptiny        = 1.e-36 ;
   Real_t monoq_max_slope    = m_monoq_max_slope;
   Real_t monoq_limiter_mult = m_monoq_limiter_mult;

   //
   // calculate the monotonic q for pure regions
   //
   Index_t elength = m_numElem;
#pragma aspen control probability(1)
   if (elength > 0) {
      Real_t qlc_monoq = m_qlc_monoq;
      Real_t qqc_monoq = m_qqc_monoq;
      CalcMonotonicQRegionForElems(// parameters
                           qlc_monoq,
                           qqc_monoq,
                           monoq_limiter_mult,
                           monoq_max_slope,
                           ptiny,
                           // the elemset length
                           elength,
                           m_matElemlist,m_elemBC,m_delx_xi,m_delx_eta,m_delx_zeta,
                           m_delv_xi,m_delv_eta,m_delv_zeta,m_lxim,m_lxip,m_letam,m_letap,
                           m_lzetam,m_lzetap,m_vnew,m_vdov,m_volo,m_elemMass,m_qq,m_ql);
   }
}

static inline
void CalcQForElems()
{
   Index_t i;
   Real_t qstop = m_qstop;
   Index_t numElem = m_numElem;

   //
   // MONOTONIC Q option
   //

   /* Calculate velocity gradients */
   CalcMonotonicQGradientsForElems(m_nodelist,m_x,m_y,m_z,m_xd,m_yd,m_zd,m_volo,m_vnew,
       m_delx_zeta,m_delv_zeta,m_delx_xi,m_delv_xi,m_delx_eta,m_delv_eta) ;

   /* Transfer veloctiy gradients in the first order elements */
   /* problem->commElements->Transfer(CommElements::monoQ) ; */
   CalcMonotonicQForElems() ;

   /* Don't allow excessive artificial viscosity */
#pragma aspen control ignore
   if (numElem != 0) {
      //Index_t idx = -1; 
      Index_t idx = 0; 
#ifdef _OPENACC
#pragma acc parallel loop independent present(m_q) reduction(||:idx)
#else
#pragma omp parallel for reduction(||:idx)
#endif
      for (i=0; i<numElem; ++i) {
         if ( m_q[i] > qstop ) {
            idx = 1 ;
            //break ;
         }
      }

      if(idx == 1) {
         fprintf(stderr, "QStopError in CalcQForElems(); exit\n");
         exit(QStopError) ;
      }
   }
}

static inline
void CalcPressureForElems(Real_t p_new[T_NUMELEM], Real_t bvc[T_NUMELEM],
                          Real_t pbvc[T_NUMELEM], Real_t e_old[T_NUMELEM],
                          Real_t compression[T_NUMELEM], Real_t vnewc[T_NUMELEM],
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          Index_t length)
{

   Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop independent present(bvc, pbvc, compression)
#else
#pragma omp parallel for private(i) firstprivate(length)
#endif
   for (i = 0; i < length ; ++i) {
      Real_t c1s = 2.0/3.0 ;
      bvc[i] = c1s * (compression[i] + 1.);
      pbvc[i] = c1s;
   }

#ifdef _OPENACC
#pragma acc parallel loop independent present(bvc, p_new, e_old, vnewc)
#else
#pragma omp parallel for private(i) firstprivate(length, pmin, p_cut, eosvmax)
#endif
   for (i = 0 ; i < length ; ++i){
      p_new[i] = bvc[i] * e_old[i] ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_pnew:0.1)
#pragma aspen control probability(aspen_param_pnew)
#else
#pragma aspen control probability(1)
#endif
      if    (FABS8(p_new[i]) <  p_cut   )
         p_new[i] = 0.0 ;

#pragma aspen control ignore
      if    ( vnewc[i] >= eosvmax ) /* impossible condition here? */
         p_new[i] = 0.0 ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_pnew2:0.1)
#pragma aspen control probability(aspen_param_pnew2)
#else
#pragma aspen control probability(1)
#endif
      if    (p_new[i]       <  pmin)
         p_new[i]   = pmin ;
   }
}

static inline
void CalcEnergyForElems(Real_t p_new[T_LENGTH], Real_t e_new[T_LENGTH], Real_t q_new[T_LENGTH],
                        Real_t bvc[T_LENGTH], Real_t pbvc[T_LENGTH],
                        Real_t p_old[T_LENGTH], Real_t e_old[T_LENGTH], Real_t q_old[T_LENGTH],
                        Real_t compression[T_LENGTH], Real_t compHalfStep[T_LENGTH],
                        Real_t vnewc[T_LENGTH], Real_t* work, Real_t* delvc, Real_t pmin,
                        Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
                        Real_t* qq, Real_t* ql,
                        Real_t rho0,
                        Real_t eosvmax,
                        Index_t length)
{
   Index_t i;
#if LULESH_PRINT_SIZE
       printf("T_LENGTH\t%d\n", length);
#endif
#if LULESH_CHECK_SIZE
       if (length != T_LENGTH) {
          printf("T_LENGTH should be %d\n", length);
          exit(1);
       }
#endif
   Real_t *pHalfStep = Allocate(length) ;

#ifdef _OPENACC
#pragma acc data create(pHalfStep[0:length])
{
#endif
#ifdef _OPENACC
#pragma acc parallel loop independent present(e_new, e_old, p_old, q_old, delvc, work)
#else
#pragma omp parallel for private(i) firstprivate(length, emin)
#endif
   for (i = 0 ; i < length ; ++i) {
      e_new[i] = e_old[i] - 0.5 * delvc[i] * (p_old[i] + q_old[i])
         + 0.5 * work[i];

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_e_new_prob:1)
#pragma aspen control probability(aspen_param_e_new_prob)
#else
#pragma aspen control probability(1)
#endif
      if (e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                   pmin, p_cut, eosvmax, length);

#ifdef _OPENACC
#pragma acc parallel loop independent present(compHalfStep, pHalfStep, delvc, p_old,\
q_old, ql, qq, q_new, pbvc, bvc, e_new)
#else
#pragma omp parallel for private(i) firstprivate(length, rho0)
#endif
   for (i = 0 ; i < length ; ++i) {
      Real_t vhalf = 1. / (1. + compHalfStep[i]) ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_delvc:0.1)
#pragma aspen control probability(aspen_param_delvc)
#else
#pragma aspen control probability(1)
#endif
      if ( delvc[i] > 0. ) {
         q_new[i] /* = qq[i] = ql[i] */ = 0. ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;

#pragma aspen control ignore
         if ( ssc <= 0. ) {
            ssc =.333333e-36 ;
         } else {
            ssc = SQRT8(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] + 0.5 * delvc[i]
         * (  3.0*(p_old[i]     + q_old[i])
              - 4.0*(pHalfStep[i] + q_new[i])) ;
   }

#ifdef _OPENACC
#pragma acc parallel loop present(e_new, work)
#else
#pragma omp parallel for private(i) firstprivate(length, emin, e_cut)
#endif
   for (i = 0 ; i < length ; ++i) {

      e_new[i] += 0.5 * work[i];

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_enew:0.1)
#pragma aspen control probability(aspen_param_enew)
#else
#pragma aspen control probability(1)
#endif
      if (FABS8(e_new[i]) < e_cut) {
         e_new[i] = 0.  ;
      }
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_enew2:0.1)
#pragma aspen control probability(aspen_param_enew2)
#else
#pragma aspen control probability(1)
#endif
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

#ifdef _OPENACC
#pragma acc parallel loop present(pHalfStep, delvc, pbvc, e_new, bvc, ql,\
qq, p_old, q_old, p_new, q_new, vnewc)
#else
#pragma omp parallel for private(i) firstprivate(length, rho0, emin, e_cut)
#endif
   for (i = 0 ; i < length ; ++i){
      const Real_t sixth = 1.0 / 6.0 ;
      Real_t q_tilde ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_delvc:0.1)
#pragma aspen control probability(aspen_param_delvc)
#else
#pragma aspen control probability(1)
#endif
      if (delvc[i] > 0.) {
         q_tilde = 0. ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;

#pragma aspen control ignore
         if ( ssc <= 0. ) {
            ssc = .333333e-36 ;
         } else {
            ssc = SQRT8(ssc) ;
         }

         q_tilde = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] - (  7.0*(p_old[i]     + q_old[i])
                               - 8.0*(pHalfStep[i] + q_new[i])
                               + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_enew3:0.1)
#pragma aspen control probability(aspen_param_enew3)
#else
#pragma aspen control probability(1)
#endif
      if (FABS8(e_new[i]) < e_cut) {
         e_new[i] = 0.  ;
      }
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_enew4:0.1)
#pragma aspen control probability(aspen_param_enew4)
#else
#pragma aspen control probability(1)
#endif
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

#ifdef _OPENACC
#pragma acc parallel loop present(delvc, pbvc, e_new, vnewc, bvc, ql,\
qq, p_new, q_new)
#else
#pragma omp parallel for private(i) firstprivate(length, rho0, q_cut)
#endif
   for (i = 0 ; i < length ; ++i){

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_delvc2:0.1)
#pragma aspen control probability(aspen_param_delvc2)
#else
#pragma aspen control probability(1)
#endif
      if ( delvc[i] <= 0. ) {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;

#pragma aspen control ignore
         if ( ssc <= 0. ) {
            ssc = .333333e-36 ;
         } else {
            ssc = SQRT8(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_qnew:0.1)
#pragma aspen control probability(aspen_param_qnew)
#else
#pragma aspen control probability(1)
#endif
         if (FABS8(q_new[i]) < q_cut) q_new[i] = 0. ;
      }
   }

#ifdef _OPENACC
} //end acc data
#endif

   Release(&pHalfStep) ;

   return ;
}

static inline
void CalcSoundSpeedForElems(Real_t vnewc[T_LENGTH], Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t ss4o3, Index_t nz,
                            Index_t p_matElemlist[T_NUMELEM], Real_t p_ss[T_NUMELEM])
{
   Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(vnewc, \
                                  p_matElemlist, \
                                  pbvc, \
                                  enewc, \
                                  bvc, \
                                  pnewc, \
                                  p_ss) \
                          firstprivate(rho0)
#else
#pragma omp parallel for private(i) firstprivate(nz, rho0)
#endif
   for (i = 0; i < nz ; ++i) {
      Index_t iz = p_matElemlist[i];
      Real_t ssTmp = (pbvc[i] * enewc[i] + vnewc[i] * vnewc[i] *
                 bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= 1.111111e-36) {
         ssTmp = 1.111111e-36;
      }
      p_ss[iz] = SQRT8(ssTmp);
   }
}

static inline
void EvalEOSForElems(Real_t vnewc[T_LENGTH], Index_t length,
         Index_t p_matElemlist[T_NUMELEM], Real_t p_e[T_NUMELEM],
         Real_t p_delv[T_NUMELEM], Real_t p_p[T_NUMELEM], Real_t p_q[T_NUMELEM],
         Real_t p_qq[T_NUMELEM], Real_t p_ql[T_NUMELEM], Real_t p_ss[T_NUMELEM])
{
   Real_t  e_cut = m_e_cut;
   Real_t  p_cut = m_p_cut;
   Real_t  ss4o3 = m_ss4o3;
   Real_t  q_cut = m_q_cut;

   Real_t eosvmax = m_eosvmax ;
   Real_t eosvmin = m_eosvmin ;
   Real_t pmin    = m_pmin ;
   Real_t emin    = m_emin ;
   Real_t rho0    = m_refdens ;

#if LULESH_PRINT_SIZE
       printf("T_LENGTH\t%d\n", length);
#endif
#if LULESH_CHECK_SIZE
       if (length != T_LENGTH) {
          printf("T_LENGTH should be %d\n", length);
          exit(1);
       }
#endif
/*
   Real_t *e_old = Allocate(length) ;
   Real_t *delvc = Allocate(length) ;
   Real_t *p_old = Allocate(length) ;
   Real_t *q_old = Allocate(length) ;
   Real_t *compression = Allocate(length) ;
   Real_t *compHalfStep = Allocate(length) ;
   Real_t *qq = Allocate(length) ;
   Real_t *ql = Allocate(length) ;
   Real_t *work = Allocate(length) ;
   Real_t *p_new = Allocate(length) ;
   Real_t *e_new = Allocate(length) ;
   Real_t *q_new = Allocate(length) ;
   Real_t *bvc = Allocate(length) ;
   Real_t *pbvc = Allocate(length) ;
*/

   Index_t i;

   /* compress data, minimal set */
#ifdef _OPENACC
#pragma acc data present(e_old, delvc, p_old, compression,\
compHalfStep, qq, ql, work, p_new, e_new, q_new, \
bvc, pbvc) present(p_matElemlist) 
# else
#pragma omp parallel
#endif
   {
#ifdef _OPENACC
#pragma acc parallel loop independent present(e_old, delvc, p_old, q_old, \
p_e, p_delv, p_p, p_q, p_matElemlist)
#else
#pragma omp for private(i) firstprivate(length)
#endif
      for (i=0; i<length; ++i) {
         Index_t zidx = p_matElemlist[i] ;
         e_old[i] = p_e[zidx] ;
         delvc[i] = p_delv[zidx] ;
         p_old[i] = p_p[zidx] ;
         q_old[i] = p_q[zidx] ;
      }

#ifdef _OPENACC
#pragma acc parallel loop independent present(compression, vnewc, delvc, compHalfStep)
#else
#pragma omp for private(i) firstprivate(length)
#endif
      for (i = 0; i < length ; ++i) {
         Real_t vchalf ;
         compression[i] = 1. / vnewc[i] - 1.;
         vchalf = vnewc[i] - delvc[i] * .5;
         compHalfStep[i] = 1. / vchalf - 1.;
      }

   /* Check for v > eosvmax or v < eosvmin */

#ifdef _OPENACC
#pragma acc parallel loop independent present(vnewc, compHalfStep, compression, \
p_matElemlist, qq, ql, p_qq, p_ql, p_old, work)
#else
#pragma omp for private(i) firstprivate(length, eosvmax, eosvmin)
#endif
      for (i = 0 ; i < length ; ++i) {
         Index_t zidx = p_matElemlist[i] ;
         qq[i] = p_qq[zidx] ;
         ql[i] = p_ql[zidx] ;
         work[i] = 0. ; 
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_vnewc:0)
#pragma aspen control probability(aspen_param_vnewc)
#else
#pragma aspen control probability(1)
#endif
		 if( (eosvmin != 0.0) && (vnewc[i] <= eosvmin) ) {
               compHalfStep[i] = compression[i] ;
		 }
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_vnewc2:0)
#pragma aspen control probability(aspen_param_vnewc2)
#else
#pragma aspen control probability(1)
#endif
		 if( (eosvmax != 0.0) && (vnewc[i] >= eosvmax) ) {
               p_old[i]        = 0. ;
               compression[i]  = 0. ;
               compHalfStep[i] = 0. ;
		 }
      }
   }

   CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc,
                 p_old, e_old,  q_old, compression, compHalfStep,
                 vnewc, work,  delvc, pmin,
                 p_cut, e_cut, q_cut, emin,
                 qq, ql, rho0, eosvmax, length);

#ifdef _OPENACC
#pragma acc parallel loop independent present(p_new, e_new, q_new, p_p, p_e, p_q, p_matElemlist)
#else
#pragma omp parallel for firstprivate(length)
#endif
      for (i=0; i<length; ++i) {
         Index_t zidx = p_matElemlist[i] ;
         p_p[zidx] = p_new[i] ;
         p_e[zidx] = e_new[i] ;
         p_q[zidx] = q_new[i] ;
      }

   CalcSoundSpeedForElems(vnewc, rho0, e_new, p_new,
             pbvc, bvc, ss4o3, length, p_matElemlist, p_ss) ;

/*
   Release(&pbvc) ;
   Release(&bvc) ;
   Release(&q_new) ;
   Release(&e_new) ;
   Release(&p_new) ;
   Release(&work) ;
   Release(&ql) ;
   Release(&qq) ;
   Release(&compHalfStep) ;
   Release(&compression) ;
   Release(&q_old) ;
   Release(&p_old) ;
   Release(&delvc) ;
   Release(&e_old) ;
*/
}

static inline
void ApplyMaterialPropertiesForElems(Index_t p_matElemlist[T_NUMELEM],
         Real_t p_vnew[T_NUMELEM], Real_t p_v[T_NUMELEM],
         Real_t p_e[T_NUMELEM],
         Real_t p_delv[T_NUMELEM], Real_t p_p[T_NUMELEM], Real_t p_q[T_NUMELEM],
         Real_t p_qq[T_NUMELEM], Real_t p_ql[T_NUMELEM], Real_t p_ss[T_NUMELEM])

{
  Index_t i;
  Index_t length = m_numElem;

#pragma aspen control probability(1)
  if (length != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = m_eosvmin;
    Real_t eosvmax = m_eosvmax;
#if LULESH_PRINT_SIZE
       printf("T_LENGTH\t%d\n", length);
#endif
#if LULESH_CHECK_SIZE
       if (length != T_LENGTH) {
          printf("T_LENGTH should be %d\n", length);
          exit(1);
       }
#endif
/*
    Real_t *vnewc = Allocate(length) ;
*/

Real_t vc = 1.0;

#ifdef _OPENACC
#pragma acc data present(vnewc[0:m_numElem]) present(p_v, p_matElemlist)
#else
#pragma omp parallel firstprivate(length, eosvmin, eosvmax)
#endif
    {
#ifdef _OPENACC
#pragma acc parallel loop independent
#else
#pragma omp for private(i)
#endif
       for (i=0 ; i<length ; ++i) {
          Index_t zn = p_matElemlist[i] ;
          vnewc[i] = p_vnew[zn] ;
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_vnewc3:0.1)
#pragma aspen control probability(aspen_param_vnewc3)
#else
#pragma aspen control probability(1)
#endif
          if( (eosvmin != 0.0) && (vnewc[i] < eosvmin) ) {
                vnewc[i] = eosvmin ;
          }
#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_vnewc4:0.1)
#pragma aspen control probability(aspen_param_vnewc4)
#else
#pragma aspen control probability(1)
#endif
          if( (eosvmax != 0.0) && (vnewc[i] > eosvmax) ) {
                vnewc[i] = eosvmax ;
          }
       }

#ifdef _OPENACC
#pragma acc parallel loop reduction(min: vc) present(p_v, p_matElemlist)
#else
//#pragma omp for private(i) reduction(min:vc) //min is not recognized by GCC.
#pragma omp for private(i)
#endif
       for (i=0; i<length; ++i) {
          Index_t zn = p_matElemlist[i] ;
          vc = p_v[zn] ;
#pragma aspen control execute flops(2:traits(dp))
          if (eosvmin != 0.) {
             if (vc < eosvmin)
                vc = eosvmin ;
          }
#pragma aspen control execute flops(2:traits(dp))
          if (eosvmax != 0.) {
             if (vc > eosvmax)
                vc = eosvmax ;
          }
#ifndef _OPENACC
          if (vc <= 0.) {
             fprintf(stderr, "VolumeError in ApplyMaterialPropertiesForElems(); exit\n");
             exit(VolumeError) ;
          }
#endif
       }
    }
#ifdef _OPENACC
#pragma aspen control ignore
          if (vc <= 0.) {
             fprintf(stderr, "VolumeError in ApplyMaterialPropertiesForElems(); exit\n");
             exit(VolumeError) ;
          }
#endif

    EvalEOSForElems(vnewc,length,p_matElemlist,p_e,p_delv,p_p,p_q,p_qq,p_ql,p_ss);

/*
    Release(&vnewc) ;
*/

  }
}

static inline
void UpdateVolumesForElems(Real_t p_vnew[T_NUMELEM], Real_t p_v[T_NUMELEM])
{
   Index_t i;
   Index_t numElem = m_numElem;
#pragma aspen control probability(1)
   if (numElem != 0) {
      Real_t v_cut = m_v_cut;

#ifdef _OPENACC
#pragma acc parallel loop present(p_vnew, p_v)
#else
#pragma omp parallel for private(i) firstprivate(numElem,v_cut)
#endif
      for(i=0 ; i<numElem ; ++i) {
         Real_t tmpV ;
         tmpV = p_vnew[i] ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_tmpV:0.1)
#pragma aspen control probability(aspen_param_tmpV)
#else
#pragma aspen control probability(1)
#endif
         if ( FABS8(tmpV - 1.0) < v_cut )
            tmpV = 1.0 ;
         p_v[i] = tmpV ;
      }
   }

   return ;
}

static inline
void LagrangeElements()
{
  const Real_t deltatime = m_deltatime;

  CalcLagrangeElements(deltatime,m_vnew,m_vdov,m_dxx,m_dyy,m_dzz) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems() ;

  ApplyMaterialPropertiesForElems(m_matElemlist,m_vnew,m_v,m_e,m_delv,m_p,m_q,
         m_qq,m_ql,m_ss);

  UpdateVolumesForElems(m_vnew,m_v) ;
}

static inline
void CalcCourantConstraintForElems(Index_t p_matElemlist[T_NUMELEM],Real_t p_ss[T_NUMELEM],
         Real_t p_vdov[T_NUMELEM], Real_t p_arealg[T_NUMELEM])
{
   Index_t i;
   Real_t dtcourant = 1.0e+20 ;
   Index_t   courant_elem = -1 ;
   Real_t      qqc = m_qqc;
   Index_t length = m_numElem ;

   Real_t  qqc2 = 64.0 * qqc * qqc ;

//[FIXME] OpenMP pragma is temporarily disabled due to a bug.
//#pragma omp parallel for private(i) firstprivate(length,qqc2) shared(dtcourant,courant_elem, p_matElemlist, p_ss, p_vdov, p_arealg)
   for (i = 0 ; i < length ; ++i) {
      Index_t indx = p_matElemlist[i] ;

      Real_t dtf = p_ss[indx] * p_ss[indx] ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_vdov3:0.1)
#pragma aspen control probability(aspen_param_vdov3)
#else
#pragma aspen control probability(1)
#endif
      if ( p_vdov[indx] < 0. ) {

         dtf = dtf
            + qqc2 * p_arealg[indx] * p_arealg[indx]
            * p_vdov[indx] * p_vdov[indx] ;
      }

      dtf = SQRT8(dtf) ;

      dtf = p_arealg[indx] / dtf ;

   /* determine minimum timestep with its corresponding elem */
#pragma aspen control execute flops(2:traits(dp)) loads(1*aspen_param_double:from(p_vdov):traits(stride(1)))
      if (p_vdov[indx] != 0.) {
         if ( dtf < dtcourant ) {
//#pragma omp critical
            {
               dtcourant = dtf ;
               courant_elem = indx ;
            }
         }
      }
   }

   /* Don't try to register a time constraint if none of the elements
    * were active */
   if (courant_elem != -1) {
      m_dtcourant = dtcourant ;
   }

   return ;
}

static inline
void CalcHydroConstraintForElems(Index_t p_matElemlist[T_NUMELEM], Real_t p_vdov[T_NUMELEM])
{
   Index_t i;
   Real_t dthydro = 1.0e+20 ;
   Index_t hydro_elem = -1 ;
   Real_t dvovmax = m_dvovmax;
   Index_t length = m_numElem;


//[FIXME] OpenMP pragma is temporarily disabled due to a bug.
//#pragma omp parallel for private(i) firstprivate(length) shared(dthydro,hydro_elem, p_matElemlist, p_vdov)
   for (i = 0 ; i < length ; ++i) {
      Index_t indx = p_matElemlist[i] ;

#if ALLOW_ASPENIFSTMT == 1
#pragma aspen declare param(aspen_param_vdov:0.1)
#pragma aspen control probability(aspen_param_vdov)
#else
#pragma aspen control probability(1)
#endif
      if (p_vdov[indx] != 0.) {
         Real_t dtdvov = dvovmax / (FABS8(p_vdov[indx])+1.e-20) ;
#pragma aspen control ignore
         if ( dthydro > dtdvov ) {
//#pragma omp critical
            {
               dthydro = dtdvov ;
               hydro_elem = indx ;
            }
         }
      }
   }

   if (hydro_elem != -1) {
      m_dthydro = dthydro ;
   }

   return ;
}

static inline
void CalcTimeConstraintsForElems() {
#ifdef _OPENACC
#pragma acc update host(m_vdov, m_ss, m_arealg)
#endif
   /* evaluate time constraint */
   CalcCourantConstraintForElems(m_matElemlist,m_ss,m_vdov,m_arealg);

   /* check hydro constraint */
   CalcHydroConstraintForElems(m_matElemlist,m_vdov);
}

static inline
void LagrangeLeapFrog()
{
//#pragma acc wait
   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal();

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements();

   CalcTimeConstraintsForElems();

   // LagrangeRelease() ;  Creation/destruction of temps may be important to capture 
}

int main(int argc, char *argv[])
{
   Index_t plane, row, col, i, lnode, j;
   // Real_t ds = Real_t(1.125)/Real_t(edgeElems) ; /* may accumulate roundoff */
   Real_t tx, ty, tz ;
   Index_t nidx, zidx ;
   Index_t domElems ;
#if LULESH_STORE_OUTPUT
   FILE *fp;
#endif

#if LULESH_MEASURE_TIME
   double strt_time1, end_time1;
   double strt_time2, end_time2;

   strt_time1 = my_timer();
#endif
   edgeElems = T_EDGEELEM ;
   edgeNodes = edgeElems+1 ;

   /* get run options to measure various metrics */

   /* ... */

   /****************************/
   /*   Initialize Sedov Mesh  */
   /****************************/

   /* construct a uniform box for this processor */

   m_sizeX   = edgeElems ;
   m_sizeY   = edgeElems ;
   m_sizeZ   = edgeElems ;
   m_numElem = edgeElems*edgeElems*edgeElems ;
   m_numElem8 = m_numElem * 8 ;
   m_numNode = edgeNodes*edgeNodes*edgeNodes ;

   domElems = m_numElem;


   /* allocate field memory */

#if LULESH_PRINT_SIZE
       printf("T_NUMELEM\t%d\n", m_numElem);
       printf("T_NUMNODE\t%d\n", m_numNode);
       printf("T_NUMNODESETS\t%d\n", edgeNodes*edgeNodes);
#endif
#if LULESH_CHECK_SIZE
       if (m_numElem != T_NUMELEM) {
          printf("T_NUMELEM should be %d\n", m_numElem);
          exit(1);
       }
       if (m_numNode != T_NUMNODE) {
          printf("T_NUMNODE should be %d\n", m_numNode);
          exit(1);
       }
       if ((edgeNodes*edgeNodes) != T_NUMNODESETS) {
          printf("T_NUMNODESETS should be %d\n", (edgeNodes*edgeNodes));
          exit(1);
       }
#endif
   AllocateElemPersistent(m_numElem) ;
   AllocateElemTemporary (m_numElem) ;

   AllocateNodalPersistent(m_numNode) ;
   AllocateNodesets(edgeNodes*edgeNodes) ;
   AllocateTemporary(m_numElem8);
   AllocateTemporary2(m_numElem);

   /* initialize nodal coordinates */

   nidx = 0 ;
   tz  = 0. ;
   for (plane=0; plane<edgeNodes; ++plane) {
      ty = 0. ;
      for (row=0; row<edgeNodes; ++row) {
         tx = 0. ;
         for (col=0; col<edgeNodes; ++col) {
            m_x[nidx] = tx ;
            m_y[nidx] = ty ;
            m_z[nidx] = tz ;
            ++nidx ;
            // tx += ds ; /* may accumulate roundoff... */
            tx = 1.125*((Real_t)(col+1))/((Real_t)edgeElems) ;
         }
         // ty += ds ;  /* may accumulate roundoff... */
         ty = 1.125*((Real_t)(row+1))/((Real_t)edgeElems) ;
      }
      // tz += ds ;  /* may accumulate roundoff... */
      tz = 1.125*((Real_t)(plane+1))/((Real_t)edgeElems) ;
   }


   /* embed hexehedral elements in nodal point lattice */

   nidx = 0 ;
   zidx = 0 ;
   for (plane=0; plane<edgeElems; ++plane) {
      for (row=0; row<edgeElems; ++row) {
         for (col=0; col<edgeElems; ++col) {
            Index_t *localNode = &m_nodelist[8*zidx];
            localNode[0] = nidx                                       ;
            localNode[1] = nidx                                   + 1 ;
            localNode[2] = nidx                       + edgeNodes + 1 ;
            localNode[3] = nidx                       + edgeNodes     ;
            localNode[4] = nidx + edgeNodes*edgeNodes                 ;
            localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
            localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
            localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
            ++zidx ;
            ++nidx ;
         }
         ++nidx ;
      }
      nidx += edgeNodes ;
   }

   AllocateNodeElemIndexes() ;

   /* Create a material IndexSet (entire domain same material for now) */
   for (i=0; i<domElems; ++i) {
      m_matElemlist[i] = i ;
   }
   
   /* initialize material parameters */
   m_dtfixed = -1.0e-7 ;
   m_deltatime = 1.0e-7 ;
   m_deltatimemultlb = 1.1 ;
   m_deltatimemultub = 1.2 ;
   m_stoptime  = 1.0e-2 ;
   m_dtcourant = 1.0e+20 ;
   m_dthydro   = 1.0e+20 ;
   m_dtmax     = 1.0e-2 ;
   m_time    = 0. ;
   m_cycle   = 0 ;

   m_e_cut = 1.0e-7 ;
   m_p_cut = 1.0e-7 ;
   m_q_cut = 1.0e-7 ;
   m_u_cut = 1.0e-7 ;
   m_v_cut = 1.0e-10 ;

   m_hgcoef      = 3.0 ;
   m_ss4o3       = 4.0/3.0 ;

   m_qstop              =  1.0e+12 ;
   m_monoq_max_slope    =  1.0 ;
   m_monoq_limiter_mult =  2.0 ;
   m_qlc_monoq          = 0.5 ;
   m_qqc_monoq          = 2.0/3.0 ;
   m_qqc                = 2.0 ;

   m_pmin =  0. ;
   m_emin = -1.0e+15 ;

   m_dvovmax =  0.1 ;

   m_eosvmax =  1.0e+9 ;
   m_eosvmin =  1.0e-9 ;

   m_refdens =  (1.0) ;

   /* initialize field data */
   for (i=0; i<domElems; ++i) {
      Real_t x_local[8], y_local[8], z_local[8] ;
#pragma aspen declare data(elemToNode:traits(Array(8,aspen_param_int)))
      Index_t *elemToNode = &m_nodelist[8*i] ;
      Real_t volume;
      for( lnode=0 ; lnode<8 ; ++lnode )
      {
        Index_t gnode = elemToNode[lnode];
        x_local[lnode] = m_x[gnode];
        y_local[lnode] = m_y[gnode];
        z_local[lnode] = m_z[gnode];
      }

      // volume calculations
      volume = CalcElemVolume(x_local, y_local, z_local );
      m_volo[i] = volume ;
      m_elemMass[i] = volume ;
      for (j=0; j<8; ++j) {
         Index_t idx = elemToNode[j] ;
         m_nodalMass[idx] += volume / (8.0) ;
      }
   }

   /* deposit energy */
   m_e[0] = (3.948746e+7) ;

   /* set up symmetry nodesets */
   nidx = 0 ;
   for (i=0; i<edgeNodes; ++i) {
      Index_t planeInc = i*edgeNodes*edgeNodes ;
      Index_t rowInc   = i*edgeNodes ;
      for (j=0; j<edgeNodes; ++j) {
         m_symmX[nidx] = planeInc + j*edgeNodes ;
         m_symmY[nidx] = planeInc + j ;
         m_symmZ[nidx] = rowInc   + j ;
         ++nidx ;
      }
   }

   /* set up elemement connectivity information */
   m_lxim[0] = 0 ;
   for (i=1; i<domElems; ++i) {
      m_lxim[i]   = i-1 ;
      m_lxip[i-1] = i ;
   }
   m_lxip[domElems-1] = domElems-1 ;

   for (i=0; i<edgeElems; ++i) {
      m_letam[i] = i ; 
      m_letap[domElems-edgeElems+i] = domElems-edgeElems+i ;
   }
   for (i=edgeElems; i<domElems; ++i) {
      m_letam[i] = i-edgeElems ;
      m_letap[i-edgeElems] = i ;
   }

   for (i=0; i<edgeElems*edgeElems; ++i) {
      m_lzetam[i] = i ;
      m_lzetap[domElems-edgeElems*edgeElems+i] = domElems-edgeElems*edgeElems+i ;
   }
   for (i=edgeElems*edgeElems; i<domElems; ++i) {
      m_lzetam[i] = i - edgeElems*edgeElems ;
      m_lzetap[i-edgeElems*edgeElems] = i ;
   }

   /* set up boundary condition information */
   for (i=0; i<domElems; ++i) {
      m_elemBC[i] = 0 ;  /* clear BCs by default */
   }

   /* faces on "external" boundaries will be */
   /* symmetry plane or free surface BCs */
   for (i=0; i<edgeElems; ++i) {
      Index_t planeInc = i*edgeElems*edgeElems ;
      Index_t rowInc   = i*edgeElems ;
      for (j=0; j<edgeElems; ++j) {
         m_elemBC[planeInc+j*edgeElems] |= XI_M_SYMM ;
         m_elemBC[planeInc+j*edgeElems+edgeElems-1] |= XI_P_FREE ;
         m_elemBC[planeInc+j] |= ETA_M_SYMM ;
         m_elemBC[planeInc+j+edgeElems*edgeElems-edgeElems] |= ETA_P_FREE ;
         m_elemBC[rowInc+j] |= ZETA_M_SYMM ;
         m_elemBC[rowInc+j+domElems-edgeElems*edgeElems] |= ZETA_P_FREE ;
      }
   }
#if LULESH_MEASURE_TIME
   end_time1 = my_timer();
   strt_time2 = my_timer();
#endif
   i = 0;
   /* timestep to solution */
//[DEBUG] while-loop count = 593 for edgeElem == 10
//                         = 1041 for edgeElem == 20
//                         = 1248 for edgeElem == 30
//                         = 1420 for edgeElem == 40
//                         = 1566 for edgeElem == 50
// Use linear regression to estimate loop count using edgeElems.
#pragma aspen declare param(aspen_param_whilecnt:23.25*edgeElems+476.1)
#ifdef _OPENACCM
	acc_init(acc_device_default);
#endif

#pragma aspen modelregion
#ifdef _OPENACC
#pragma acc data create(m_fx[0:m_numNode], \
                        m_fy[0:m_numNode], \
                        m_fz[0:m_numNode], \
                        fx_elem[0:m_numElem8], \
                        fy_elem[0:m_numElem8], \
                        fz_elem[0:m_numElem8], \
                        dvdx[0:m_numElem8], \
                        dvdy[0:m_numElem8], \
                        dvdz[0:m_numElem8], \
                        x8n[0:m_numElem8], \
                        y8n[0:m_numElem8], \
                        z8n[0:m_numElem8], \
                        sigxx[0:m_numElem], \
                        sigyy[0:m_numElem], \
                        sigzz[0:m_numElem], \
                        determ[0:m_numElem], \
                        m_dxx[0:m_numElem], \
                        m_dyy[0:m_numElem], \
                        m_dzz[0:m_numElem], \
                        m_vnew[0:m_numElem], \
                        m_delx_xi[0:m_numElem], \
                        m_delx_eta[0:m_numElem], \
                        m_delx_zeta[0:m_numElem], \
                        m_delv_xi[0:m_numElem], \
                        m_delv_eta[0:m_numElem], \
                        m_delv_zeta[0:m_numElem], \
                        e_old[0:m_numElem],  \
                        delvc[0:m_numElem],  \
                        p_old[0:m_numElem],  \
                        q_old[0:m_numElem],  \
                        compression[0:m_numElem],  \
                        compHalfStep[0:m_numElem],  \
                        qq[0:m_numElem],  \
                        ql[0:m_numElem],  \
                        work[0:m_numElem],  \
                        p_new[0:m_numElem],  \
                        e_new[0:m_numElem],  \
                        q_new[0:m_numElem],  \
                        bvc[0:m_numElem],  \
                        pbvc[0:m_numElem], \
                        vnewc[0:m_numElem]) \
                 copy(m_x[0:m_numNode], \
                      m_y[0:m_numNode], \
                      m_z[0:m_numNode], \
                      m_xd[0:m_numNode], \
                      m_yd[0:m_numNode], \
                      m_zd[0:m_numNode], \
                      m_p[0:m_numElem], \
                      m_e[0:m_numElem]) \
                 create( m_arealg[0:m_numElem], \
                        m_delv[0:m_numElem], \
                        m_q[0:m_numElem], \
                        m_ql[0:m_numElem], \
                        m_qq[0:m_numElem], \
                        m_ss[0:m_numElem],       \
                        m_vdov[0:m_numElem] \
                       ) \
                 copyin(m_symmX[0:edgeNodes*edgeNodes], \
                        m_symmY[0:edgeNodes*edgeNodes], \
                        m_symmZ[0:edgeNodes*edgeNodes], \
                        m_xdd[0:m_numNode], \
                        m_ydd[0:m_numNode], \
                        m_zdd[0:m_numNode], \
                        m_v[0:m_numElem], \
                        m_volo[0:m_numElem], \
                        m_nodalMass[0:m_numNode], \
                        m_elemMass[0:m_numElem], \
                        m_lxim[0:m_numElem], \
                        m_lxip[0:m_numElem], \
                        m_letam[0:m_numElem], \
                        m_letap[0:m_numElem], \
                        m_lzetam[0:m_numElem], \
                        m_lzetap[0:m_numElem], \
                        m_nodelist[0:m_numElem8], \
                        m_nodeElemCount[0:m_numNode], \
                        m_nodeElemStart[0:m_numNode], \
                        m_nodeElemCornerList[0:m_nCorner], \
                        m_matElemlist[0:m_numElem], \
                        m_elemBC[m_numElem])
#endif
{
#pragma aspen control loop(aspen_param_whilecnt)
   while(m_time < m_stoptime ) {
      TimeIncrement() ;
      LagrangeLeapFrog() ;
      i++;
      /* problem->commNodes->Transfer(CommNodes::syncposvel) ; */
#if LULESH_SHOW_PROGRESS
      printf("time = %e, dt=%e\n",
             (double)m_time, (double)m_deltatime ) ;
#endif
   }
   printf("iterations: %d\n",i);
}
#ifdef _OPENACCM
	acc_shutdown(acc_device_default);
#endif
#if LULESH_MEASURE_TIME
   end_time2 = my_timer();
   printf ("Init time = %lf sec\n", end_time1 - strt_time1);
   printf ("Main Comp. time = %lf sec\n", end_time2 - strt_time2);
   printf ("Total elapsed time = %lf sec\n", (end_time1 - strt_time1) + (end_time2 - strt_time2));
#endif

#if LULESH_STORE_OUTPUT
   fp = fopen("lulesh.out", "w");
   for (i=0; i<m_numElem; i++) {
      fprintf(fp, "%.6f\n",m_x[i]);
   }
   for (i=0; i<m_numElem; i++) {
      fprintf(fp, "%.6f\n",m_y[i]);
   }
   for (i=0; i<m_numElem; i++) {
      fprintf(fp, "%.6f\n",m_z[i]);
   }
   fclose(fp);
#endif


   return 0 ;
}

