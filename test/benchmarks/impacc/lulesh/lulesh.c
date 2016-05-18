/*
  This is a Version 2.0 MPI + Open{ACC,MP} Beta implementation of LULESH

                 Copyright (c) 2010-2013.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 2.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
* Addition of regions to make work more representative of multi-material codes
* Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
  more representative of our actual working set sizes
* Single source distribution supports pure serial, pure OpenMP, MPI-only, 
  and MPI+OpenMP
* Addition of ability to visualize the mesh using VisIt 
  https://wci.llnl.gov/codes/visit/download.html
* Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
      printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
      printf(" -p              : Print out progress\n");
      printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
      printf(" -h              : This message\n");
      printf("\n\n");

*Notable changes in LULESH 2.0

* Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-util.cc - Non-timed functions
*
* The concept of "regions" was added, although every region is the same ideal gas material, and the same sedov blast wave problem is still the only problem its hardcoded to solve. Regions allow two things important to making this proxy app more representative:
* Four of the LULESH routines are now performed on a region-by-region basis, making the memory access patterns non-unit stride
* Artificial load imbalances can be easily introduced that could impact parallelization strategies.  
   * The load balance flag changes region assignment.  Region number is raised to the power entered for assignment probability.  Most likely regions changes with MPI process id.
   * The cost flag raises the cost of ~45% of the regions to evaluate EOS by the entered multiple.  The cost of 5% is 10x the entered multiple.
* MPI and OpenMP were added, and coalesced into a single version of the source that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow for numerous test cases without needing to recompile
* Performance optimizations and code cleanup uncovered during study of LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and output in support of using LULESH 2.0 for the 2017 CORAL procurement
*
* Possible Differences in Final Release (other changes possible)
*
* High Level mesh structure to allow data structure transformations
* Different default parameters
* Minor code performance changes and cleanup

TODO in future versions
* Add reader for (truly) unstructured meshes, probably serial only
* CMake based build system

//////////////

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

//#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
//#include <ctype.h>
#include <time.h>

#if !defined(_OPENACC) && defined(_OPENMP)
# include <omp.h>
#endif
#include <unistd.h>

#include "lulesh.h"

#define LULESH_DUMP_OUTPUT  0

#pragma openarc impacc ignoreglobal(stdout,__stdoutp)

/*********************************/
/* Data structure implementation */
/*********************************/

/* These structs are used to turn local, constant-sized arrays into scalars
   inside of accelerated regions. */
typedef struct
{
  Real_t v0;
  Real_t v1;
  Real_t v2;
  Real_t v3;
  Real_t v4;
  Real_t v5;
  Real_t v6;
  Real_t v7;
} val8;

typedef struct
{
  Real_t v0;
  Real_t v1;
  Real_t v2;
  Real_t v3;
  Real_t v4;
  Real_t v5;
} val6;

typedef struct
{
  // 3x8 matrix for loop unrolling
  Real_t v0_0; Real_t v0_1; Real_t v0_2; Real_t v0_3; Real_t v0_4; Real_t v0_5; Real_t v0_6; Real_t v0_7;
  Real_t v1_0; Real_t v1_1; Real_t v1_2; Real_t v1_3; Real_t v1_4; Real_t v1_5; Real_t v1_6; Real_t v1_7;
  Real_t v2_0; Real_t v2_1; Real_t v2_2; Real_t v2_3; Real_t v2_4; Real_t v2_5; Real_t v2_6; Real_t v2_7;
} bmat;

typedef struct
{
  // 8x4 matrix for loop unrolling
  Real_t v0_0; Real_t v0_1;Real_t v0_2;Real_t v0_3;
  Real_t v1_0; Real_t v1_1;Real_t v1_2;Real_t v1_3;
  Real_t v2_0; Real_t v2_1;Real_t v2_2;Real_t v2_3;
  Real_t v3_0; Real_t v3_1;Real_t v3_2;Real_t v3_3;
  Real_t v4_0; Real_t v4_1;Real_t v4_2;Real_t v4_3;
  Real_t v5_0; Real_t v5_1;Real_t v5_2;Real_t v5_3;
  Real_t v6_0; Real_t v6_1;Real_t v6_2;Real_t v6_3;
  Real_t v7_0; Real_t v7_1;Real_t v7_2;Real_t v7_3;
} hourmat;


#if USE_MPI   
  // Communication Work space 
  Real_t *commDataSend ;
  Real_t *commDataRecv ;

  // Maximum number of block neighbors 
  MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners 
  MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners 
#endif

  int m_numDevs;

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

  /* tmp arrays that are allocated globally for OpenACC */
  Real_t* m_fx_elem ;
  Real_t* m_fy_elem ;
  Real_t* m_fz_elem ;
  Real_t* m_dvdx ;
  Real_t* m_dvdy ;
  Real_t* m_dvdz ;
  Real_t* m_x8n ;
  Real_t* m_y8n ;
  Real_t* m_z8n ;
  Real_t* m_sigxx ;
  Real_t* m_sigyy ;
  Real_t* m_sigzz ;
  Real_t* m_determ ;
  Real_t* m_e_old ;
  Real_t* m_delvc ;
  Real_t* m_p_old ;
  Real_t* m_q_old ;
  Real_t* m_compression ;
  Real_t* m_compHalfStep ;
  Real_t* m_qq_old ;
  Real_t* m_ql_old ;
  Real_t* m_work ;
  Real_t* m_p_new ;
  Real_t* m_e_new ;
  Real_t* m_q_new ;
  Real_t* m_bvc ;
  Real_t* m_pbvc ;

  Real_t* m_nodalMass ;  /* mass */

  Index_t* m_symmX;  /* symmetry plane nodesets */
  Index_t* m_symmY;
  Index_t* m_symmZ;

  bool m_symmXempty;
  bool m_symmYempty;
  bool m_symmZempty;

  // Element-centered

  // Region information
  Int_t    m_numReg ;
  Int_t    m_cost; //imbalance cost
  Int_t   *m_regElemSize ;   // Size of region sets
  Index_t *m_regNumList ;    // Region number per domain element
  Index_t **m_regElemlist ;  // region indexset 

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

  // Cutoffs (treat as constants)
  Real_t  m_e_cut ;             // energy tolerance 
  Real_t  m_p_cut ;             // pressure tolerance 
  Real_t  m_q_cut ;             // q tolerance 
  Real_t  m_v_cut ;             // relative volume tolerance 
  Real_t  m_u_cut ;             // velocity tolerance 

  // Other constants (usually setable, but hardcoded in this proxy app)

  Real_t  m_hgcoef ;            // hourglass control 
  Real_t  m_ss4o3 ;
  Real_t  m_qstop ;             // excessive q indicator 
  Real_t  m_monoq_max_slope ;
  Real_t  m_monoq_limiter_mult ;
  Real_t  m_qlc_monoq ;         // linear term coef for q 
  Real_t  m_qqc_monoq ;         // quadratic term coef for q 
  Real_t  m_qqc ;
  Real_t  m_eosvmax ;
  Real_t  m_eosvmin ;
  Real_t  m_pmin ;              // pressure floor 
  Real_t  m_emin ;              // energy floor 
  Real_t  m_dvovmax ;           // maximum allowable volume change 
  Real_t  m_refdens ;           // reference density 

  // Variables to keep track of timestep, simulation time, and cycle
  Real_t  m_dtcourant ;         // courant constraint 
  Real_t  m_dthydro ;           // volume change constraint 
  Int_t   m_cycle ;             // iteration count for simulation 
  Real_t  m_dtfixed ;           // fixed time increment 
  Real_t  m_time ;              // current time 
  Real_t  m_deltatime ;         // variable time increment 
  Real_t  m_deltatimemultlb ;
  Real_t  m_deltatimemultub ;
  Real_t  m_dtmax ;             // maximum allowable time increment 
  Real_t  m_stoptime ;          // end time for simulation 


  Int_t   m_numRanks ;


  Index_t m_colLoc ;
  Index_t m_rowLoc ;
  Index_t m_planeLoc ;
  Index_t m_tp ;

  Index_t m_sizeX ;
  Index_t m_sizeY ;
  Index_t m_sizeZ ;
  Index_t m_numElem ;
  Index_t m_numNode ;

  Index_t m_maxPlaneSize ;
  Index_t m_maxEdgeSize ;

  // OMP hack 
  Index_t *m_nodeElemCount ;
  Index_t *m_nodeElemStart ;
  Index_t *m_nodeElemCornerList ;

  // Used in setup
  Index_t m_rowMin, m_rowMax;
  Index_t m_colMin, m_colMax;
  Index_t m_planeMin, m_planeMax ;

  // Accelerator device type
#ifdef _OPENACC
  acc_device_t m_devtype = acc_device_nvidia;
#endif

/******************************************/

/* Work Routines */

static inline
void TimeIncrement()
{
  Real_t targetdt = m_stoptime - m_time ;

  if ((m_dtfixed <= (Real_t)(0.0)) && (m_cycle != (Int_t)(0))) {
    Real_t ratio ;
    Real_t olddt = m_deltatime ;

    /* This will require a reduction in parallel */
    Real_t gnewdt = (Real_t)(1.0e+20) ;
    Real_t newdt ;
    if (m_dtcourant < gnewdt) {
      gnewdt = m_dtcourant / (Real_t)(2.0) ;
    }
    if (m_dthydro < gnewdt) {
      gnewdt = m_dthydro * (Real_t)(2.0) / (Real_t)(3.0) ;
    }

#if USE_MPI
    MPI_Allreduce(&gnewdt, &newdt, 1, ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE), MPI_MIN, MPI_COMM_WORLD) ;
#else
    newdt = gnewdt;
#endif

    ratio = newdt / olddt ;
    if (ratio >= (Real_t)(1.0)) {
      if (ratio < m_deltatimemultlb) {
        newdt = olddt ;
      }
      else if (ratio > m_deltatimemultub) {
        newdt = olddt*m_deltatimemultub ;
      }
    }

    if (newdt > m_dtmax) {
      newdt = m_dtmax ;
    }
    m_deltatime = newdt ;
  }

  /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
  if ((targetdt > m_deltatime) &&
      (targetdt < ((Real_t)(4.0) * m_deltatime / (Real_t)(3.0))) ) {
    targetdt = (Real_t)(2.0) * m_deltatime / (Real_t)(3.0) ;
  }

  if (targetdt < m_deltatime) {
    m_deltatime = targetdt ;
  }

  m_time += m_deltatime ;

  ++m_cycle ;
}

/******************************************/

static inline 
void InitStressTermsForElems(Real_t *p, Real_t *q,
                             Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                             Index_t numElem)
{
   //
   // pull in the stresses appropriate to the hydro integration
   //
   Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(p[numElem], q[numElem], \
                                  sigxx[numElem],sigyy[numElem],sigzz[numElem]) 
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numElem)
#endif
  for (i = 0 ; i < numElem ; ++i){
    sigxx[i] = sigyy[i] = sigzz[i] =  - p[i] - q[i] ;
  }
}

/******************************************/
#define CalcElemShapeFunctionDerivatives_unrolled(x,y,z,b,volume) \
do {\
  Real_t fjxxi, fjxet, fjxze;\
  Real_t fjyxi, fjyet, fjyze;\
  Real_t fjzxi, fjzet, fjzze;\
  Real_t cjxxi, cjxet, cjxze;\
  Real_t cjyxi, cjyet, cjyze;\
  Real_t cjzxi, cjzet, cjzze;\
\
  fjxxi = (Real_t)(.125) * ( (x.v6-x.v0) + (x.v5-x.v3) - (x.v7-x.v1) - (x.v4-x.v2) );\
  fjxet = (Real_t)(.125) * ( (x.v6-x.v0) - (x.v5-x.v3) + (x.v7-x.v1) - (x.v4-x.v2) );\
  fjxze = (Real_t)(.125) * ( (x.v6-x.v0) + (x.v5-x.v3) + (x.v7-x.v1) + (x.v4-x.v2) );\
\
  fjyxi = (Real_t)(.125) * ( (y.v6-y.v0) + (y.v5-y.v3) - (y.v7-y.v1) - (y.v4-y.v2) );\
  fjyet = (Real_t)(.125) * ( (y.v6-y.v0) - (y.v5-y.v3) + (y.v7-y.v1) - (y.v4-y.v2) );\
  fjyze = (Real_t)(.125) * ( (y.v6-y.v0) + (y.v5-y.v3) + (y.v7-y.v1) + (y.v4-y.v2) );\
\
  fjzxi = (Real_t)(.125) * ( (z.v6-z.v0) + (z.v5-z.v3) - (z.v7-z.v1) - (z.v4-z.v2) );\
  fjzet = (Real_t)(.125) * ( (z.v6-z.v0) - (z.v5-z.v3) + (z.v7-z.v1) - (z.v4-z.v2) );\
  fjzze = (Real_t)(.125) * ( (z.v6-z.v0) + (z.v5-z.v3) + (z.v7-z.v1) + (z.v4-z.v2) );\
\
  /* compute cofactors */\
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);\
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);\
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);\
\
  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);\
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);\
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);\
\
  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);\
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);\
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);\
\
  /* calculate partials :\
     this need only be done for l = 0,1,2,3   since , by symmetry ,\
     (6,7,4,5) = - (0,1,2,3) .\
  */\
  (b.v0_0) =   -  cjxxi  -  cjxet  -  cjxze;\
  (b.v0_1) =      cjxxi  -  cjxet  -  cjxze;\
  (b.v0_2) =      cjxxi  +  cjxet  -  cjxze;\
  (b.v0_3) =   -  cjxxi  +  cjxet  -  cjxze;\
  (b.v0_4) = -(b.v0_2);\
  (b.v0_5) = -(b.v0_3);\
  (b.v0_6) = -(b.v0_0);\
  (b.v0_7) = -(b.v0_1);\
\
  (b.v1_0) =   -  cjyxi  -  cjyet  -  cjyze;\
  (b.v1_1) =      cjyxi  -  cjyet  -  cjyze;\
  (b.v1_2) =      cjyxi  +  cjyet  -  cjyze;\
  (b.v1_3) =   -  cjyxi  +  cjyet  -  cjyze;\
  (b.v1_4) = -(b.v1_2);\
  (b.v1_5) = -(b.v1_3);\
  (b.v1_6) = -(b.v1_0);\
  (b.v1_7) = -(b.v1_1);\
\
  (b.v2_0) =   -  cjzxi  -  cjzet  -  cjzze;\
  (b.v2_1) =      cjzxi  -  cjzet  -  cjzze;\
  (b.v2_2) =      cjzxi  +  cjzet  -  cjzze;\
  (b.v2_3) =   -  cjzxi  +  cjzet  -  cjzze;\
  (b.v2_4) = -(b.v2_2);\
  (b.v2_5) = -(b.v2_3);\
  (b.v2_6) = -(b.v2_0);\
  (b.v2_7) = -(b.v2_1);\
\
  /* calculate jacobian determinant (volume) */\
  (volume) = (Real_t)(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);\
} while(0)\

/******************************************/

//static inline
#define SumElemFaceNormal(normalX0, normalY0, normalZ0,\
                          normalX1, normalY1, normalZ1,\
                          normalX2, normalY2, normalZ2,\
                          normalX3, normalY3, normalZ3,\
                          x0,  y0,  z0,\
                          x1,  y1,  z1,\
                          x2,  y2,  z2,\
                          x3,  y3,  z3)\
do {\
  Real_t bisectX0 = (Real_t)(0.5) * ((x3) + (x2) - (x1) - (x0));\
  Real_t bisectY0 = (Real_t)(0.5) * ((y3) + (y2) - (y1) - (y0));\
  Real_t bisectZ0 = (Real_t)(0.5) * ((z3) + (z2) - (z1) - (z0));\
  Real_t bisectX1 = (Real_t)(0.5) * ((x2) + (x1) - (x3) - (x0));\
  Real_t bisectY1 = (Real_t)(0.5) * ((y2) + (y1) - (y3) - (y0));\
  Real_t bisectZ1 = (Real_t)(0.5) * ((z2) + (z1) - (z3) - (z0));\
  Real_t areaX = (Real_t)(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);\
  Real_t areaY = (Real_t)(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);\
  Real_t areaZ = (Real_t)(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);\
\
  (normalX0) += areaX;\
  (normalX1) += areaX;\
  (normalX2) += areaX;\
  (normalX3) += areaX;\
\
  (normalY0) += areaY;\
  (normalY1) += areaY;\
  (normalY2) += areaY;\
  (normalY3) += areaY;\
\
  (normalZ0) += areaZ;\
  (normalZ1) += areaZ;\
  (normalZ2) += areaZ;\
  (normalZ3) += areaZ;\
} while(0)

/******************************************/
#define CalcElemNodeNormals_unrolled(B,x,y,z)\
do {\
  (B.v0_0) = (Real_t)(0.0);\
  (B.v1_0) = (Real_t)(0.0);\
  (B.v2_0) = (Real_t)(0.0);\
  (B.v0_1) = (Real_t)(0.0);\
  (B.v1_1) = (Real_t)(0.0);\
  (B.v2_1) = (Real_t)(0.0);\
  (B.v0_2) = (Real_t)(0.0);\
  (B.v1_2) = (Real_t)(0.0);\
  (B.v2_2) = (Real_t)(0.0);\
  (B.v0_3) = (Real_t)(0.0);\
  (B.v1_3) = (Real_t)(0.0);\
  (B.v2_3) = (Real_t)(0.0);\
  (B.v0_4) = (Real_t)(0.0);\
  (B.v1_4) = (Real_t)(0.0);\
  (B.v2_4) = (Real_t)(0.0);\
  (B.v0_5) = (Real_t)(0.0);\
  (B.v1_5) = (Real_t)(0.0);\
  (B.v2_5) = (Real_t)(0.0);\
  (B.v0_6) = (Real_t)(0.0);\
  (B.v1_6) = (Real_t)(0.0);\
  (B.v2_6) = (Real_t)(0.0);\
  (B.v0_7) = (Real_t)(0.0);\
  (B.v1_7) = (Real_t)(0.0);\
  (B.v2_7) = (Real_t)(0.0);\
  /* evaluate face one: nodes 0, 1, 2, 3 */\
  SumElemFaceNormal((B.v0_0), (B.v1_0), (B.v2_0),\
                    (B.v0_1), (B.v1_1), (B.v2_1),\
                    (B.v0_2), (B.v1_2), (B.v2_2),\
                    (B.v0_3), (B.v1_3), (B.v2_3),\
                    (x.v0), (y.v0), (z.v0), (x.v1), (y.v1), (z.v1),\
                    (x.v2), (y.v2), (z.v2), (x.v3), (y.v3), (z.v3));\
  /* evaluate face two: nodes 0, 4, 5, 1 */\
  SumElemFaceNormal((B.v0_0), (B.v1_0), (B.v2_0),\
                    (B.v0_4), (B.v1_4), (B.v2_4),\
                    (B.v0_5), (B.v1_5), (B.v2_5),\
                    (B.v0_1), (B.v1_1), (B.v2_1),\
                    (x.v0), (y.v0), (z.v0), (x.v4), (y.v4), (z.v4),\
                    (x.v5), (y.v5), (z.v5), (x.v1), (y.v1), (z.v1));\
  /* evaluate face three: nodes 1, 5, 6, 2 */\
  SumElemFaceNormal((B.v0_1), (B.v1_1), (B.v2_1),\
                    (B.v0_5), (B.v1_5), (B.v2_5),\
                    (B.v0_6), (B.v1_6), (B.v2_6),\
                    (B.v0_2), (B.v1_2), (B.v2_2),\
                    (x.v1), (y.v1), (z.v1), (x.v5), (y.v5), (z.v5),\
                    (x.v6), (y.v6), (z.v6), (x.v2), (y.v2), (z.v2));\
  /* evaluate face four: nodes 2, 6, 7, 3 */\
  SumElemFaceNormal((B.v0_2), (B.v1_2), (B.v2_2),\
                    (B.v0_6), (B.v1_6), (B.v2_6),\
                    (B.v0_7), (B.v1_7), (B.v2_7),\
                    (B.v0_3), (B.v1_3), (B.v2_3),\
                    (x.v2), (y.v2), (z.v2), (x.v6), (y.v6), (z.v6),\
                    (x.v7), (y.v7), (z.v7), (x.v3), (y.v3), (z.v3));\
  /* evaluate face five: nodes 3, 7, 4, 0 */\
  SumElemFaceNormal((B.v0_3), (B.v1_3), (B.v2_3),\
                    (B.v0_7), (B.v1_7), (B.v2_7),\
                    (B.v0_4), (B.v1_4), (B.v2_4),\
                    (B.v0_0), (B.v1_0), (B.v2_0),\
                    (x.v3), (y.v3), (z.v3), (x.v7), (y.v7), (z.v7),\
                    (x.v4), (y.v4), (z.v4), (x.v0), (y.v0), (z.v0));\
  /* evaluate face six: nodes 4, 7, 6, 5 */\
  SumElemFaceNormal((B.v0_4), (B.v1_4), (B.v2_4),\
                    (B.v0_7), (B.v1_7), (B.v2_7),\
                    (B.v0_6), (B.v1_6), (B.v2_6),\
                    (B.v0_5), (B.v1_5), (B.v2_5),\
                    (x.v4), (y.v4), (z.v4), (x.v7), (y.v7), (z.v7),\
                    (x.v6), (y.v6), (z.v6), (x.v5), (y.v5), (z.v5));\
} while(0)

/******************************************/
static inline
void IntegrateStressForElems( Index_t *nodelist,
                              Real_t *x,  Real_t *y,  Real_t *z,
                              Real_t *fx, Real_t *fy, Real_t *fz,
                              Real_t *fx_elem, Real_t *fy_elem, Real_t *fz_elem,
                              Index_t *nodeElemCount,
                              Index_t *nodeElemStart,
                              Index_t *nodeElemCornerList,
                              Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                              Real_t *determ, Index_t numElem, Index_t numNode)
{
  volatile Index_t numElem8 = numElem * 8 ;
  Index_t k;

  // loop over all elements
#ifdef _OPENACC
#pragma acc parallel loop present(x[numNode],         \
                                  y[numNode],         \
                                  z[numNode],         \
                                  determ[numElem],    \
                                  nodelist[numElem8], \
                                  sigxx[numElem],     \
                                  sigyy[numElem],     \
                                  sigzz[numElem],     \
                                  fx_elem[numElem8],  \
                                  fy_elem[numElem8],  \
                                  fz_elem[numElem8])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numElem)
#endif
  for(k = 0; k < numElem; ++k )
  {
    const Index_t *elemToNode = &(nodelist[8*k]);
    bmat B; // shape function derivatives
    val8 x_local;
    val8 y_local;
    val8 z_local;
    Index_t gnode;

    // get nodal coordinates from global arrays and copy into local arrays.
    // Loop unrolled because the PGI OpenACC implementation currently stores
    // locally-defined arrays in a global, shared context. Thus we have to use
    // scalars instead to get them in registers.
    gnode = elemToNode[0];
    x_local.v0 = x[gnode];
    y_local.v0 = y[gnode];
    z_local.v0 = z[gnode];
    gnode = elemToNode[1];
    x_local.v1 = x[gnode];
    y_local.v1 = y[gnode];
    z_local.v1 = z[gnode];
    gnode = elemToNode[2];
    x_local.v2 = x[gnode];
    y_local.v2 = y[gnode];
    z_local.v2 = z[gnode];
    gnode = elemToNode[3];
    x_local.v3 = x[gnode];
    y_local.v3 = y[gnode];
    z_local.v3 = z[gnode];
    gnode = elemToNode[4];
    x_local.v4 = x[gnode];
    y_local.v4 = y[gnode];
    z_local.v4 = z[gnode];
    gnode = elemToNode[5];
    x_local.v5 = x[gnode];
    y_local.v5 = y[gnode];
    z_local.v5 = z[gnode];
    gnode = elemToNode[6];
    x_local.v6 = x[gnode];
    y_local.v6 = y[gnode];
    z_local.v6 = z[gnode];
    gnode = elemToNode[7];
    x_local.v7 = x[gnode];
    y_local.v7 = y[gnode];
    z_local.v7 = z[gnode];

    // Volume calculation involves extra work for numerical consistency
    CalcElemShapeFunctionDerivatives_unrolled(x_local, y_local, z_local, B, determ[k]);
    CalcElemNodeNormals_unrolled( B, x_local, y_local, z_local );

    // Eliminate thread writing conflicts at the nodes by giving
    // each element its own copy to write to
    // NOTE: This is a manually inlined macro. Moving it back into macro form
    //       requires some more pointer arithmetic which causes the current
    //       PGI compiler to segfault during compilation (version 13.6-accel).
    fx_elem[k*8 + 0] = -( sigxx[k] * B.v0_0 );
    fy_elem[k*8 + 0] = -( sigyy[k] * B.v1_0 );
    fz_elem[k*8 + 0] = -( sigzz[k] * B.v2_0 );
    fx_elem[k*8 + 1] = -( sigxx[k] * B.v0_1 );
    fy_elem[k*8 + 1] = -( sigyy[k] * B.v1_1 );
    fz_elem[k*8 + 1] = -( sigzz[k] * B.v2_1 );
    fx_elem[k*8 + 2] = -( sigxx[k] * B.v0_2 );
    fy_elem[k*8 + 2] = -( sigyy[k] * B.v1_2 );
    fz_elem[k*8 + 2] = -( sigzz[k] * B.v2_2 );
    fx_elem[k*8 + 3] = -( sigxx[k] * B.v0_3 );
    fy_elem[k*8 + 3] = -( sigyy[k] * B.v1_3 );
    fz_elem[k*8 + 3] = -( sigzz[k] * B.v2_3 );
    fx_elem[k*8 + 4] = -( sigxx[k] * B.v0_4 );
    fy_elem[k*8 + 4] = -( sigyy[k] * B.v1_4 );
    fz_elem[k*8 + 4] = -( sigzz[k] * B.v2_4 );
    fx_elem[k*8 + 5] = -( sigxx[k] * B.v0_5 );
    fy_elem[k*8 + 5] = -( sigyy[k] * B.v1_5 );
    fz_elem[k*8 + 5] = -( sigzz[k] * B.v2_5 );
    fx_elem[k*8 + 6] = -( sigxx[k] * B.v0_6 );
    fy_elem[k*8 + 6] = -( sigyy[k] * B.v1_6 );
    fz_elem[k*8 + 6] = -( sigzz[k] * B.v2_6 );
    fx_elem[k*8 + 7] = -( sigxx[k] * B.v0_7 );
    fy_elem[k*8 + 7] = -( sigyy[k] * B.v1_7 );
    fz_elem[k*8 + 7] = -( sigzz[k] * B.v2_7 );
  }

  // If threaded, then we need to copy the data out of the temporary
  // arrays used above into the final forces field

  /* volatile because otherwise it will be optimized out of the pragma and
     break things. */
  volatile Index_t nCorner = nodeElemStart[numNode-1] 
                             + nodeElemCount[numNode-1];
  Index_t gnode;
#ifdef _OPENACC
#pragma acc kernels loop independent vector(256) \
                          present(fx_elem[numElem8], \
                                  fy_elem[numElem8], \
                                  fz_elem[numElem8], \
                                  nodelist[numElem8],\
                                  fx[numElem],       \
                                  fy[numElem],       \
                                  fz[numElem],       \
                                  nodeElemCount[numNode], \
                                  nodeElemCornerList[nCorner], \
                                  nodeElemStart[numNode])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numNode)
#endif
  for( gnode=0 ; gnode<numNode ; ++gnode )
  {
    Index_t count = nodeElemCount[gnode] ;
    Index_t start = nodeElemStart[gnode] ;
    Real_t fx_tmp = (Real_t)(0.0) ;
    Real_t fy_tmp = (Real_t)(0.0) ;
    Real_t fz_tmp = (Real_t)(0.0) ;
    Index_t i;
    for (i=0 ; i < count ; ++i) {
      Index_t elem = nodeElemCornerList[start+i] ;
      fx_tmp += fx_elem[elem] ;
      fy_tmp += fy_elem[elem] ;
      fz_tmp += fz_elem[elem] ;
    }
    fx[gnode] = fx_tmp ;
    fy[gnode] = fy_tmp ;
    fz[gnode] = fz_tmp ;

  }
}

/******************************************/

//static inline
#define CollectDomainNodesToElemNodes(x, y, z, \
                                      elemToNode, \
                                      elemX, elemY, elemZ) \
do { \
  Index_t nd0i = (elemToNode)[0] ; \
  Index_t nd1i = (elemToNode)[1] ; \
  Index_t nd2i = (elemToNode)[2] ; \
  Index_t nd3i = (elemToNode)[3] ; \
  Index_t nd4i = (elemToNode)[4] ; \
  Index_t nd5i = (elemToNode)[5] ; \
  Index_t nd6i = (elemToNode)[6] ; \
  Index_t nd7i = (elemToNode)[7] ; \
 \
  (elemX).v0 = (x)[nd0i]; \
  (elemX).v1 = (x)[nd1i]; \
  (elemX).v2 = (x)[nd2i]; \
  (elemX).v3 = (x)[nd3i]; \
  (elemX).v4 = (x)[nd4i]; \
  (elemX).v5 = (x)[nd5i]; \
  (elemX).v6 = (x)[nd6i]; \
  (elemX).v7 = (x)[nd7i]; \
 \
  (elemY).v0 = (y)[nd0i]; \
  (elemY).v1 = (y)[nd1i]; \
  (elemY).v2 = (y)[nd2i]; \
  (elemY).v3 = (y)[nd3i]; \
  (elemY).v4 = (y)[nd4i]; \
  (elemY).v5 = (y)[nd5i]; \
  (elemY).v6 = (y)[nd6i]; \
  (elemY).v7 = (y)[nd7i]; \
 \
  (elemZ).v0 = (z)[nd0i]; \
  (elemZ).v1 = (z)[nd1i]; \
  (elemZ).v2 = (z)[nd2i]; \
  (elemZ).v3 = (z)[nd3i]; \
  (elemZ).v4 = (z)[nd4i]; \
  (elemZ).v5 = (z)[nd5i]; \
  (elemZ).v6 = (z)[nd6i]; \
  (elemZ).v7 = (z)[nd7i]; \
} while(0)

/******************************************/

//static inline
#define VoluDer(x0,  x1,  x2, \
                x3,  x4,  x5, \
                y0,  y1,  y2, \
                y3,  y4,  y5, \
                z0,  z1,  z2, \
                z3,  z4,  z5, \
                dvdx, dvdy, dvdz) \
do { \
  const Real_t twelfth = (Real_t)(1.0) / (Real_t)(12.0) ; \
 \
  (dvdx) = \
    ((y1) + (y2)) * ((z0) + (z1)) - ((y0) + (y1)) * ((z1) + (z2)) + \
    ((y0) + (y4)) * ((z3) + (z4)) - ((y3) + (y4)) * ((z0) + (z4)) - \
    ((y2) + (y5)) * ((z3) + (z5)) + ((y3) + (y5)) * ((z2) + (z5)); \
  (dvdy) = \
    - ((x1) + (x2)) * ((z0) + (z1)) + ((x0) + (x1)) * ((z1) + (z2)) - \
    ((x0) + (x4)) * ((z3) + (z4)) + ((x3) + (x4)) * ((z0) + (z4)) + \
    ((x2) + (x5)) * ((z3) + (z5)) - ((x3) + (x5)) * ((z2) + (z5)); \
 \
  (dvdz) = \
    - ((y1) + (y2)) * ((x0) + (x1)) + ((y0) + (y1)) * ((x1) + (x2)) - \
    ((y0) + (y4)) * ((x3) + (x4)) + ((y3) + (y4)) * ((x0) + (x4)) + \
    ((y2) + (y5)) * ((x3) + (x5)) - ((y3) + (y5)) * ((x2) + (x5)); \
 \
  (dvdx) *= twelfth; \
  (dvdy) *= twelfth; \
  (dvdz) *= twelfth; \
} while(0)

/******************************************/

///static inline
#define CalcElemVolumeDerivative(dvdx, dvdy, dvdz, \
                                 x, y, z) \
do { \
  VoluDer(x.v1, x.v2, x.v3, x.v4, x.v5, x.v7, \
          y.v1, y.v2, y.v3, y.v4, y.v5, y.v7, \
          z.v1, z.v2, z.v3, z.v4, z.v5, z.v7, \
          dvdx.v0, dvdy.v0, dvdz.v0); \
  VoluDer(x.v0, x.v1, x.v2, x.v7, x.v4, x.v6, \
          y.v0, y.v1, y.v2, y.v7, y.v4, y.v6, \
          z.v0, z.v1, z.v2, z.v7, z.v4, z.v6, \
          dvdx.v3, dvdy.v3, dvdz.v3); \
  VoluDer(x.v3, x.v0, x.v1, x.v6, x.v7, x.v5, \
          y.v3, y.v0, y.v1, y.v6, y.v7, y.v5, \
          z.v3, z.v0, z.v1, z.v6, z.v7, z.v5, \
          dvdx.v2, dvdy.v2, dvdz.v2); \
  VoluDer(x.v2, x.v3, x.v0, x.v5, x.v6, x.v4, \
          y.v2, y.v3, y.v0, y.v5, y.v6, y.v4, \
          z.v2, z.v3, z.v0, z.v5, z.v6, z.v4, \
          dvdx.v1, dvdy.v1, dvdz.v1); \
  VoluDer(x.v7, x.v6, x.v5, x.v0, x.v3, x.v1, \
          y.v7, y.v6, y.v5, y.v0, y.v3, y.v1, \
          z.v7, z.v6, z.v5, z.v0, z.v3, z.v1, \
          dvdx.v4, dvdy.v4, dvdz.v4); \
  VoluDer(x.v4, x.v7, x.v6, x.v1, x.v0, x.v2, \
          y.v4, y.v7, y.v6, y.v1, y.v0, y.v2, \
          z.v4, z.v7, z.v6, z.v1, z.v0, z.v2, \
          dvdx.v5, dvdy.v5, dvdz.v5); \
  VoluDer(x.v5, x.v4, x.v7, x.v2, x.v1, x.v3, \
          y.v5, y.v4, y.v7, y.v2, y.v1, y.v3, \
          z.v5, z.v4, z.v7, z.v2, z.v1, z.v3, \
          dvdx.v6, dvdy.v6, dvdz.v6); \
  VoluDer(x.v6, x.v5, x.v4, x.v3, x.v2, x.v0, \
          y.v6, y.v5, y.v4, y.v3, y.v2, y.v0, \
          z.v6, z.v5, z.v4, z.v3, z.v2, z.v0, \
          dvdx.v7, dvdy.v7, dvdz.v7); \
} while(0)

/******************************************/

//static inline
#define CalcElemFBHourglassForce(xd, yd, zd,  \
                                 hourgam, coefficient, \
                                 hgfx, hgfy, hgfz) \
do { \
  val8 hxx; \
  hxx.v0 = hourgam.v0_0 * xd.v0 + hourgam.v1_0 * xd.v1 + \
           hourgam.v2_0 * xd.v2 + hourgam.v3_0 * xd.v3 + \
           hourgam.v4_0 * xd.v4 + hourgam.v5_0 * xd.v5 + \
           hourgam.v6_0 * xd.v6 + hourgam.v7_0 * xd.v7; \
  hxx.v1 = hourgam.v0_1 * xd.v0 + hourgam.v1_1 * xd.v1 + \
           hourgam.v2_1 * xd.v2 + hourgam.v3_1 * xd.v3 + \
           hourgam.v4_1 * xd.v4 + hourgam.v5_1 * xd.v5 + \
           hourgam.v6_1 * xd.v6 + hourgam.v7_1 * xd.v7; \
  hxx.v2 = hourgam.v0_2 * xd.v0 + hourgam.v1_2 * xd.v1 + \
           hourgam.v2_2 * xd.v2 + hourgam.v3_2 * xd.v3 + \
           hourgam.v4_2 * xd.v4 + hourgam.v5_2 * xd.v5 + \
           hourgam.v6_2 * xd.v6 + hourgam.v7_2 * xd.v7; \
  hxx.v3 = hourgam.v0_3 * xd.v0 + hourgam.v1_3 * xd.v1 + \
           hourgam.v2_3 * xd.v2 + hourgam.v3_3 * xd.v3 + \
           hourgam.v4_3 * xd.v4 + hourgam.v5_3 * xd.v5 + \
           hourgam.v6_3 * xd.v6 + hourgam.v7_3 * xd.v7; \
 \
  hgfx.v0 = coefficient * \
    (hourgam.v0_0 * hxx.v0 + hourgam.v0_1 * hxx.v1 + \
     hourgam.v0_2 * hxx.v2 + hourgam.v0_3 * hxx.v3); \
  hgfx.v1 = coefficient * \
    (hourgam.v1_0 * hxx.v0 + hourgam.v1_1 * hxx.v1 + \
     hourgam.v1_2 * hxx.v2 + hourgam.v1_3 * hxx.v3); \
  hgfx.v2 = coefficient * \
    (hourgam.v2_0 * hxx.v0 + hourgam.v2_1 * hxx.v1 + \
     hourgam.v2_2 * hxx.v2 + hourgam.v2_3 * hxx.v3); \
  hgfx.v3 = coefficient * \
    (hourgam.v3_0 * hxx.v0 + hourgam.v3_1 * hxx.v1 + \
     hourgam.v3_2 * hxx.v2 + hourgam.v3_3 * hxx.v3); \
  hgfx.v4 = coefficient * \
    (hourgam.v4_0 * hxx.v0 + hourgam.v4_1 * hxx.v1 + \
     hourgam.v4_2 * hxx.v2 + hourgam.v4_3 * hxx.v3); \
  hgfx.v5 = coefficient * \
    (hourgam.v5_0 * hxx.v0 + hourgam.v5_1 * hxx.v1 + \
     hourgam.v5_2 * hxx.v2 + hourgam.v5_3 * hxx.v3); \
  hgfx.v6 = coefficient * \
    (hourgam.v6_0 * hxx.v0 + hourgam.v6_1 * hxx.v1 + \
     hourgam.v6_2 * hxx.v2 + hourgam.v6_3 * hxx.v3); \
  hgfx.v7 = coefficient * \
    (hourgam.v7_0 * hxx.v0 + hourgam.v7_1 * hxx.v1 + \
     hourgam.v7_2 * hxx.v2 + hourgam.v7_3 * hxx.v3); \
 \
  hxx.v0 = hourgam.v0_0 * yd.v0 + hourgam.v1_0 * yd.v1 + \
           hourgam.v2_0 * yd.v2 + hourgam.v3_0 * yd.v3 + \
           hourgam.v4_0 * yd.v4 + hourgam.v5_0 * yd.v5 + \
           hourgam.v6_0 * yd.v6 + hourgam.v7_0 * yd.v7; \
  hxx.v1 = hourgam.v0_1 * yd.v0 + hourgam.v1_1 * yd.v1 + \
           hourgam.v2_1 * yd.v2 + hourgam.v3_1 * yd.v3 + \
           hourgam.v4_1 * yd.v4 + hourgam.v5_1 * yd.v5 + \
           hourgam.v6_1 * yd.v6 + hourgam.v7_1 * yd.v7; \
  hxx.v2 = hourgam.v0_2 * yd.v0 + hourgam.v1_2 * yd.v1 + \
           hourgam.v2_2 * yd.v2 + hourgam.v3_2 * yd.v3 + \
           hourgam.v4_2 * yd.v4 + hourgam.v5_2 * yd.v5 + \
           hourgam.v6_2 * yd.v6 + hourgam.v7_2 * yd.v7; \
  hxx.v3 = hourgam.v0_3 * yd.v0 + hourgam.v1_3 * yd.v1 + \
           hourgam.v2_3 * yd.v2 + hourgam.v3_3 * yd.v3 + \
           hourgam.v4_3 * yd.v4 + hourgam.v5_3 * yd.v5 + \
           hourgam.v6_3 * yd.v6 + hourgam.v7_3 * yd.v7; \
 \
  hgfy.v0 = coefficient * \
    (hourgam.v0_0 * hxx.v0 + hourgam.v0_1 * hxx.v1 + \
     hourgam.v0_2 * hxx.v2 + hourgam.v0_3 * hxx.v3); \
  hgfy.v1 = coefficient * \
    (hourgam.v1_0 * hxx.v0 + hourgam.v1_1 * hxx.v1 + \
     hourgam.v1_2 * hxx.v2 + hourgam.v1_3 * hxx.v3); \
  hgfy.v2 = coefficient * \
    (hourgam.v2_0 * hxx.v0 + hourgam.v2_1 * hxx.v1 + \
     hourgam.v2_2 * hxx.v2 + hourgam.v2_3 * hxx.v3); \
  hgfy.v3 = coefficient * \
    (hourgam.v3_0 * hxx.v0 + hourgam.v3_1 * hxx.v1 + \
     hourgam.v3_2 * hxx.v2 + hourgam.v3_3 * hxx.v3); \
  hgfy.v4 = coefficient * \
    (hourgam.v4_0 * hxx.v0 + hourgam.v4_1 * hxx.v1 + \
     hourgam.v4_2 * hxx.v2 + hourgam.v4_3 * hxx.v3); \
  hgfy.v5 = coefficient * \
    (hourgam.v5_0 * hxx.v0 + hourgam.v5_1 * hxx.v1 + \
     hourgam.v5_2 * hxx.v2 + hourgam.v5_3 * hxx.v3); \
  hgfy.v6 = coefficient * \
    (hourgam.v6_0 * hxx.v0 + hourgam.v6_1 * hxx.v1 + \
     hourgam.v6_2 * hxx.v2 + hourgam.v6_3 * hxx.v3); \
  hgfy.v7 = coefficient * \
    (hourgam.v7_0 * hxx.v0 + hourgam.v7_1 * hxx.v1 + \
     hourgam.v7_2 * hxx.v2 + hourgam.v7_3 * hxx.v3); \
 \
  hxx.v0 = hourgam.v0_0 * zd.v0 + hourgam.v1_0 * zd.v1 + \
           hourgam.v2_0 * zd.v2 + hourgam.v3_0 * zd.v3 + \
           hourgam.v4_0 * zd.v4 + hourgam.v5_0 * zd.v5 + \
           hourgam.v6_0 * zd.v6 + hourgam.v7_0 * zd.v7; \
  hxx.v1 = hourgam.v0_1 * zd.v0 + hourgam.v1_1 * zd.v1 + \
           hourgam.v2_1 * zd.v2 + hourgam.v3_1 * zd.v3 + \
           hourgam.v4_1 * zd.v4 + hourgam.v5_1 * zd.v5 + \
           hourgam.v6_1 * zd.v6 + hourgam.v7_1 * zd.v7; \
  hxx.v2 = hourgam.v0_2 * zd.v0 + hourgam.v1_2 * zd.v1 + \
           hourgam.v2_2 * zd.v2 + hourgam.v3_2 * zd.v3 + \
           hourgam.v4_2 * zd.v4 + hourgam.v5_2 * zd.v5 + \
           hourgam.v6_2 * zd.v6 + hourgam.v7_2 * zd.v7; \
  hxx.v3 = hourgam.v0_3 * zd.v0 + hourgam.v1_3 * zd.v1 + \
           hourgam.v2_3 * zd.v2 + hourgam.v3_3 * zd.v3 + \
           hourgam.v4_3 * zd.v4 + hourgam.v5_3 * zd.v5 + \
           hourgam.v6_3 * zd.v6 + hourgam.v7_3 * zd.v7; \
 \
  hgfz.v0 = coefficient * \
    (hourgam.v0_0 * hxx.v0 + hourgam.v0_1 * hxx.v1 + \
     hourgam.v0_2 * hxx.v2 + hourgam.v0_3 * hxx.v3); \
  hgfz.v1 = coefficient * \
    (hourgam.v1_0 * hxx.v0 + hourgam.v1_1 * hxx.v1 + \
     hourgam.v1_2 * hxx.v2 + hourgam.v1_3 * hxx.v3); \
  hgfz.v2 = coefficient * \
    (hourgam.v2_0 * hxx.v0 + hourgam.v2_1 * hxx.v1 + \
     hourgam.v2_2 * hxx.v2 + hourgam.v2_3 * hxx.v3); \
  hgfz.v3 = coefficient * \
    (hourgam.v3_0 * hxx.v0 + hourgam.v3_1 * hxx.v1 + \
     hourgam.v3_2 * hxx.v2 + hourgam.v3_3 * hxx.v3); \
  hgfz.v4 = coefficient * \
    (hourgam.v4_0 * hxx.v0 + hourgam.v4_1 * hxx.v1 + \
     hourgam.v4_2 * hxx.v2 + hourgam.v4_3 * hxx.v3); \
  hgfz.v5 = coefficient * \
    (hourgam.v5_0 * hxx.v0 + hourgam.v5_1 * hxx.v1 + \
     hourgam.v5_2 * hxx.v2 + hourgam.v5_3 * hxx.v3); \
  hgfz.v6 = coefficient * \
    (hourgam.v6_0 * hxx.v0 + hourgam.v6_1 * hxx.v1 + \
     hourgam.v6_2 * hxx.v2 + hourgam.v6_3 * hxx.v3); \
  hgfz.v7 = coefficient * \
    (hourgam.v7_0 * hxx.v0 + hourgam.v7_1 * hxx.v1 + \
     hourgam.v7_2 * hxx.v2 + hourgam.v7_3 * hxx.v3); \
} while(0)

/******************************************/
#define FillHourGam \
do { \
      /* i = 0 */ \
      Real_t hourmodx = \
        x8n[i3] * gamma[0][0] + x8n[i3+1] * gamma[0][1] + \
        x8n[i3+2] * gamma[0][2] + x8n[i3+3] * gamma[0][3] + \
        x8n[i3+4] * gamma[0][4] + x8n[i3+5] * gamma[0][5] + \
        x8n[i3+6] * gamma[0][6] + x8n[i3+7] * gamma[0][7]; \
 \
      Real_t hourmody = \
        y8n[i3] * gamma[0][0] + y8n[i3+1] * gamma[0][1] + \
        y8n[i3+2] * gamma[0][2] + y8n[i3+3] * gamma[0][3] + \
        y8n[i3+4] * gamma[0][4] + y8n[i3+5] * gamma[0][5] + \
        y8n[i3+6] * gamma[0][6] + y8n[i3+7] * gamma[0][7]; \
 \
      Real_t hourmodz = \
        z8n[i3] * gamma[0][0] + z8n[i3+1] * gamma[0][1] + \
        z8n[i3+2] * gamma[0][2] + z8n[i3+3] * gamma[0][3] + \
        z8n[i3+4] * gamma[0][4] + z8n[i3+5] * gamma[0][5] + \
        z8n[i3+6] * gamma[0][6] + z8n[i3+7] * gamma[0][7]; \
 \
      hourgam.v0_0 = gamma[0][0] -  volinv*(dvdx[i3  ] * hourmodx + \
                                               dvdy[i3  ] * hourmody + \
                                               dvdz[i3  ] * hourmodz ); \
 \
      hourgam.v1_0 = gamma[0][1] -  volinv*(dvdx[i3+1] * hourmodx + \
                                               dvdy[i3+1] * hourmody + \
                                               dvdz[i3+1] * hourmodz ); \
 \
      hourgam.v2_0 = gamma[0][2] -  volinv*(dvdx[i3+2] * hourmodx + \
                                               dvdy[i3+2] * hourmody + \
                                               dvdz[i3+2] * hourmodz ); \
 \
      hourgam.v3_0 = gamma[0][3] -  volinv*(dvdx[i3+3] * hourmodx + \
                                               dvdy[i3+3] * hourmody + \
                                               dvdz[i3+3] * hourmodz ); \
 \
      hourgam.v4_0 = gamma[0][4] -  volinv*(dvdx[i3+4] * hourmodx + \
                                               dvdy[i3+4] * hourmody + \
                                               dvdz[i3+4] * hourmodz ); \
 \
      hourgam.v5_0 = gamma[0][5] -  volinv*(dvdx[i3+5] * hourmodx + \
                                               dvdy[i3+5] * hourmody + \
                                               dvdz[i3+5] * hourmodz ); \
 \
      hourgam.v6_0 = gamma[0][6] -  volinv*(dvdx[i3+6] * hourmodx + \
                                               dvdy[i3+6] * hourmody + \
                                               dvdz[i3+6] * hourmodz ); \
 \
      hourgam.v7_0 = gamma[0][7] -  volinv*(dvdx[i3+7] * hourmodx + \
                                               dvdy[i3+7] * hourmody + \
                                               dvdz[i3+7] * hourmodz ); \
      /* i = 1 */ \
      hourmodx = \
        x8n[i3] * gamma[1][0] + x8n[i3+1] * gamma[1][1] + \
        x8n[i3+2] * gamma[1][2] + x8n[i3+3] * gamma[1][3] + \
        x8n[i3+4] * gamma[1][4] + x8n[i3+5] * gamma[1][5] + \
        x8n[i3+6] * gamma[1][6] + x8n[i3+7] * gamma[1][7]; \
 \
      hourmody = \
        y8n[i3] * gamma[1][0] + y8n[i3+1] * gamma[1][1] + \
        y8n[i3+2] * gamma[1][2] + y8n[i3+3] * gamma[1][3] + \
        y8n[i3+4] * gamma[1][4] + y8n[i3+5] * gamma[1][5] + \
        y8n[i3+6] * gamma[1][6] + y8n[i3+7] * gamma[1][7]; \
 \
      hourmodz = \
        z8n[i3] * gamma[1][0] + z8n[i3+1] * gamma[1][1] + \
        z8n[i3+2] * gamma[1][2] + z8n[i3+3] * gamma[1][3] + \
        z8n[i3+4] * gamma[1][4] + z8n[i3+5] * gamma[1][5] + \
        z8n[i3+6] * gamma[1][6] + z8n[i3+7] * gamma[1][7]; \
 \
      hourgam.v0_1 = gamma[1][0] -  volinv*(dvdx[i3  ] * hourmodx + \
                                               dvdy[i3  ] * hourmody + \
                                               dvdz[i3  ] * hourmodz ); \
 \
      hourgam.v1_1 = gamma[1][1] -  volinv*(dvdx[i3+1] * hourmodx + \
                                               dvdy[i3+1] * hourmody + \
                                               dvdz[i3+1] * hourmodz ); \
 \
      hourgam.v2_1 = gamma[1][2] -  volinv*(dvdx[i3+2] * hourmodx + \
                                               dvdy[i3+2] * hourmody + \
                                               dvdz[i3+2] * hourmodz ); \
 \
      hourgam.v3_1 = gamma[1][3] -  volinv*(dvdx[i3+3] * hourmodx + \
                                               dvdy[i3+3] * hourmody + \
                                               dvdz[i3+3] * hourmodz ); \
 \
      hourgam.v4_1 = gamma[1][4] -  volinv*(dvdx[i3+4] * hourmodx + \
                                               dvdy[i3+4] * hourmody + \
                                               dvdz[i3+4] * hourmodz ); \
 \
      hourgam.v5_1 = gamma[1][5] -  volinv*(dvdx[i3+5] * hourmodx + \
                                               dvdy[i3+5] * hourmody + \
                                               dvdz[i3+5] * hourmodz ); \
 \
      hourgam.v6_1 = gamma[1][6] -  volinv*(dvdx[i3+6] * hourmodx + \
                                               dvdy[i3+6] * hourmody + \
                                               dvdz[i3+6] * hourmodz ); \
 \
      hourgam.v7_1 = gamma[1][7] -  volinv*(dvdx[i3+7] * hourmodx + \
                                               dvdy[i3+7] * hourmody + \
                                               dvdz[i3+7] * hourmodz ); \
      /* i = 2 */ \
      hourmodx = \
        x8n[i3] * gamma[2][0] + x8n[i3+1] * gamma[2][1] + \
        x8n[i3+2] * gamma[2][2] + x8n[i3+3] * gamma[2][3] + \
        x8n[i3+4] * gamma[2][4] + x8n[i3+5] * gamma[2][5] + \
        x8n[i3+6] * gamma[2][6] + x8n[i3+7] * gamma[2][7]; \
 \
      hourmody = \
        y8n[i3] * gamma[2][0] + y8n[i3+1] * gamma[2][1] + \
        y8n[i3+2] * gamma[2][2] + y8n[i3+3] * gamma[2][3] + \
        y8n[i3+4] * gamma[2][4] + y8n[i3+5] * gamma[2][5] + \
        y8n[i3+6] * gamma[2][6] + y8n[i3+7] * gamma[2][7]; \
 \
      hourmodz = \
        z8n[i3] * gamma[2][0] + z8n[i3+1] * gamma[2][1] + \
        z8n[i3+2] * gamma[2][2] + z8n[i3+3] * gamma[2][3] + \
        z8n[i3+4] * gamma[2][4] + z8n[i3+5] * gamma[2][5] + \
        z8n[i3+6] * gamma[2][6] + z8n[i3+7] * gamma[2][7]; \
 \
      hourgam.v0_2 = gamma[2][0] -  volinv*(dvdx[i3  ] * hourmodx + \
                                               dvdy[i3  ] * hourmody + \
                                               dvdz[i3  ] * hourmodz ); \
 \
      hourgam.v1_2 = gamma[2][1] -  volinv*(dvdx[i3+1] * hourmodx + \
                                               dvdy[i3+1] * hourmody + \
                                               dvdz[i3+1] * hourmodz ); \
 \
      hourgam.v2_2 = gamma[2][2] -  volinv*(dvdx[i3+2] * hourmodx + \
                                               dvdy[i3+2] * hourmody + \
                                               dvdz[i3+2] * hourmodz ); \
 \
      hourgam.v3_2 = gamma[2][3] -  volinv*(dvdx[i3+3] * hourmodx + \
                                               dvdy[i3+3] * hourmody + \
                                               dvdz[i3+3] * hourmodz ); \
 \
      hourgam.v4_2 = gamma[2][4] -  volinv*(dvdx[i3+4] * hourmodx + \
                                               dvdy[i3+4] * hourmody + \
                                               dvdz[i3+4] * hourmodz ); \
 \
      hourgam.v5_2 = gamma[2][5] -  volinv*(dvdx[i3+5] * hourmodx + \
                                               dvdy[i3+5] * hourmody + \
                                               dvdz[i3+5] * hourmodz ); \
 \
      hourgam.v6_2 = gamma[2][6] -  volinv*(dvdx[i3+6] * hourmodx + \
                                               dvdy[i3+6] * hourmody + \
                                               dvdz[i3+6] * hourmodz ); \
 \
      hourgam.v7_2 = gamma[2][7] -  volinv*(dvdx[i3+7] * hourmodx + \
                                               dvdy[i3+7] * hourmody + \
                                               dvdz[i3+7] * hourmodz ); \
      /* i = 3 */ \
      hourmodx = \
        x8n[i3] * gamma[3][0] + x8n[i3+1] * gamma[3][1] + \
        x8n[i3+2] * gamma[3][2] + x8n[i3+3] * gamma[3][3] + \
        x8n[i3+4] * gamma[3][4] + x8n[i3+5] * gamma[3][5] + \
        x8n[i3+6] * gamma[3][6] + x8n[i3+7] * gamma[3][7]; \
 \
      hourmody = \
        y8n[i3] * gamma[3][0] + y8n[i3+1] * gamma[3][1] + \
        y8n[i3+2] * gamma[3][2] + y8n[i3+3] * gamma[3][3] + \
        y8n[i3+4] * gamma[3][4] + y8n[i3+5] * gamma[3][5] + \
        y8n[i3+6] * gamma[3][6] + y8n[i3+7] * gamma[3][7]; \
 \
      hourmodz = \
        z8n[i3] * gamma[3][0] + z8n[i3+1] * gamma[3][1] + \
        z8n[i3+2] * gamma[3][2] + z8n[i3+3] * gamma[3][3] + \
        z8n[i3+4] * gamma[3][4] + z8n[i3+5] * gamma[3][5] + \
        z8n[i3+6] * gamma[3][6] + z8n[i3+7] * gamma[3][7]; \
 \
      hourgam.v0_3 = gamma[3][0] -  volinv*(dvdx[i3  ] * hourmodx + \
                                               dvdy[i3  ] * hourmody + \
                                               dvdz[i3  ] * hourmodz ); \
 \
      hourgam.v1_3 = gamma[3][1] -  volinv*(dvdx[i3+1] * hourmodx + \
                                               dvdy[i3+1] * hourmody + \
                                               dvdz[i3+1] * hourmodz ); \
 \
      hourgam.v2_3 = gamma[3][2] -  volinv*(dvdx[i3+2] * hourmodx + \
                                               dvdy[i3+2] * hourmody + \
                                               dvdz[i3+2] * hourmodz ); \
 \
      hourgam.v3_3 = gamma[3][3] -  volinv*(dvdx[i3+3] * hourmodx + \
                                               dvdy[i3+3] * hourmody + \
                                               dvdz[i3+3] * hourmodz ); \
 \
      hourgam.v4_3 = gamma[3][4] -  volinv*(dvdx[i3+4] * hourmodx + \
                                               dvdy[i3+4] * hourmody + \
                                               dvdz[i3+4] * hourmodz ); \
 \
      hourgam.v5_3 = gamma[3][5] -  volinv*(dvdx[i3+5] * hourmodx + \
                                               dvdy[i3+5] * hourmody + \
                                               dvdz[i3+5] * hourmodz ); \
 \
      hourgam.v6_3 = gamma[3][6] -  volinv*(dvdx[i3+6] * hourmodx + \
                                               dvdy[i3+6] * hourmody + \
                                               dvdz[i3+6] * hourmodz ); \
 \
      hourgam.v7_3 = gamma[3][7] -  volinv*(dvdx[i3+7] * hourmodx + \
                                               dvdy[i3+7] * hourmody + \
                                               dvdz[i3+7] * hourmodz ); \
} while(0)

static inline
void CalcFBHourglassForceForElems( 
                                   Index_t *nodelist,
                                   Index_t *nodeElemCount,
                                   Index_t *nodeElemStart,
                                   Index_t *nodeElemCornerList,
                                   Real_t *determ,
                                   Real_t *fx, Real_t *fy, Real_t *fz,
                                   Real_t *x8n, Real_t *y8n, Real_t *z8n,
                                   Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                   Real_t hourg, Index_t numElem, Index_t numNode)
{
#if !defined(_OPENACC) && defined(_OPENMP)
  Index_t numthreads = omp_get_max_threads();
#else
  Index_t numthreads = 1;
#endif
  /*************************************************
   *
   *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
   *               force.
   *
   *************************************************/
  
  Index_t numElem8 = numElem * 8 ;

  Real_t *ss = m_ss;
  Real_t *elemMass = m_elemMass;
  Real_t *xd = m_xd;
  Real_t *yd = m_yd;
  Real_t *zd = m_zd;

  Real_t *fx_elem = m_fx_elem;
  Real_t *fy_elem = m_fy_elem;
  Real_t *fz_elem = m_fz_elem;

  Real_t  gamma[4][8];

  gamma[0][0] = (Real_t)( 1.);
  gamma[0][1] = (Real_t)( 1.);
  gamma[0][2] = (Real_t)(-1.);
  gamma[0][3] = (Real_t)(-1.);
  gamma[0][4] = (Real_t)(-1.);
  gamma[0][5] = (Real_t)(-1.);
  gamma[0][6] = (Real_t)( 1.);
  gamma[0][7] = (Real_t)( 1.);
  gamma[1][0] = (Real_t)( 1.);
  gamma[1][1] = (Real_t)(-1.);
  gamma[1][2] = (Real_t)(-1.);
  gamma[1][3] = (Real_t)( 1.);
  gamma[1][4] = (Real_t)(-1.);
  gamma[1][5] = (Real_t)( 1.);
  gamma[1][6] = (Real_t)( 1.);
  gamma[1][7] = (Real_t)(-1.);
  gamma[2][0] = (Real_t)( 1.);
  gamma[2][1] = (Real_t)(-1.);
  gamma[2][2] = (Real_t)( 1.);
  gamma[2][3] = (Real_t)(-1.);
  gamma[2][4] = (Real_t)( 1.);
  gamma[2][5] = (Real_t)(-1.);
  gamma[2][6] = (Real_t)( 1.);
  gamma[2][7] = (Real_t)(-1.);
  gamma[3][0] = (Real_t)(-1.);
  gamma[3][1] = (Real_t)( 1.);
  gamma[3][2] = (Real_t)(-1.);
  gamma[3][3] = (Real_t)( 1.);
  gamma[3][4] = (Real_t)( 1.);
  gamma[3][5] = (Real_t)(-1.);
  gamma[3][6] = (Real_t)( 1.);
  gamma[3][7] = (Real_t)(-1.);

  Index_t i2;

/*************************************************/
/*    compute the hourglass modes */
#ifdef _OPENACC
#pragma acc kernels copyin(gamma[4][8])         \
                    present(fx_elem[numElem8], \
                            fy_elem[numElem8], \
                            fz_elem[numElem8], \
                            xd[numNode],       \
                            yd[numNode],       \
                            zd[numNode],       \
                            dvdx[numElem8],    \
                            dvdy[numElem8],    \
                            dvdz[numElem8],    \
                            x8n[numElem8],     \
                            y8n[numElem8],     \
                            z8n[numElem8],     \
                            nodelist[numElem8],\
                            determ[numElem],   \
                            ss[numElem],       \
                            elemMass[numElem])
#pragma acc cache(gamma)
#pragma acc loop independent
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numElem, hourg)
#endif
  for(i2=0;i2<numElem;++i2){
    val8 hgfx, hgfy, hgfz;

    Real_t coefficient;

    //Real_t hourgam[8][4];
    hourmat hourgam;
    val8 xd1, yd1, zd1;

    const Index_t *elemToNode = &nodelist[i2*8];
    Index_t i3=8*i2;
    Real_t volinv=(Real_t)(1.0)/determ[i2];
    Real_t ss1, mass1, volume13 ;

    /* Large macro of unrolled loop */
    FillHourGam;

    /* compute forces */
    /* store forces into h arrays (force arrays) */

    ss1 = ss[i2];
    mass1 = elemMass[i2];
    
    volume13 = pow(determ[i2], (1.0 / 3.0));

    Index_t n0si2 = elemToNode[0];
    Index_t n1si2 = elemToNode[1];
    Index_t n2si2 = elemToNode[2];
    Index_t n3si2 = elemToNode[3];
    Index_t n4si2 = elemToNode[4];
    Index_t n5si2 = elemToNode[5];
    Index_t n6si2 = elemToNode[6];
    Index_t n7si2 = elemToNode[7];

    xd1.v0 = xd[n0si2];
    xd1.v1 = xd[n1si2];
    xd1.v2 = xd[n2si2];
    xd1.v3 = xd[n3si2];
    xd1.v4 = xd[n4si2];
    xd1.v5 = xd[n5si2];
    xd1.v6 = xd[n6si2];
    xd1.v7 = xd[n7si2];

    yd1.v0 = yd[n0si2];
    yd1.v1 = yd[n1si2];
    yd1.v2 = yd[n2si2];
    yd1.v3 = yd[n3si2];
    yd1.v4 = yd[n4si2];
    yd1.v5 = yd[n5si2];
    yd1.v6 = yd[n6si2];
    yd1.v7 = yd[n7si2];

    zd1.v0 = zd[n0si2];
    zd1.v1 = zd[n1si2];
    zd1.v2 = zd[n2si2];
    zd1.v3 = zd[n3si2];
    zd1.v4 = zd[n4si2];
    zd1.v5 = zd[n5si2];
    zd1.v6 = zd[n6si2];
    zd1.v7 = zd[n7si2];

    coefficient = - hourg * (Real_t)(0.01) * ss1 * mass1 / volume13;

    CalcElemFBHourglassForce(xd1,yd1,zd1, hourgam, coefficient, 
                             hgfx, hgfy, hgfz);

    // With the threaded version, we write into local arrays per elem
    // so we don't have to worry about race conditions
    fx_elem[i3 + 0] = hgfx.v0;
    fx_elem[i3 + 1] = hgfx.v1;
    fx_elem[i3 + 2] = hgfx.v2;
    fx_elem[i3 + 3] = hgfx.v3;
    fx_elem[i3 + 4] = hgfx.v4;
    fx_elem[i3 + 5] = hgfx.v5;
    fx_elem[i3 + 6] = hgfx.v6;
    fx_elem[i3 + 7] = hgfx.v7;

    fy_elem[i3 + 0] = hgfy.v0;
    fy_elem[i3 + 1] = hgfy.v1;
    fy_elem[i3 + 2] = hgfy.v2;
    fy_elem[i3 + 3] = hgfy.v3;
    fy_elem[i3 + 4] = hgfy.v4;
    fy_elem[i3 + 5] = hgfy.v5;
    fy_elem[i3 + 6] = hgfy.v6;
    fy_elem[i3 + 7] = hgfy.v7;

    fz_elem[i3 + 0] = hgfz.v0;
    fz_elem[i3 + 1] = hgfz.v1;
    fz_elem[i3 + 2] = hgfz.v2;
    fz_elem[i3 + 3] = hgfz.v3;
    fz_elem[i3 + 4] = hgfz.v4;
    fz_elem[i3 + 5] = hgfz.v5;
    fz_elem[i3 + 6] = hgfz.v6;
    fz_elem[i3 + 7] = hgfz.v7;

  } // end accelerated for

  /* volatile because otherwise it will be optimized out of the pragma and
     break things. */
  volatile Index_t nCorner = nodeElemStart[numNode-1] 
                             + nodeElemCount[numNode-1];

  Index_t gnode;

  // Collect the data from the local arrays into the final force arrays
#ifdef _OPENACC
#pragma acc kernels loop independent vector(256) \
                          present(nodeElemCount[numNode],      \
                                  nodeElemStart[numNode],      \
                                  nodeElemCornerList[nCorner], \
                                  fx_elem[numElem8],           \
                                  fy_elem[numElem8],           \
                                  fz_elem[numElem8],           \
                                  fx[numNode],                 \
                                  fy[numNode],                 \
                                  fz[numNode])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numNode)
#endif
  for( gnode=0 ; gnode<numNode ; ++gnode )
  {
    Index_t count = nodeElemCount[gnode] ;
    Index_t start = nodeElemStart[gnode] ;
    Real_t fx_tmp = (Real_t)(0.0) ;
    Real_t fy_tmp = (Real_t)(0.0) ;
    Real_t fz_tmp = (Real_t)(0.0) ;
    Index_t i;
    for (i=0 ; i < count ; ++i) {
      Index_t elem = nodeElemCornerList[start+i] ;
      fx_tmp += fx_elem[elem] ;
      fy_tmp += fy_elem[elem] ;
      fz_tmp += fz_elem[elem] ;
    }
    fx[gnode] += fx_tmp ;
    fy[gnode] += fy_tmp ;
    fz[gnode] += fz_tmp ;
  }

}

/******************************************/

#define LoadTmpStorageFBControl(dvdx, dvdy, dvdz, \
                                pfx, pfy, pfz, \
                                x8n, y8n, z8n, \
                                x1, y1, z1, \
                                i) \
do { \
    Index_t jj; \
    jj = 8*(i)+0; \
    (dvdx)[jj] = (pfx).v0; \
    (dvdy)[jj] = (pfy).v0; \
    (dvdz)[jj] = (pfz).v0; \
    (x8n)[jj]  = (x1).v0; \
    (y8n)[jj]  = (y1).v0; \
    (z8n)[jj]  = (z1).v0; \
    jj = 8*(i)+1; \
    (dvdx)[jj] = (pfx).v1; \
    (dvdy)[jj] = (pfy).v1; \
    (dvdz)[jj] = (pfz).v1; \
    (x8n)[jj]  = (x1).v1; \
    (y8n)[jj]  = (y1).v1; \
    (z8n)[jj]  = (z1).v1; \
    jj = 8*(i)+2; \
    (dvdx)[jj] = (pfx).v2; \
    (dvdy)[jj] = (pfy).v2; \
    (dvdz)[jj] = (pfz).v2; \
    (x8n)[jj]  = (x1).v2; \
    (y8n)[jj]  = (y1).v2; \
    (z8n)[jj]  = (z1).v2; \
    jj = 8*(i)+3; \
    (dvdx)[jj] = (pfx).v3; \
    (dvdy)[jj] = (pfy).v3; \
    (dvdz)[jj] = (pfz).v3; \
    (x8n)[jj]  = (x1).v3; \
    (y8n)[jj]  = (y1).v3; \
    (z8n)[jj]  = (z1).v3; \
    jj = 8*(i)+4; \
    (dvdx)[jj] = (pfx).v4; \
    (dvdy)[jj] = (pfy).v4; \
    (dvdz)[jj] = (pfz).v4; \
    (x8n)[jj]  = (x1).v4; \
    (y8n)[jj]  = (y1).v4; \
    (z8n)[jj]  = (z1).v4; \
    jj = 8*(i)+5; \
    (dvdx)[jj] = (pfx).v5; \
    (dvdy)[jj] = (pfy).v5; \
    (dvdz)[jj] = (pfz).v5; \
    (x8n)[jj]  = (x1).v5; \
    (y8n)[jj]  = (y1).v5; \
    (z8n)[jj]  = (z1).v5; \
    jj = 8*(i)+6; \
    (dvdx)[jj] = (pfx).v6; \
    (dvdy)[jj] = (pfy).v6; \
    (dvdz)[jj] = (pfz).v6; \
    (x8n)[jj]  = (x1).v6; \
    (y8n)[jj]  = (y1).v6; \
    (z8n)[jj]  = (z1).v6; \
    jj = 8*(i)+7; \
    (dvdx)[jj] = (pfx).v7; \
    (dvdy)[jj] = (pfy).v7; \
    (dvdz)[jj] = (pfz).v7; \
    (x8n)[jj]  = (x1).v7; \
    (y8n)[jj]  = (y1).v7; \
    (z8n)[jj]  = (z1).v7; \
} while(0) \

static inline
void CalcHourglassControlForElems(
                                  Real_t *x, Real_t *y, Real_t *z,
                                  Real_t *fx, Real_t *fy, Real_t *fz,
                                  Real_t determ[], Real_t hgcoef,
                                  Index_t *nodelist,
                                  Index_t *nodeElemCount,
                                  Index_t *nodeElemStart,
                                  Index_t *nodeElemCornerList)
{
  Index_t numElem = m_numElem ;
  volatile Index_t numElem8 = numElem * 8 ;
  volatile Index_t numNode = m_numNode;
  Real_t *dvdx = m_dvdx;
  Real_t *dvdy = m_dvdy;
  Real_t *dvdz = m_dvdz;
  Real_t *x8n  = m_x8n;
  Real_t *y8n  = m_y8n;
  Real_t *z8n  = m_z8n;
  Real_t *volo = m_volo;
  Real_t *v = m_v;

  int abort = 0;
  /* start loop over elements */
  Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(dvdx[numElem8],     \
                                  dvdy[numElem8],     \
                                  dvdz[numElem8],     \
                                  x8n[numElem8],      \
                                  y8n[numElem8],      \
                                  z8n[numElem8],      \
                                  x[numNode],         \
                                  y[numNode],         \
                                  z[numNode],         \
                                  volo[numElem],      \
                                  v[numElem],         \
                                  determ[numElem],    \
                                  nodelist[numElem8]) \
                          reduction(max: abort)
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numElem) reduction(max: abort)
#endif
  for (i=0 ; i<numElem ; ++i){
    val8 x1;
    val8 y1;
    val8 z1;
    val8 pfx;
    val8 pfy;
    val8 pfz;

    Index_t* elemToNode = &nodelist[i*8];
    CollectDomainNodesToElemNodes(x, y, z,
                                  elemToNode, x1, y1, z1);

    CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

    /* load into temporary storage for FB Hour Glass control */
    LoadTmpStorageFBControl(dvdx, dvdy, dvdz,
                            pfx, pfy, pfz,
                            x8n, y8n, z8n,
                            x1, y1, z1,
                            i);

    determ[i] = volo[i] * v[i];

    /* Do a check for negative volumes */
    if ( v[i] <= (Real_t)(0.0) ) {
      abort = 1;
    }
  } // end for

  if(abort) {
#if USE_MPI         
    MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
    exit(VolumeError);
#endif
  }

  if ( hgcoef > (Real_t)(0.) ) {
    CalcFBHourglassForceForElems(nodelist, nodeElemCount,
                                 nodeElemStart, nodeElemCornerList,
                                 determ, fx, fy, fz,
                                 x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                 hgcoef, numElem, numNode );
  }

  return ;
}

/******************************************/

static inline
void CalcVolumeForceForElems(Real_t *fx, Real_t *fy, Real_t *fz)
{
  Index_t numElem = m_numElem ;
  Index_t numNode = m_numNode;
  if (numElem != 0) {
    Real_t  hgcoef = m_hgcoef ;
    Real_t *sigxx  = m_sigxx;
    Real_t *sigyy  = m_sigyy;
    Real_t *sigzz  = m_sigzz;
    Real_t *determ = m_determ;
    Real_t *p = m_p;
    Real_t *q = m_q;
    Real_t *x = m_x;
    Real_t *y = m_y;
    Real_t *z = m_z;
    Index_t *nodelist = m_nodelist;
    Index_t *nodeElemCount = m_nodeElemCount;
    Index_t *nodeElemStart = m_nodeElemStart;
    Index_t *nodeElemCornerList = m_nodeElemCornerList;

    /* Sum contributions to total stress tensor */
    InitStressTermsForElems(p, q, sigxx, sigyy, sigzz, numElem);

    // call elemlib stress integration loop to produce nodal forces from
    // material stresses.
    IntegrateStressForElems( nodelist,
                             x, y, z,
                             fx, fy, fz,
                             m_fx_elem, m_fy_elem, m_fz_elem, 
                             nodeElemCount,
                             nodeElemStart,
                             nodeElemCornerList,
                             sigxx, sigyy, sigzz, determ, numElem,
                             numNode);
    int abort = 0;
    Index_t k;
#ifdef _OPENACC
#pragma acc parallel loop present(determ[numElem]) \
                          reduction(max: abort)
#elif defined(_OPENMP)
#pragma omp parallel for  reduction(max:abort) firstprivate(numElem)  
#endif
    for(k = 0; k < numElem; ++k) {
      if(determ[k] <= (Real_t)(0.0)) {
        abort = 1;
      }
    }

    if(abort == 1) {
#if USE_MPI            
      MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
      exit(VolumeError);
#endif
    }


    CalcHourglassControlForElems(x, y, z, fx, fy, fz, determ, hgcoef, 
                                 nodelist, nodeElemCount, nodeElemStart, 
                                 nodeElemCornerList);
  }
}

/******************************************/

static inline void CalcForceForNodes()
{
  Index_t numNode = m_numNode ;

#if USE_MPI  
  CommRecv(MSG_COMM_SBN, 3,
           m_sizeX + 1, m_sizeY + 1, m_sizeZ + 1,
           true, false) ;
#endif  

  Real_t *fx = m_fx;
  Real_t *fy = m_fy;
  Real_t *fz = m_fz;

  Index_t i;

#ifdef _OPENACC
#pragma acc parallel loop present(fx[numNode], \
                                  fy[numNode], \
                                  fz[numNode])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numNode)
#endif
  for (i=0; i<numNode; ++i) {
    fx[i] = (Real_t)(0.0);
    fy[i] = (Real_t)(0.0);
    fz[i] = (Real_t)(0.0);
  }

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems(fx, fy, fz) ;

#if USE_MPI  
  Real_t *fieldData[3] ;

#pragma acc data present(fx[numNode], \
                         fy[numNode], \
                         fz[numNode])
  {
#pragma acc update host(fx[numNode], \
                        fy[numNode], \
                        fz[numNode])
    
    fieldData[0] = fx;
    fieldData[1] = fy;
    fieldData[2] = fz;
    CommSend(MSG_COMM_SBN, 3, fieldData,
             m_sizeX + 1, m_sizeY + 1, m_sizeZ +  1,
             true, false) ;
    CommSBN(3, fieldData) ;
#pragma acc update device(fx[numNode], \
                          fy[numNode], \
                          fz[numNode]) \
                   async
  } // end acc data
#endif  
}

/******************************************/

static inline
void CalcAccelerationForNodes(Real_t *xdd, Real_t *ydd, Real_t *zdd,
                              Real_t *fx, Real_t *fy, Real_t *fz,
                              Real_t *nodalMass, Index_t numNode)
{
  Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(fx[numNode], \
                                  fy[numNode], \
                                  fz[numNode], \
                                  xdd[numNode], \
                                  ydd[numNode], \
                                  zdd[numNode], \
                                  nodalMass[numNode])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numNode)
#endif
  for (i = 0; i < numNode; ++i) {
    xdd[i] = fx[i] / nodalMass[i];
    ydd[i] = fy[i] / nodalMass[i];
    zdd[i] = fz[i] / nodalMass[i];
  }
}

/******************************************/

static inline
void ApplyAccelerationBoundaryConditionsForNodes(
                                                 Real_t *xdd, Real_t *ydd, Real_t *zdd)
{
  volatile Index_t numNode = m_numNode;
  volatile Index_t size = m_sizeX;
  Index_t numNodeBC = (size+1)*(size+1) ;
  Index_t *symmX = m_symmX;
  Index_t *symmY = m_symmY;
  Index_t *symmZ = m_symmZ;

  /* replace conditional loops with altered end conditions. This allows to do
     the equivalent of a nowait on the device too. */
  Index_t endX = m_symmXempty ? 0 : numNodeBC;
  Index_t endY = m_symmYempty ? 0 : numNodeBC;
  Index_t endZ = m_symmZempty ? 0 : numNodeBC;

  if (!endX && !endY && !endZ) return; /* OPENARC BUG */

  Index_t i;

#ifdef _OPENACC
#pragma acc parallel firstprivate(numNodeBC) \
                     present(xdd[numNode], \
                             ydd[numNode], \
                             zdd[numNode], \
                             symmX[numNodeBC], \
                             symmY[numNodeBC], \
                             symmZ[numNodeBC])
#elif defined(_OPENMP)
#pragma omp parallel firstprivate(numNodeBC)
#endif
  {

#ifdef _OPENACC
#pragma acc loop 
#elif defined(_OPENMP)
#pragma omp for nowait
#endif
    for(i=0 ; i<endX ; ++i) {
      xdd[symmX[i]] = (Real_t)(0.0) ;
    }

#ifdef _OPENACC
#pragma acc loop 
#elif defined(_OPENMP)
#pragma omp for nowait
#endif
    for(i=0 ; i<endY ; ++i) {
      ydd[symmY[i]] = (Real_t)(0.0) ;
    }

#ifdef _OPENACC
#pragma acc loop 
#elif defined(_OPENMP)
#pragma omp for nowait
#endif
    for(i=0 ; i<endZ ; ++i) {
      zdd[symmZ[i]] = (Real_t)(0.0) ;
    }

  } // end parallel region

}

/******************************************/

static inline
void CalcVelocityForNodes(Real_t *xd,  Real_t *yd,  Real_t *zd,
                          Real_t *xdd, Real_t *ydd, Real_t *zdd,
                          const Real_t dt, const Real_t u_cut,
                          Index_t numNode)
{
  Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(xd[numNode], \
                                  yd[numNode], \
                                  zd[numNode], \
                                  xdd[numNode], \
                                  ydd[numNode], \
                                  zdd[numNode])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numNode)
#endif
  for ( i = 0 ; i < numNode ; ++i )
  {
    Real_t xdtmp, ydtmp, zdtmp ;

    xdtmp = xd[i] + xdd[i] * dt ;
    if( fabs(xdtmp) < u_cut ) xdtmp = (Real_t)(0.0);
    xd[i] = xdtmp ;

    ydtmp = yd[i] + ydd[i] * dt ;
    if( fabs(ydtmp) < u_cut ) ydtmp = (Real_t)(0.0);
    yd[i] = ydtmp ;

    zdtmp = zd[i] + zdd[i] * dt ;
    if( fabs(zdtmp) < u_cut ) zdtmp = (Real_t)(0.0);
    zd[i] = zdtmp ;
  }
}

/******************************************/

static inline
void CalcPositionForNodes(Real_t *x,  Real_t *y,  Real_t *z,
                          Real_t *xd, Real_t *yd, Real_t *zd,
                          const Real_t dt, Index_t numNode)
{
  Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(x[numNode], \
                                  y[numNode], \
                                  z[numNode], \
                                  xd[numNode], \
                                  yd[numNode], \
                                  zd[numNode])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numNode)
#endif
  for ( i = 0 ; i < numNode ; ++i )
  {
    x[i] += xd[i] * dt ;
    y[i] += yd[i] * dt ;
    z[i] += zd[i] * dt ;
  }
}

/******************************************/

static inline
void LagrangeNodal()
{
#ifdef SEDOV_SYNC_POS_VEL_EARLY
  Real_t *fieldData[6] ;
#endif

  const Real_t delt = m_deltatime ;
  Real_t u_cut = m_u_cut ;

  Index_t numNode = m_numNode;
  Index_t numElem = m_numElem;
  Real_t *fx = m_fx;
  Real_t *fy = m_fy;
  Real_t *fz = m_fz;

  Real_t *x = m_x;
  Real_t *y = m_y;
  Real_t *z = m_z;

  Real_t *xd = m_xd;
  Real_t *yd = m_yd;
  Real_t *zd = m_zd;

  Real_t *xdd = m_xdd;
  Real_t *ydd = m_ydd;
  Real_t *zdd = m_zdd;
  
  Real_t *nodalMass = m_nodalMass;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes();

#if USE_MPI  
#ifdef SEDOV_SYNC_POS_VEL_EARLY
  CommRecv(MSG_SYNC_POS_VEL, 6,
           m_sizeX + 1, m_sizeY + 1, m_sizeZ + 1,
           false, false) ;
#endif
#endif

// redundant data region to allow for early acc updates before communication
#pragma acc data present(x[numNode], \
                         y[numNode], \
                         z[numNode], \
                         xd[numNode], \
                         yd[numNode], \
                         zd[numNode])
  {
#if USE_MPI
    /* used for async update */
    volatile int up = 1;
    /* wait for async device update in CalcForceForNodes to complete */
#pragma acc wait
#endif

    CalcAccelerationForNodes(xdd, ydd, zdd,
                             fx, fy, fz,
                             nodalMass, numNode);

    ApplyAccelerationBoundaryConditionsForNodes(xdd, ydd, zdd);

    CalcVelocityForNodes( xd, yd,  zd,
                          xdd, ydd, zdd,
                          delt, u_cut, m_numNode) ;
#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
  /* start to update velocities asynchronously before the MPI comm */
#pragma acc update host(xd[numNode], \
                        yd[numNode], \
                        zd[numNode]) \
                   async(up)
#endif
#endif

    CalcPositionForNodes( x,  y,  z,
                          xd, yd, zd,
                          delt, m_numNode );

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
#pragma acc update host(x[numNode], \
                        y[numNode], \
                        z[numNode]) \
                   async(up)
#pragma acc wait(up)
    fieldData[0] = x ;
    fieldData[1] = y ;
    fieldData[2] = z ;
    fieldData[3] = xd ;
    fieldData[4] = yd ;
    fieldData[5] = zd ;

    CommSend(MSG_SYNC_POS_VEL, 6, fieldData,
        m_sizeX + 1, m_sizeY + 1, m_sizeZ + 1,
        false, false) ;
    CommSyncPosVel() ;

/* update device after CommRecv */
#pragma acc update device(x[numNode], \
                          y[numNode], \
                          z[numNode], \
                          xd[numNode], \
                          yd[numNode], \
                          zd[numNode]) \
                   async
#endif
#endif
  } // end acc data
   
  return;
}

/******************************************/
static inline
Real_t CalcElemVolumeStatic( const Real_t x0, const Real_t x1,
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
  Real_t twelveth = (Real_t)(1.0)/(Real_t)(12.0);

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

/* defined again outside because you can not define macros within macros */
#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

//static inline
#define CalcElemVolume_Full(x0, x1, \
                            x2, x3, \
                            x4, x5, \
                            x6, x7, \
                            y0, y1, \
                            y2, y3, \
                            y4, y5, \
                            y6, y7, \
                            z0, z1, \
                            z2, z3, \
                            z4, z5, \
                            z6, z7) \
do { \
  Real_t twelveth = (Real_t)(1.0)/(Real_t)(12.0); \
 \
  Real_t dx61 = (x6) - (x1); \
  Real_t dy61 = (y6) - (y1); \
  Real_t dz61 = (z6) - (z1); \
 \
  Real_t dx70 = (x7) - (x0); \
  Real_t dy70 = (y7) - (y0); \
  Real_t dz70 = (z7) - (z0); \
 \
  Real_t dx63 = (x6) - (x3); \
  Real_t dy63 = (y6) - (y3); \
  Real_t dz63 = (z6) - (z3); \
 \
  Real_t dx20 = (x2) - (x0); \
  Real_t dy20 = (y2) - (y0); \
  Real_t dz20 = (z2) - (z0); \
 \
  Real_t dx50 = (x5) - (x0); \
  Real_t dy50 = (y5) - (y0); \
  Real_t dz50 = (z5) - (z0); \
 \
  Real_t dx64 = (x6) - (x4); \
  Real_t dy64 = (y6) - (y4); \
  Real_t dz64 = (z6) - (z4); \
 \
  Real_t dx31 = (x3) - (x1); \
  Real_t dy31 = (y3) - (y1); \
  Real_t dz31 = (z3) - (z1); \
 \
  Real_t dx72 = (x7) - (x2); \
  Real_t dy72 = (y7) - (y2); \
  Real_t dz72 = (z7) - (z2); \
 \
  Real_t dx43 = (x4) - (x3); \
  Real_t dy43 = (y4) - (y3); \
  Real_t dz43 = (z4) - (z3); \
 \
  Real_t dx57 = (x5) - (x7); \
  Real_t dy57 = (y5) - (y7); \
  Real_t dz57 = (z5) - (z7); \
 \
  Real_t dx14 = (x1) - (x4); \
  Real_t dy14 = (y1) - (y4); \
  Real_t dz14 = (z1) - (z4); \
 \
  Real_t dx25 = (x2) - (x5); \
  Real_t dy25 = (y2) - (y5); \
  Real_t dz25 = (z2) - (z5); \
 \
  volume = \
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20, \
       dy31 + dy72, dy63, dy20, \
       dz31 + dz72, dz63, dz20) + \
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70, \
       dy43 + dy57, dy64, dy70, \
       dz43 + dz57, dz64, dz70) + \
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50, \
       dy14 + dy25, dy61, dy50, \
       dz14 + dz25, dz61, dz50); \
 \
  volume *= twelveth; \
} while(0)

/******************************************/

//inline
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolumeStatic( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);

}

//static inline 
#define CalcElemVolume_macro(x,y,z) \
do { \
  CalcElemVolume_Full((x.v0), (x.v1), (x.v2), (x.v3), (x.v4), (x.v5), (x.v6), (x.v7), \
                      (y.v0), (y.v1), (y.v2), (y.v3), (y.v4), (y.v5), (y.v6), (y.v7), \
                      (z.v0), (z.v1), (z.v2), (z.v3), (z.v4), (z.v5), (z.v6), (z.v7)); \
} while(0)

/******************************************/

//static inline
#define AreaFace_macro(x0, x1, \
                       x2, x3, \
                       y0, y1, \
                       y2, y3, \
                       z0, z1, \
                       z2, z3) \
do { \
  Real_t fx = (x2 - x0) - (x3 - x1); \
  Real_t fy = (y2 - y0) - (y3 - y1); \
  Real_t fz = (z2 - z0) - (z3 - z1); \
  Real_t gx = (x2 - x0) + (x3 - x1); \
  Real_t gy = (y2 - y0) + (y3 - y1); \
  Real_t gz = (z2 - z0) + (z3 - z1); \
  a = \
    (fx * fx + fy * fy + fz * fz) * \
    (gx * gx + gy * gy + gz * gz) - \
    (fx * gx + fy * gy + fz * gz) * \
    (fx * gx + fy * gy + fz * gz); \
} while(0)

/******************************************/

//static inline
#define CalcElemCharacteristicLength_macro(x, \
                                           y, \
                                           z, \
                                           volume) \
do { \
  charLength = (Real_t)(0.0); \
  Real_t a;  \
 \
  AreaFace_macro(x.v0,x.v1,x.v2,x.v3, \
                 y.v0,y.v1,y.v2,y.v3, \
                 z.v0,z.v1,z.v2,z.v3) ; \
  charLength = MAX(a,charLength) ; \
 \
  AreaFace_macro(x.v4,x.v5,x.v6,x.v7, \
                 y.v4,y.v5,y.v6,y.v7, \
                 z.v4,z.v5,z.v6,z.v7) ; \
  charLength = MAX(a,charLength) ; \
 \
  AreaFace_macro(x.v0,x.v1,x.v5,x.v4, \
                 y.v0,y.v1,y.v5,y.v4, \
                 z.v0,z.v1,z.v5,z.v4) ; \
  charLength = MAX(a,charLength) ; \
 \
  AreaFace_macro(x.v1,x.v2,x.v6,x.v5, \
                 y.v1,y.v2,y.v6,y.v5, \
                 z.v1,z.v2,z.v6,z.v5) ; \
  charLength = MAX(a,charLength) ; \
 \
  AreaFace_macro(x.v2,x.v3,x.v7,x.v6, \
                 y.v2,y.v3,y.v7,y.v6, \
                 z.v2,z.v3,z.v7,z.v6) ; \
  charLength = MAX(a,charLength) ; \
 \
  AreaFace_macro(x.v3,x.v0,x.v4,x.v7, \
                 y.v3,y.v0,y.v4,y.v7, \
                 z.v3,z.v0,z.v4,z.v7) ; \
  charLength = MAX(a,charLength) ; \
 \
  charLength = (Real_t)(4.0) * volume / sqrt(charLength); \
} while(0)

/******************************************/

//static inline
#define CalcElemVelocityGradient_macro(xvel, \
                                       yvel, \
                                       zvel, \
                                       b, \
                                       detJ, \
                                       d) \
do { \
  const Real_t inv_detJ = (Real_t)(1.0) / detJ ; \
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz; \
 \
  d.v0 = inv_detJ * ( b.v0_0 * (xvel.v0-xvel.v6) \
                     + b.v0_1 * (xvel.v1-xvel.v7) \
                     + b.v0_2 * (xvel.v2-xvel.v4) \
                     + b.v0_3 * (xvel.v3-xvel.v5) ); \
 \
  d.v1 = inv_detJ * ( b.v1_0 * (yvel.v0-yvel.v6) \
                     + b.v1_1 * (yvel.v1-yvel.v7) \
                     + b.v1_2 * (yvel.v2-yvel.v4) \
                     + b.v1_3 * (yvel.v3-yvel.v5) ); \
 \
  d.v2 = inv_detJ * ( b.v2_0 * (zvel.v0-zvel.v6) \
                     + b.v2_1 * (zvel.v1-zvel.v7) \
                     + b.v2_2 * (zvel.v2-zvel.v4) \
                     + b.v2_3 * (zvel.v3-zvel.v5) ); \
 \
  dyddx  = inv_detJ * ( b.v0_0 * (yvel.v0-yvel.v6) \
                      + b.v0_1 * (yvel.v1-yvel.v7) \
                      + b.v0_2 * (yvel.v2-yvel.v4) \
                      + b.v0_3 * (yvel.v3-yvel.v5) ); \
 \
  dxddy  = inv_detJ * ( b.v1_0 * (xvel.v0-xvel.v6) \
                      + b.v1_1 * (xvel.v1-xvel.v7) \
                      + b.v1_2 * (xvel.v2-xvel.v4) \
                      + b.v1_3 * (xvel.v3-xvel.v5) ); \
 \
  dzddx  = inv_detJ * ( b.v0_0 * (zvel.v0-zvel.v6) \
                      + b.v0_1 * (zvel.v1-zvel.v7) \
                      + b.v0_2 * (zvel.v2-zvel.v4) \
                      + b.v0_3 * (zvel.v3-zvel.v5) ); \
 \
  dxddz  = inv_detJ * ( b.v2_0 * (xvel.v0-xvel.v6) \
                      + b.v2_1 * (xvel.v1-xvel.v7) \
                      + b.v2_2 * (xvel.v2-xvel.v4) \
                      + b.v2_3 * (xvel.v3-xvel.v5) ); \
 \
  dzddy  = inv_detJ * ( b.v1_0 * (zvel.v0-zvel.v6) \
                      + b.v1_1 * (zvel.v1-zvel.v7) \
                      + b.v1_2 * (zvel.v2-zvel.v4) \
                      + b.v1_3 * (zvel.v3-zvel.v5) ); \
 \
  dyddz  = inv_detJ * ( b.v2_0 * (yvel.v0-yvel.v6) \
                      + b.v2_1 * (yvel.v1-yvel.v7) \
                      + b.v2_2 * (yvel.v2-yvel.v4) \
                      + b.v2_3 * (yvel.v3-yvel.v5) ); \
  d.v5  = (Real_t)( .5) * ( dxddy + dyddx ); \
  d.v4  = (Real_t)( .5) * ( dxddz + dzddx ); \
  d.v3  = (Real_t)( .5) * ( dzddy + dyddz ); \
} while(0)

/******************************************/

//static inline
void CalcKinematicsForElems( Index_t *nodelist,
                             Real_t *x,   Real_t *y,   Real_t *z,
                             Real_t *xd,  Real_t *yd,  Real_t *zd,
                             Real_t *dxx, Real_t *dyy, Real_t *dzz,
                             Real_t *v, Real_t *volo,
                             Real_t *vnew, Real_t *delv, Real_t *arealg,
                             Real_t deltaTime, Index_t numElem, Index_t numNode)
{
  volatile Index_t numElem8 = numElem * 8;

  Index_t k;

  // loop over all elements
#ifdef _OPENACC
#pragma acc parallel loop present(dxx[numElem], \
                                  dyy[numElem], \
                                  dzz[numElem], \
                                  x[numNode], \
                                  y[numNode], \
                                  z[numNode], \
                                  xd[numNode], \
                                  yd[numNode], \
                                  zd[numNode], \
                                  v[numElem], \
                                  volo[numElem], \
                                  vnew[numElem], \
                                  delv[numElem], \
                                  arealg[numElem], \
                                  nodelist[numElem8])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numElem, deltaTime)
#endif
  for( k=0 ; k<numElem ; ++k )
  {
    bmat B ; /** shape function derivatives */
    val6 D ;
    val8 x_local ;
    val8 y_local ;
    val8 z_local ;
    val8 xd_local ;
    val8 yd_local ;
    val8 zd_local ;
    Real_t detJ = (Real_t)(0.0) ;

    Real_t volume ;
    Real_t relativeVolume ;
    const Index_t* const elemToNode = &nodelist[8*k] ;

    // get nodal coordinates from global arrays and copy into local arrays.
    // Loop unrolled because the PGI OpenACC implementation currently stores
    // locally-defined arrays in a global, shared context. Thus we have to use
    // scalars instead to get them in registers.
    Index_t gnode;
    gnode = elemToNode[0];
    x_local.v0 = x[gnode];
    y_local.v0 = y[gnode];
    z_local.v0 = z[gnode];
    gnode = elemToNode[1];
    x_local.v1 = x[gnode];
    y_local.v1 = y[gnode];
    z_local.v1 = z[gnode];
    gnode = elemToNode[2];
    x_local.v2 = x[gnode];
    y_local.v2 = y[gnode];
    z_local.v2 = z[gnode];
    gnode = elemToNode[3];
    x_local.v3 = x[gnode];
    y_local.v3 = y[gnode];
    z_local.v3 = z[gnode];
    gnode = elemToNode[4];
    x_local.v4 = x[gnode];
    y_local.v4 = y[gnode];
    z_local.v4 = z[gnode];
    gnode = elemToNode[5];
    x_local.v5 = x[gnode];
    y_local.v5 = y[gnode];
    z_local.v5 = z[gnode];
    gnode = elemToNode[6];
    x_local.v6 = x[gnode];
    y_local.v6 = y[gnode];
    z_local.v6 = z[gnode];
    gnode = elemToNode[7];
    x_local.v7 = x[gnode];
    y_local.v7 = y[gnode];
    z_local.v7 = z[gnode];

    // volume calculations - CalcElemVolume is a macro that sets volume
    CalcElemVolume_macro(x_local, y_local, z_local );
    relativeVolume = volume / volo[k] ;
    vnew[k] = relativeVolume ;
    delv[k] = relativeVolume - v[k] ;

    // set characteristic length
    Real_t charLength;
    CalcElemCharacteristicLength_macro(x_local, y_local, z_local,
                                       volume);
    arealg[k] = charLength;

    // get nodal velocities from global array and copy into local arrays.
    gnode = elemToNode[0];
    xd_local.v0 = xd[gnode];
    yd_local.v0 = yd[gnode];
    zd_local.v0 = zd[gnode];
    gnode = elemToNode[1];
    xd_local.v1 = xd[gnode];
    yd_local.v1 = yd[gnode];
    zd_local.v1 = zd[gnode];
    gnode = elemToNode[2];
    xd_local.v2 = xd[gnode];
    yd_local.v2 = yd[gnode];
    zd_local.v2 = zd[gnode];
    gnode = elemToNode[3];
    xd_local.v3 = xd[gnode];
    yd_local.v3 = yd[gnode];
    zd_local.v3 = zd[gnode];
    gnode = elemToNode[4];
    xd_local.v4 = xd[gnode];
    yd_local.v4 = yd[gnode];
    zd_local.v4 = zd[gnode];
    gnode = elemToNode[5];
    xd_local.v5 = xd[gnode];
    yd_local.v5 = yd[gnode];
    zd_local.v5 = zd[gnode];
    gnode = elemToNode[6];
    xd_local.v6 = xd[gnode];
    yd_local.v6 = yd[gnode];
    zd_local.v6 = zd[gnode];
    gnode = elemToNode[7];
    xd_local.v7 = xd[gnode];
    yd_local.v7 = yd[gnode];
    zd_local.v7 = zd[gnode];

    Real_t dt2 = (Real_t)(0.5) * deltaTime;
    x_local.v0 -= dt2 * xd_local.v0;
    y_local.v0 -= dt2 * yd_local.v0;
    z_local.v0 -= dt2 * zd_local.v0;
    x_local.v1 -= dt2 * xd_local.v1;
    y_local.v1 -= dt2 * yd_local.v1;
    z_local.v1 -= dt2 * zd_local.v1;
    x_local.v2 -= dt2 * xd_local.v2;
    y_local.v2 -= dt2 * yd_local.v2;
    z_local.v2 -= dt2 * zd_local.v2;
    x_local.v3 -= dt2 * xd_local.v3;
    y_local.v3 -= dt2 * yd_local.v3;
    z_local.v3 -= dt2 * zd_local.v3;
    x_local.v4 -= dt2 * xd_local.v4;
    y_local.v4 -= dt2 * yd_local.v4;
    z_local.v4 -= dt2 * zd_local.v4;
    x_local.v5 -= dt2 * xd_local.v5;
    y_local.v5 -= dt2 * yd_local.v5;
    z_local.v5 -= dt2 * zd_local.v5;
    x_local.v6 -= dt2 * xd_local.v6;
    y_local.v6 -= dt2 * yd_local.v6;
    z_local.v6 -= dt2 * zd_local.v6;
    x_local.v7 -= dt2 * xd_local.v7;
    y_local.v7 -= dt2 * yd_local.v7;
    z_local.v7 -= dt2 * zd_local.v7;

    CalcElemShapeFunctionDerivatives_unrolled( x_local, y_local, z_local,
                                               B, detJ );

    CalcElemVelocityGradient_macro( xd_local, yd_local, zd_local,
                                    B, detJ, D );

    // put velocity gradient quantities into their global arrays.
    dxx[k] = D.v0;
    dyy[k] = D.v1;
    dzz[k] = D.v2;
  }
}

/******************************************/

static inline
void CalcLagrangeElements(Real_t* vnew)
{
  Index_t numElem = m_numElem ;
  Index_t numNode = m_numNode ;
  if (numElem > 0) {
    const Real_t deltatime = m_deltatime ;

    // strains are now allocated at startup to prevent unnecessary mem transfers
    Real_t *dxx = m_dxx;
    Real_t *dyy = m_dyy;
    Real_t *dzz = m_dzz;

    Real_t *x = m_x;
    Real_t *y = m_y;
    Real_t *z = m_z;
    Real_t *xd = m_xd;
    Real_t *yd = m_yd;
    Real_t *zd = m_zd;
    Real_t *v = m_v;
    Real_t *volo = m_volo;
    Real_t *vdov = m_vdov;
    Real_t *delv = m_delv;
    Real_t *arealg = m_arealg;

    Index_t *nodelist = m_nodelist;

    CalcKinematicsForElems(nodelist,
                           x, y, z,
                           xd, yd, zd,
                           dxx, dyy, dzz,
                           v, volo,
                           vnew, delv, arealg,
                           deltatime, numElem, numNode);

    // element loop to do some stuff not included in the elemlib function.
    int abort = 0;
    Index_t k;
#ifdef _OPENACC
#pragma acc parallel loop present(vdov[numElem], \
                                  dxx[numElem], \
                                  dyy[numElem], \
                                  dzz[numElem], \
                                  vnew[numElem]) \
                          reduction(max: abort)
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numElem) reduction(max: abort)
#endif
    for ( k=0 ; k<numElem ; ++k )
    {
      // calc strain rate and apply as constraint (only done in FB element)
      Real_t vdov_k = dxx[k] + dyy[k] + dzz[k] ;
      Real_t vdovthird = vdov_k/(Real_t)(3.0) ;

      // make the rate of deformation tensor deviatoric
      vdov[k] = vdov_k ;
      dxx[k] -= vdovthird ;
      dyy[k] -= vdovthird ;
      dzz[k] -= vdovthird ;

      // See if any volumes are negative, and take appropriate action.
      if (vnew[k] <= (Real_t)(0.0))
      {
        abort = 1;
      }
    }
    if(abort) {
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
        exit(VolumeError);
#endif
    }

  } // end if numElem > 0
}

/******************************************/

static inline
void CalcMonotonicQGradientsForElems(Real_t vnew[], Index_t allElem)
{
  volatile Index_t numNode = m_numNode;
  Index_t numElem = m_numElem;
  volatile Int_t numElem8 = m_numElem * 8;

  Real_t *x = m_x;
  Real_t *y = m_y;
  Real_t *z = m_z;
  Real_t *xd = m_xd;
  Real_t *yd = m_yd;
  Real_t *zd = m_zd;
  Real_t *volo = m_volo;
  Index_t *nodelist = m_nodelist;

  Real_t *delv_xi = m_delv_xi;
  Real_t *delv_eta = m_delv_eta;
  Real_t *delv_zeta = m_delv_zeta;
  Real_t *delx_xi = m_delx_xi;
  Real_t *delx_eta = m_delx_eta;
  Real_t *delx_zeta = m_delx_zeta;

  Index_t i;

#ifdef _OPENACC
#pragma acc parallel loop present(vnew[numElem], \
                                  nodelist[numElem8], \
                                  x[numNode], \
                                  y[numNode], \
                                  z[numNode], \
                                  xd[numNode], \
                                  yd[numNode], \
                                  zd[numNode], \
                                  volo[numElem], \
                                  delx_xi[allElem], \
                                  delx_eta[allElem], \
                                  delx_zeta[allElem], \
                                  delv_xi[allElem], \
                                  delv_eta[allElem], \
                                  delv_zeta[allElem])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numElem)
#endif
  for (i = 0 ; i < numElem ; ++i ) {
    const Real_t ptiny = (Real_t)(1.e-36) ;
    Real_t ax,ay,az ;
    Real_t dxv,dyv,dzv ;

    const Index_t *elemToNode = &nodelist[i*8];
    Index_t n0 = elemToNode[0] ;
    Index_t n1 = elemToNode[1] ;
    Index_t n2 = elemToNode[2] ;
    Index_t n3 = elemToNode[3] ;
    Index_t n4 = elemToNode[4] ;
    Index_t n5 = elemToNode[5] ;
    Index_t n6 = elemToNode[6] ;
    Index_t n7 = elemToNode[7] ;

    Real_t x0 = x[n0] ;
    Real_t x1 = x[n1] ;
    Real_t x2 = x[n2] ;
    Real_t x3 = x[n3] ;
    Real_t x4 = x[n4] ;
    Real_t x5 = x[n5] ;
    Real_t x6 = x[n6] ;
    Real_t x7 = x[n7] ;

    Real_t y0 = y[n0] ;
    Real_t y1 = y[n1] ;
    Real_t y2 = y[n2] ;
    Real_t y3 = y[n3] ;
    Real_t y4 = y[n4] ;
    Real_t y5 = y[n5] ;
    Real_t y6 = y[n6] ;
    Real_t y7 = y[n7] ;

    Real_t z0 = z[n0] ;
    Real_t z1 = z[n1] ;
    Real_t z2 = z[n2] ;
    Real_t z3 = z[n3] ;
    Real_t z4 = z[n4] ;
    Real_t z5 = z[n5] ;
    Real_t z6 = z[n6] ;
    Real_t z7 = z[n7] ;

    Real_t xv0 = xd[n0] ;
    Real_t xv1 = xd[n1] ;
    Real_t xv2 = xd[n2] ;
    Real_t xv3 = xd[n3] ;
    Real_t xv4 = xd[n4] ;
    Real_t xv5 = xd[n5] ;
    Real_t xv6 = xd[n6] ;
    Real_t xv7 = xd[n7] ;

    Real_t yv0 = yd[n0] ;
    Real_t yv1 = yd[n1] ;
    Real_t yv2 = yd[n2] ;
    Real_t yv3 = yd[n3] ;
    Real_t yv4 = yd[n4] ;
    Real_t yv5 = yd[n5] ;
    Real_t yv6 = yd[n6] ;
    Real_t yv7 = yd[n7] ;

    Real_t zv0 = zd[n0] ;
    Real_t zv1 = zd[n1] ;
    Real_t zv2 = zd[n2] ;
    Real_t zv3 = zd[n3] ;
    Real_t zv4 = zd[n4] ;
    Real_t zv5 = zd[n5] ;
    Real_t zv6 = zd[n6] ;
    Real_t zv7 = zd[n7] ;

    Real_t vol = volo[i]*vnew[i] ;
    Real_t norm = (Real_t)(1.0) / ( vol + ptiny ) ;

    Real_t dxj = (Real_t)(-0.25)*((x0+x1+x5+x4) - (x3+x2+x6+x7)) ;
    Real_t dyj = (Real_t)(-0.25)*((y0+y1+y5+y4) - (y3+y2+y6+y7)) ;
    Real_t dzj = (Real_t)(-0.25)*((z0+z1+z5+z4) - (z3+z2+z6+z7)) ;

    Real_t dxi = (Real_t)( 0.25)*((x1+x2+x6+x5) - (x0+x3+x7+x4)) ;
    Real_t dyi = (Real_t)( 0.25)*((y1+y2+y6+y5) - (y0+y3+y7+y4)) ;
    Real_t dzi = (Real_t)( 0.25)*((z1+z2+z6+z5) - (z0+z3+z7+z4)) ;

    Real_t dxk = (Real_t)( 0.25)*((x4+x5+x6+x7) - (x0+x1+x2+x3)) ;
    Real_t dyk = (Real_t)( 0.25)*((y4+y5+y6+y7) - (y0+y1+y2+y3)) ;
    Real_t dzk = (Real_t)( 0.25)*((z4+z5+z6+z7) - (z0+z1+z2+z3)) ;

    /* find delvk and delxk ( i cross j ) */

    ax = dyi*dzj - dzi*dyj ;
    ay = dzi*dxj - dxi*dzj ;
    az = dxi*dyj - dyi*dxj ;

    delx_zeta[i] = vol / sqrt(ax*ax + ay*ay + az*az + ptiny) ;

    ax *= norm ;
    ay *= norm ;
    az *= norm ;

    dxv = (Real_t)(0.25)*((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3)) ;
    dyv = (Real_t)(0.25)*((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3)) ;
    dzv = (Real_t)(0.25)*((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3)) ;

    delv_zeta[i] = ax*dxv + ay*dyv + az*dzv ;

    /* find delxi and delvi ( j cross k ) */

    ax = dyj*dzk - dzj*dyk ;
    ay = dzj*dxk - dxj*dzk ;
    az = dxj*dyk - dyj*dxk ;

    delx_xi[i] = vol / sqrt(ax*ax + ay*ay + az*az + ptiny) ;

    ax *= norm ;
    ay *= norm ;
    az *= norm ;

    dxv = (Real_t)(0.25)*((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4)) ;
    dyv = (Real_t)(0.25)*((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4)) ;
    dzv = (Real_t)(0.25)*((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4)) ;

    delv_xi[i] = ax*dxv + ay*dyv + az*dzv ;

    /* find delxj and delvj ( k cross i ) */

    ax = dyk*dzi - dzk*dyi ;
    ay = dzk*dxi - dxk*dzi ;
    az = dxk*dyi - dyk*dxi ;

    delx_eta[i] = vol / sqrt(ax*ax + ay*ay + az*az + ptiny) ;

    ax *= norm ;
    ay *= norm ;
    az *= norm ;

    dxv = (Real_t)(-0.25)*((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7)) ;
    dyv = (Real_t)(-0.25)*((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7)) ;
    dzv = (Real_t)(-0.25)*((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7)) ;

    delv_eta[i] = ax*dxv + ay*dyv + az*dzv ;
  }
}

/******************************************/
/*
 * NOTES: This function uses several goto statements. They are used in the
 * place of breaks. This is the result of a bug in the PGI compiler (version
 * (13.4-accelerator) in which breaks inside of switches jump out of the omp
 * loops they are placed in. We decided that using gotos is a more readable
 * alternative than rewriting them all to if-else blocks.
 */
static inline
void CalcMonotonicQRegionForElems(Int_t r,
                                  Real_t vnew[], Real_t ptiny, Index_t allElem)
{
  Real_t monoq_limiter_mult = m_monoq_limiter_mult;
  Real_t monoq_max_slope = m_monoq_max_slope;
  Real_t qlc_monoq = m_qlc_monoq;
  Real_t qqc_monoq = m_qqc_monoq;

  Index_t *lxim = m_lxim;
  Index_t *lxip = m_lxip;
  Index_t *letam = m_letam;
  Index_t *letap = m_letap;
  Index_t *lzetam = m_lzetam;
  Index_t *lzetap = m_lzetap;

  Real_t *delv_xi = m_delv_xi;
  Real_t *delv_eta = m_delv_eta;
  Real_t *delv_zeta = m_delv_zeta;
  Real_t *delx_xi = m_delx_xi;
  Real_t *delx_eta = m_delx_eta;
  Real_t *delx_zeta = m_delx_zeta;

  Real_t *qq = m_qq;
  Real_t *ql = m_ql;
  
  Real_t *elemMass = m_elemMass;
  Real_t *volo = m_volo;
  Real_t *vdov = m_vdov;

  Index_t regElemSize = m_regElemSize[r];
  Index_t *regElemlist = m_regElemlist[r];

  volatile Index_t numElem = m_numElem;
  Int_t *elemBC = m_elemBC;

  Index_t ielem;

#ifdef _OPENACC
#pragma acc parallel loop firstprivate(qlc_monoq, qqc_monoq, \
                                       monoq_limiter_mult, monoq_max_slope, \
                                       ptiny) \
                          copyin(regElemlist[regElemSize]) \
                          present(vnew[numElem], \
                                  vdov[numElem], \
                                  delx_xi[allElem], \
                                  delx_eta[allElem], \
                                  delx_zeta[allElem], \
                                  delv_xi[allElem], \
                                  delv_eta[allElem], \
                                  delv_zeta[allElem], \
                                  elemMass[numElem], \
                                  volo[numElem], \
                                  lxip[numElem], \
                                  lxim[numElem], \
                                  letam[numElem], \
                                  letap[numElem], \
                                  lzetam[numElem], \
                                  lzetap[numElem], \
                                  ql[numElem], \
                                  qq[numElem], \
                                  elemBC[numElem])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(qlc_monoq, qqc_monoq, monoq_limiter_mult, monoq_max_slope, ptiny)
#endif
  for ( ielem = 0 ; ielem < regElemSize; ++ielem ) {
    Index_t i = regElemlist[ielem];
    Real_t qlin, qquad ;
    Real_t phixi, phieta, phizeta ;
    Int_t bcMask = elemBC[i];
    Real_t delvm, delvp ;

    /*  phixi     */
    Real_t norm = (Real_t)(1.) / (delv_xi[i]+ ptiny ) ;

    switch (bcMask & XI_M) {
      case XI_M_COMM: /* needs comm data */
      case 0:         delvm = delv_xi[lxim[i]];      goto BCMASK_AND_XI_M;
      case XI_M_SYMM: delvm = delv_xi[i] ;           goto BCMASK_AND_XI_M;
      case XI_M_FREE: delvm = (Real_t)(0.0) ;          goto BCMASK_AND_XI_M;
      default:        /* ERROR */ ;                  goto BCMASK_AND_XI_M;
    }
    BCMASK_AND_XI_M:

    switch (bcMask & XI_P) {
      case XI_P_COMM: /* needs comm data */
      case 0:         delvp = delv_xi[lxip[i]] ;     goto BCMASK_AND_XI_P;
      case XI_P_SYMM: delvp = delv_xi[i] ;           goto BCMASK_AND_XI_P;
      case XI_P_FREE: delvp = (Real_t)(0.0) ;          goto BCMASK_AND_XI_P;
      default:        /* ERROR */ ;                  goto BCMASK_AND_XI_P;
    }
    BCMASK_AND_XI_P:

    delvm = delvm * norm ;
    delvp = delvp * norm ;

    phixi = (Real_t)(.5) * ( delvm + delvp ) ;

    delvm *= monoq_limiter_mult ;
    delvp *= monoq_limiter_mult ;

    if ( delvm < phixi ) phixi = delvm ;
    if ( delvp < phixi ) phixi = delvp ;
    if ( phixi < (Real_t)(0.)) phixi = (Real_t)(0.) ;
    if ( phixi > monoq_max_slope) phixi = monoq_max_slope;

    /*  phieta     */
    norm = (Real_t)(1.) / ( delv_eta[i] + ptiny ) ;

    switch (bcMask & ETA_M) {
      case ETA_M_COMM: /* needs comm data */
      case 0:          delvm = delv_eta[letam[i]] ;  goto BCMASK_AND_ETA_M;
      case ETA_M_SYMM: delvm = delv_eta[i] ;         goto BCMASK_AND_ETA_M;
      case ETA_M_FREE: delvm = (Real_t)(0.0) ;         goto BCMASK_AND_ETA_M;
      default:         /* ERROR */ ;                 goto BCMASK_AND_ETA_M;
    }
    BCMASK_AND_ETA_M:

    switch (bcMask & ETA_P) {
      case ETA_P_COMM: /* needs comm data */
      case 0:          delvp = delv_eta[letap[i]] ;  goto BCMASK_AND_ETA_P;
      case ETA_P_SYMM: delvp = delv_eta[i] ;         goto BCMASK_AND_ETA_P;
      case ETA_P_FREE: delvp = (Real_t)(0.0) ;         goto BCMASK_AND_ETA_P;
      default:         /* ERROR */ ;                 goto BCMASK_AND_ETA_P;
    }
    BCMASK_AND_ETA_P:

    delvm = delvm * norm ;
    delvp = delvp * norm ;

    phieta = (Real_t)(.5) * ( delvm + delvp ) ;

    delvm *= monoq_limiter_mult ;
    delvp *= monoq_limiter_mult ;

    if ( delvm  < phieta ) phieta = delvm ;
    if ( delvp  < phieta ) phieta = delvp ;
    if ( phieta < (Real_t)(0.)) phieta = (Real_t)(0.) ;
    if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

    /*  phizeta     */
    norm = (Real_t)(1.) / ( delv_zeta[i] + ptiny ) ;

    switch (bcMask & ZETA_M) {
      case ZETA_M_COMM: /* needs comm data */
      case 0:           delvm = delv_zeta[lzetam[i]] ;  goto BCMASK_AND_ZETA_M;
      case ZETA_M_SYMM: delvm = delv_zeta[i] ;          goto BCMASK_AND_ZETA_M;
      case ZETA_M_FREE: delvm = (Real_t)(0.0) ;           goto BCMASK_AND_ZETA_M;
      default:          /* ERROR */ ;                   goto BCMASK_AND_ZETA_M;
    }
    BCMASK_AND_ZETA_M:

    switch (bcMask & ZETA_P) {
      case ZETA_P_COMM: /* needs comm data */
      case 0:           delvp = delv_zeta[lzetap[i]] ;  goto BCMASK_AND_ZETA_P;
      case ZETA_P_SYMM: delvp = delv_zeta[i] ;          goto BCMASK_AND_ZETA_P;
      case ZETA_P_FREE: delvp = (Real_t)(0.0) ;           goto BCMASK_AND_ZETA_P;
      default:          /* ERROR */ ;                   goto BCMASK_AND_ZETA_P;
    }
    BCMASK_AND_ZETA_P:

    delvm = delvm * norm ;
    delvp = delvp * norm ;

    phizeta = (Real_t)(.5) * ( delvm + delvp ) ;

    delvm *= monoq_limiter_mult ;
    delvp *= monoq_limiter_mult ;

    if ( delvm   < phizeta ) phizeta = delvm ;
    if ( delvp   < phizeta ) phizeta = delvp ;
    if ( phizeta < (Real_t)(0.)) phizeta = (Real_t)(0.);
    if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

    /* Remove length scale */

    if ( vdov[i] > (Real_t)(0.) )  {
      qlin  = (Real_t)(0.) ;
      qquad = (Real_t)(0.) ;
    }
    else {
      Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
      Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
      Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

      if ( delvxxi   > (Real_t)(0.) ) delvxxi   = (Real_t)(0.) ;
      if ( delvxeta  > (Real_t)(0.) ) delvxeta  = (Real_t)(0.) ;
      if ( delvxzeta > (Real_t)(0.) ) delvxzeta = (Real_t)(0.) ;

      Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;

      qlin = -qlc_monoq * rho *
        (  delvxxi   * ((Real_t)(1.) - phixi) +
           delvxeta  * ((Real_t)(1.) - phieta) +
           delvxzeta * ((Real_t)(1.) - phizeta)  ) ;

      qquad = qqc_monoq * rho *
        (  delvxxi*delvxxi     * ((Real_t)(1.) - phixi*phixi) +
           delvxeta*delvxeta   * ((Real_t)(1.) - phieta*phieta) +
           delvxzeta*delvxzeta * ((Real_t)(1.) - phizeta*phizeta)  ) ;
    }

    qq[i] = qquad ;
    ql[i] = qlin  ;
  }
}

/******************************************/

static inline
void CalcMonotonicQForElems(Real_t vnew[], Index_t allElem)
{  
  //
  // initialize parameters
  // 
  const Real_t ptiny = (Real_t)(1.e-36) ;

  Index_t r;

  //
  // calculate the monotonic q for all regions
  //
  for (r=0 ; r<m_numReg ; ++r) {

    if (m_regElemSize[r] > 0) {
      CalcMonotonicQRegionForElems(r, vnew, ptiny, allElem);
    }
  }
}

/******************************************/

static inline
void CalcQForElems(Real_t vnew[])
{
  //
  // MONOTONIC Q option
  //

  Index_t numElem = m_numElem ;

  if (numElem != 0) {
    int allElem = numElem +  /* local elem */
      2*m_sizeX*m_sizeY + /* plane ghosts */
      2*m_sizeX*m_sizeZ + /* row ghosts */
      2*m_sizeY*m_sizeZ ; /* col ghosts */

    /* Gradients are allocated globally now to reduce memory transfers to 
       device */
    //AllocateGradients(allElem);

#if USE_MPI      
    CommRecv(MSG_MONOQ, 3,
        m_sizeX, m_sizeY, m_sizeZ,
        true, true) ;
#endif      

    /* Calculate velocity gradients */
    CalcMonotonicQGradientsForElems(vnew, allElem);

#if USE_MPI      
    Real_t *fieldData[3] ;
    Real_t *delv_xi = m_delv_xi;
    Real_t *delv_eta = m_delv_eta;
    Real_t *delv_zeta = m_delv_zeta;
#pragma acc data present(delv_xi[allElem], \
                         delv_eta[allElem], \
                         delv_zeta[allElem])
{
#pragma acc update host(delv_xi[allElem], \
                        delv_eta[allElem], \
                        delv_zeta[allElem])

    /* Transfer veloctiy gradients in the first order elements */
    /* problem->commElements->Transfer(CommElements::monoQ) ; */

    fieldData[0] = delv_xi;
    fieldData[1] = delv_eta;
    fieldData[2] = delv_zeta;

    CommSend(MSG_MONOQ, 3, fieldData,
             m_sizeX, m_sizeY, m_sizeZ,
             true, true) ;

    CommMonoQ() ;

} // end acc data
#endif      

    CalcMonotonicQForElems(vnew, allElem) ;

    // Free up memory
    //DeallocateGradients();

    /* Don't allow excessive artificial viscosity */
    Index_t idx = -1; 
    Index_t i;
    for (i=0; i<numElem; ++i) {
      if ( m_q[i] > m_qstop ) {
        idx = i ;
        break ;
      }
    }

    if(idx >= 0) {
#if USE_MPI         
      MPI_Abort(MPI_COMM_WORLD, QStopError) ;
#else
      exit(QStopError);
#endif
    }
  }
}

/******************************************/

static inline
void CalcPressureForElems(Real_t* p_new, Real_t* bvc,
                          Real_t* pbvc, Real_t* e_old,
                          Real_t* compression, Real_t *vnewc,
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          Index_t length, Index_t *regElemList)
{

  volatile Index_t numElem = m_numElem;
  Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(regElemList[length], \
                         compression[length], \
                         pbvc[length], \
                         p_new[length], \
                         bvc[length], \
                         e_old[length], \
                         vnewc[numElem])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(length, pmin, p_cut, eosvmax)
#endif
  for (i = 0 ; i < length ; ++i){
    Index_t elem = regElemList[i];

    // fused loop
    Real_t c1s = (Real_t)(2.0)/(Real_t)(3.0) ;
    bvc[i] = c1s * (compression[i] + (Real_t)(1.));
    pbvc[i] = c1s;

    p_new[i] = bvc[i] * e_old[i] ;

    if    (fabs(p_new[i]) <  p_cut   )
      p_new[i] = (Real_t)(0.0) ;

    if    ( vnewc[elem] >= eosvmax ) /* impossible condition here? */
      p_new[i] = (Real_t)(0.0) ;

    if    (p_new[i]       <  pmin)
      p_new[i]   = pmin ;
  }
}

/******************************************/

static inline
void CalcEnergyForElems(Real_t* p_new, Real_t* e_new, Real_t* q_new,
                        Real_t* bvc, Real_t* pbvc,
                        Real_t* p_old, Real_t* e_old, Real_t* q_old,
                        Real_t* compression, Real_t* compHalfStep,
                        Real_t* vnewc, Real_t* work, Real_t* delvc, Real_t pmin,
                        Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
                        Real_t* qq_old, Real_t* ql_old,
                        Real_t rho0,
                        Real_t eosvmax,
                        Int_t length, Index_t *regElemList)
{
  Real_t *pHalfStep = (Real_t*) malloc(sizeof(Real_t) * (length)) ;

    Index_t i;

#ifdef _OPENACC
  volatile Index_t numElem = m_numElem;
#pragma acc data create(pHalfStep[length])
{
#endif

#ifdef _OPENACC
#pragma acc parallel loop present(e_new[length], \
                                  e_old[length], \
                                  p_old[length], \
                                  q_old[length], \
                                  delvc[length], \
                                  work[length])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(length, emin)
#endif
  for (i = 0 ; i < length ; ++i) {
    e_new[i] = e_old[i] - (Real_t)(0.5) * delvc[i] * (p_old[i] + q_old[i])
      + (Real_t)(0.5) * work[i];

    if (e_new[i]  < emin ) {
      e_new[i] = emin ;
    }
  }

  CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
      pmin, p_cut, eosvmax, length, regElemList);

#ifdef _OPENACC
#pragma acc parallel loop present(compHalfStep[length], \
                                  pHalfStep[length], \
                                  delvc[length], \
                                  p_old[length], \
                                  q_old[length], \
                                  ql_old[length], \
                                  qq_old[length], \
                                  q_new[length], \
                                  pbvc[length], \
                                  bvc[length], \
                                  e_new[length])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(length, rho0)
#endif
  for (i = 0 ; i < length ; ++i) {
    Real_t vhalf = (Real_t)(1.) / ((Real_t)(1.) + compHalfStep[i]) ;

    if ( delvc[i] > (Real_t)(0.) ) {
      q_new[i] /* = qq_old[i] = ql_old[i] */ = (Real_t)(0.) ;
    }
    else {
      Real_t ssc = ( pbvc[i] * e_new[i]
          + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;

      if ( ssc <= (Real_t)(.1111111e-36) ) {
        ssc = (Real_t)(.3333333e-18) ;
      } else {
        ssc = sqrt(ssc) ;
      }

      q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
    }

    e_new[i] = e_new[i] + (Real_t)(0.5) * delvc[i]
      * (  (Real_t)(3.0)*(p_old[i]     + q_old[i])
         - (Real_t)(4.0)*(pHalfStep[i] + q_new[i])) ;
  }

#ifdef _OPENACC
#pragma acc parallel loop present(e_new[length], \
                                  work[length])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(length, emin, e_cut)
#endif
  for (i = 0 ; i < length ; ++i) {

    e_new[i] += (Real_t)(0.5) * work[i];

    if (fabs(e_new[i]) < e_cut) {
      e_new[i] = (Real_t)(0.)  ;
    }
    if (     e_new[i]  < emin ) {
      e_new[i] = emin ;
    }
  }

  CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                       pmin, p_cut, eosvmax, length, regElemList);

#ifdef _OPENACC
#pragma acc parallel loop present(regElemList[length], \
                                  pHalfStep[length], \
                                  delvc[length], \
                                  pbvc[length], \
                                  e_new[length], \
                                  bvc[length], \
                                  ql_old[length], \
                                  qq_old[length], \
                                  p_old[length], \
                                  q_old[length], \
                                  p_new[length], \
                                  q_new[length], \
                                  vnewc[numElem])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(length, rho0, emin, e_cut)
#endif
  for (i = 0 ; i < length ; ++i){
    const Real_t sixth = (Real_t)(1.0) / (Real_t)(6.0) ;
    Index_t elem = regElemList[i];
    Real_t q_tilde ;

    if (delvc[i] > (Real_t)(0.)) {
      q_tilde = (Real_t)(0.) ;
    }
    else {
      Real_t ssc = ( pbvc[i] * e_new[i]
          + vnewc[elem] * vnewc[elem] * bvc[i] * p_new[i] ) / rho0 ;

      if ( ssc <= (Real_t)(.1111111e-36) ) {
        ssc = (Real_t)(.3333333e-18) ;
      } else {
        ssc = sqrt(ssc) ;
      }

      q_tilde = (ssc*ql_old[i] + qq_old[i]) ;
    }

    e_new[i] = e_new[i] - (  (Real_t)(7.0)*(p_old[i]     + q_old[i])
        - (Real_t)(8.0)*(pHalfStep[i] + q_new[i])
        + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

    if (fabs(e_new[i]) < e_cut) {
      e_new[i] = (Real_t)(0.)  ;
    }
    if (     e_new[i]  < emin ) {
      e_new[i] = emin ;
    }
  }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax, length, regElemList);

#ifdef _OPENACC
#pragma acc parallel loop present(regElemList[length], \
                                  delvc[length], \
                                  pbvc[length], \
                                  e_new[length], \
                                  vnewc[numElem], \
                                  bvc[length], \
                                  ql_old[length], \
                                  qq_old[length], \
                                  p_new[length], \
                                  q_new[length])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(length, rho0, q_cut)
#endif
   for (i = 0 ; i < length ; ++i){
      Index_t elem = regElemList[i];

      if ( delvc[i] <= (Real_t)(0.) ) {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[elem] * vnewc[elem] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= (Real_t)(.1111111e-36) ) {
            ssc = (Real_t)(.3333333e-18) ;
         } else {
            ssc = sqrt(ssc) ;
         }

         q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;

         if (fabs(q_new[i]) < q_cut) q_new[i] = (Real_t)(0.) ;
      }
   }

#ifdef _OPENACC
} // end acc data
#endif
   if (pHalfStep != NULL) free(pHalfStep);

   return ;
}

/******************************************/

static inline
void CalcSoundSpeedForElems(Real_t *ss,
                            Real_t *vnewc, Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t ss4o3,
                            Index_t numElem, Int_t len, Index_t *regElemList)
{
   Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(vnewc[numElem], \
                                  regElemList[len], \
                                  pbvc[len], \
                                  enewc[len], \
                                  bvc[len], \
                                  pnewc[len], \
                                  ss[numElem]) \
                          firstprivate(rho0, ss4o3)
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(rho0, ss4o3)
#endif
   for (i = 0; i < len ; ++i) {
      int elem = regElemList[i];
      Real_t ssTmp = (pbvc[i] * enewc[i] + vnewc[elem] * vnewc[elem] *
                      bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= (Real_t)(.1111111e-36)) {
         ssTmp = (Real_t)(.3333333e-18);
      }
      else {
         ssTmp = sqrt(ssTmp);
      }
      ss[elem] = ssTmp ;
   }
}

/******************************************/

static inline
void EvalEOSForElems(Real_t *vnewc,
                     Int_t numElemReg, Index_t *regElemList, Int_t rep)
{
  Real_t  e_cut = m_e_cut ;
  Real_t  p_cut = m_p_cut ;
  Real_t  ss4o3 = m_ss4o3 ;
  Real_t  q_cut = m_q_cut ;

  Real_t eosvmax = m_eosvmax ;
  Real_t eosvmin = m_eosvmin ;
  Real_t pmin    = m_pmin ;
  Real_t emin    = m_emin ;
  Real_t rho0    = m_refdens ;

  Real_t *e_old = m_e_old ;
  Real_t *delvc = m_delvc ;
  Real_t *p_old = m_p_old ;
  Real_t *q_old = m_q_old ;
  Real_t *compression = m_compression ;
  Real_t *compHalfStep = m_compHalfStep ;
  Real_t *qq_old = m_qq_old ;
  Real_t *ql_old = m_ql_old ;
  Real_t *work = m_work ;
  Real_t *p_new = m_p_new ;
  Real_t *e_new = m_e_new ;
  Real_t *q_new = m_q_new ;
  Real_t *bvc = m_bvc ;
  Real_t *pbvc = m_pbvc ;

  Real_t *e = m_e;
  Real_t *delv = m_delv;
  Real_t *p = m_p;
  Real_t *q = m_q;
  Real_t *qq = m_qq;
  Real_t *ql = m_ql;
  Index_t numElem = m_numElem;
 
#ifdef _OPENACC
#pragma acc data present(e_old[numElemReg], \
                        delvc[numElemReg], \
                        p_old[numElemReg], \
                        q_old[numElemReg], \
                        compression[numElemReg], \
                        compHalfStep[numElemReg], \
                        qq_old[numElemReg], \
                        ql_old[numElemReg], \
                        work[numElemReg], \
                        p_new[numElemReg], \
                        e_new[numElemReg], \
                        q_new[numElemReg], \
                        bvc[numElemReg], \
                        pbvc[numElemReg]) \
                 copyin(regElemList[numElemReg])
#endif
{ // acc data brace

  Index_t i;
  Int_t j;
  //loop to add load imbalance based on region number 
  for(j = 0; j < rep; j++) {
    /* compress data, minimal set */
#if !defined(_OPENACC) && defined(_OPENMP)
#pragma omp parallel 
#endif
//{ // omp parallel brace

#ifdef _OPENACC
#pragma acc parallel loop present(e_old[numElemReg], \
                                  delvc[numElemReg], \
                                  p_old[numElemReg], \
                                  q_old[numElemReg], \
                                  regElemList[numElemReg], \
                                  qq_old[numElemReg], \
                                  ql_old[numElemReg], \
                                  p[numElem], \
                                  e[numElem], \
                                  q[numElem], \
                                  delv[numElem], \
                                  qq[numElem], \
                                  ql[numElem])
#elif defined(_OPENMP)
#pragma omp for nowait firstprivate(numElemReg)
#endif
    for (i=0; i<numElemReg; ++i) {
      int elem = regElemList[i];
      e_old[i] = e[elem] ;
      delvc[i] = delv[elem] ;
      p_old[i] = p[elem] ;
      q_old[i] = q[elem] ;
      qq_old[i] = qq[elem] ;
      ql_old[i] = ql[elem] ;
    }

#ifdef _OPENACC
#pragma acc parallel loop present(vnewc[numElem], \
                                  compression[numElemReg], \
                                  delvc[numElemReg], \
                                  compHalfStep[numElemReg], \
                                  regElemList[numElemReg])
#elif defined(_OPENMP)
#pragma omp for
#endif
    for (i = 0; i < numElemReg ; ++i) {
      int elem = regElemList[i];
      Real_t vchalf ;
      compression[i] = (Real_t)(1.) / vnewc[elem] - (Real_t)(1.);
      vchalf = vnewc[elem] - delvc[i] * (Real_t)(.5);
      compHalfStep[i] = (Real_t)(1.) / vchalf - (Real_t)(1.);
    }

// Fused some loops here to reduce overhead of repeatedly calling small kernels
#ifdef _OPENACC
#pragma acc parallel loop present(vnewc[numElem], \
                                  compHalfStep[numElemReg], \
                                  compression[numElemReg], \
                                  regElemList[numElemReg], \
                                  p_old[numElemReg], \
                                  compHalfStep[numElemReg], \
                                  work[numElemReg])
#elif defined(_OPENMP)
#pragma omp for
#endif
    for(i = 0; i < numElemReg; ++i) {
      int elem = regElemList[i];
      if (eosvmin != 0.0 && vnewc[elem] <= eosvmin)  { /* impossible due to calling func? */
        compHalfStep[i] = compression[i] ;
      }
      if (eosvmax != 0.0 && vnewc[elem] >= eosvmax) { /* impossible due to calling func? */
        p_old[i]        = (Real_t)(0.) ;
        compression[i]  = (Real_t)(0.) ;
        compHalfStep[i] = (Real_t)(0.) ;
      }
      work[i] = (Real_t)(0.) ; 
    }
//} // end omp parallel

    CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc,
        p_old, e_old,  q_old, compression, compHalfStep,
        vnewc, work,  delvc, pmin,
        p_cut, e_cut, q_cut, emin,
        qq_old, ql_old, rho0, eosvmax,
        numElemReg, regElemList);
} // end foreach repetition

#ifdef _OPENACC
#pragma acc parallel loop present(p_new[numElemReg], \
                                  e_new[numElemReg], \
                                  q_new[numElemReg], \
                                  p[numElem], \
                                  e[numElem], \
                                  q[numElem])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(numElemReg)
#endif
  for (i=0; i<numElemReg; ++i) {
    int elem = regElemList[i];
    p[elem] = p_new[i] ;
    e[elem] = e_new[i] ;
    q[elem] = q_new[i] ;
  }

  Real_t *ss = m_ss;
  CalcSoundSpeedForElems(ss,
      vnewc, rho0, e_new, p_new,
      pbvc, bvc, ss4o3,
      numElem, numElemReg, regElemList) ;
} // end acc data

}

/******************************************/

static inline
void ApplyMaterialPropertiesForElems(Real_t vnew[])
{
  Index_t numElem = m_numElem ;

  Index_t i;

  if (numElem != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = m_eosvmin ;
    Real_t eosvmax = m_eosvmax ;

#ifdef _OPENACC
#pragma acc data present(vnew[numElem])
#elif defined(_OPENMP)
#pragma omp parallel firstprivate(numElem)
#endif
    {
      // Bound the updated relative volumes with eosvmin/max
      if (eosvmin != (Real_t)(0.)) {
#ifdef _OPENACC
#pragma acc parallel loop
#elif defined(_OPENMP)
#pragma omp for
#endif
        for(i=0 ; i<numElem ; ++i) {
          if (vnew[i] < eosvmin)
            vnew[i] = eosvmin ;
        }
      }

      if (eosvmax != (Real_t)(0.)) {
#ifdef _OPENACC
#pragma acc parallel loop
#elif defined(_OPENMP)
#pragma omp for nowait
#endif
        for(i=0 ; i<numElem ; ++i) {
          if (vnew[i] > eosvmax)
            vnew[i] = eosvmax ;
        }
      }

      // This check may not make perfect sense in LULESH, but
      // it's representative of something in the full code -
      // just leave it in, please
      Real_t *v = m_v;
      Real_t vc = 1.;
#ifdef _OPENACC
#pragma acc parallel loop reduction(min: vc) \
                          present(v[numElem])
#elif defined(_OPENMP)
#pragma omp for nowait private(vc) reduction(min: vc)
#endif
      for (i=0; i<numElem; ++i) {
        vc = v[i];
        if (eosvmin != (Real_t)(0.)) {
          if (vc < eosvmin)
            vc = eosvmin ;
        }
        if (eosvmax != (Real_t)(0.)) {
          if (vc > eosvmax)
            vc = eosvmax ;
        }
      }

      if (vc <= 0.) {
#if USE_MPI             
        MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
        exit(VolumeError);
#endif
      }
    } // end acc data

    Int_t r;

    for (r=0 ; r<m_numReg ; r++) {
      int numElemReg = m_regElemSize[r];
      int *regElemList = m_regElemlist[r];
      Int_t rep;
      //Determine load imbalance for this region
      //round down the number with lowest cost
      if(r < m_numReg/2)
        rep = 1;
      //you don't get an expensive region unless you at least have 5 regions
      else if(r < (m_numReg - (m_numReg+15)/20))
        rep = 1 + m_cost;
      //very expensive regions
      else
        rep = 10 * (1+ m_cost);
      //[DEBUG by Seyong Lee] If-statement is added to invoke EvalEOSForElems()
      //only if numElemReg > 0.
      if( numElemReg > 0 ) {
        EvalEOSForElems(vnew, numElemReg, regElemList, rep);
      }
    }

  }
}

/******************************************/

static inline
void UpdateVolumesForElems(Real_t *vnew, Real_t *v,
                           Real_t v_cut, Index_t length)
{
  if (length != 0) {
    Index_t i;
#ifdef _OPENACC
#pragma acc parallel loop present(vnew[length], \
                                  v[length])
#elif defined(_OPENMP)
#pragma omp parallel for firstprivate(length, v_cut)
#endif
    for(i=0 ; i<length ; ++i) {
      Real_t tmpV = vnew[i] ;

      if ( fabs(tmpV - (Real_t)(1.0)) < v_cut )
        tmpV = (Real_t)(1.0) ;

      v[i] = tmpV ;
    }
  }

  return ;
}

/******************************************/

static inline
void LagrangeElements(Index_t numElem)
{
  Real_t *vnew = m_vnew;  /* new relative vol -- temp */

  CalcLagrangeElements(vnew) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(vnew) ;

  ApplyMaterialPropertiesForElems(vnew) ;

  UpdateVolumesForElems(vnew, m_v,
                        m_v_cut, numElem) ;
}

/******************************************/

static inline
void CalcCourantConstraintForElems(Int_t length,
                                   Index_t *regElemlist, Real_t *ss,
                                   Real_t *vdov, Real_t *arealg,
                                   Real_t qqc,
                                   Real_t* dtcourant, Index_t numElem)
{
#if !defined(_OPENACC) && defined(_OPENMP)
  Index_t threads = omp_get_max_threads();
  static Index_t *courant_elem_per_thread;
  static Real_t *dtcourant_per_thread;
  static bool first = true;
  if (first) {
    courant_elem_per_thread = (Index_t*) calloc(threads, sizeof(Index_t));
    dtcourant_per_thread = (Real_t*) calloc(threads, sizeof(Real_t));
    first = false;
  }
#else
  Index_t threads = 1;
  Index_t courant_elem_per_thread[1];
  Real_t  dtcourant_per_thread[1];
#endif

  Index_t i;

#if !defined(_OPENACC) && defined(_OPENMP)
#pragma omp parallel firstprivate(length, qqc)
#endif
  {
    Real_t   qqc2 = (Real_t)(64.0) * qqc * qqc ;
    Real_t   dtcourant_tmp = *dtcourant;
    Index_t  courant_elem  = -1 ;

#if !defined(_OPENACC) && defined(_OPENMP)
    Index_t thread_num = omp_get_thread_num();
#else
    Index_t thread_num = 0;
#endif      

#if !defined(_OPENACC) && defined(_OPENMP)
#pragma omp for 
#endif
    for (i = 0 ; i < length ; ++i) {
      Index_t indx = regElemlist[i] ;
      Real_t dtf = ss[indx] * ss[indx] ;

      if ( vdov[indx] < (Real_t)(0.) ) {
        dtf = dtf
          + qqc2 * arealg[indx] * arealg[indx]
          * vdov[indx] * vdov[indx] ;
      }

      dtf = SQRT(dtf) ;
      dtf = arealg[indx] / dtf ;

      if (vdov[indx] != (Real_t)(0.)) {
        if ( dtf < dtcourant_tmp ) {
          dtcourant_tmp = dtf ;
          courant_elem  = indx ;
        }
      }
    }

    dtcourant_per_thread[thread_num]    = dtcourant_tmp ;
    courant_elem_per_thread[thread_num] = courant_elem ;
  }

  for (i = 1; i < threads; ++i) {
    if (dtcourant_per_thread[i] < dtcourant_per_thread[0] ) {
      dtcourant_per_thread[0]    = dtcourant_per_thread[i];
      courant_elem_per_thread[0] = courant_elem_per_thread[i];
    }
  }

  if (courant_elem_per_thread[0] != -1) {
    *dtcourant = dtcourant_per_thread[0] ;
  }

  return ;

}

/******************************************/

static inline
void CalcHydroConstraintForElems(Int_t length,
                                 Index_t *regElemlist, Real_t *vdov,
                                 Real_t dvovmax,
                                 Real_t* dthydro, Index_t numElem)
{
  /* ACC: vdov was updated in CalcCourantConstraintForElems so we don't need to
     update it again. */

#if !defined(_OPENACC) && defined(_OPENMP)
  Index_t threads = omp_get_max_threads();
  static Index_t *hydro_elem_per_thread;
  static Real_t *dthydro_per_thread;
  static bool first = true;
  if (first) {
    hydro_elem_per_thread = (Index_t*) calloc(threads, sizeof(Index_t));
    dthydro_per_thread = (Real_t*) calloc(threads, sizeof(Real_t));
    first = false;
  }
#else
  Index_t threads = 1;
  Index_t hydro_elem_per_thread[1];
  Real_t  dthydro_per_thread[1];
#endif

  Index_t i;

#if !defined(_OPENACC) && defined(_OPENMP)
#pragma omp parallel firstprivate(length, dvovmax)
#endif
  {
    Real_t dthydro_tmp = *dthydro ;
    Index_t hydro_elem = -1 ;

#if !defined(_OPENACC) && defined(_OPENMP)
    Index_t thread_num = omp_get_thread_num();
#else      
    Index_t thread_num = 0;
#endif      

#if !defined(_OPENACC) && defined(_OPENMP)
#pragma omp for
#endif
    for (i = 0 ; i < length ; ++i) {
      Index_t indx = regElemlist[i] ;

      if (vdov[indx] != (Real_t)(0.)) {
        Real_t dtdvov = dvovmax / (FABS(vdov[indx])+(Real_t)(1.e-20)) ;

        if ( dthydro_tmp > dtdvov ) {
          dthydro_tmp = dtdvov ;
          hydro_elem = indx ;
        }
      }
    }

    dthydro_per_thread[thread_num]    = dthydro_tmp ;
    hydro_elem_per_thread[thread_num] = hydro_elem ;
  }

  for (i = 1; i < threads; ++i) {
    if(dthydro_per_thread[i] < dthydro_per_thread[0]) {
      dthydro_per_thread[0]    = dthydro_per_thread[i];
      hydro_elem_per_thread[0] =  hydro_elem_per_thread[i];
    }
  }

  if (hydro_elem_per_thread[0] != -1) {
    *dthydro =  dthydro_per_thread[0] ;
  }

  return ;
}

/******************************************/

static inline
void CalcTimeConstraintsForElems() {

  // Initialize conditions to a very large value
  m_dtcourant = 1.0e+20;
  m_dthydro = 1.0e+20;

  Index_t r;
  /* wait for async mem updates to finish */
#pragma acc wait

  for (r=0 ; r < m_numReg ; ++r) {
    /* evaluate time constraint */
    CalcCourantConstraintForElems(m_regElemSize[r],
                                  m_regElemlist[r], m_ss,
                                  m_vdov, m_arealg,
                                  m_qqc,
                                  &m_dtcourant, m_numElem) ;

    /* check hydro constraint */
    CalcHydroConstraintForElems(m_regElemSize[r],
                                m_regElemlist[r], m_vdov,
                                m_dvovmax,
                                &m_dthydro, m_numElem);
  }
}

/******************************************/

static inline
void LagrangeLeapFrog()
{
  Index_t numElem = m_numElem;

#ifdef SEDOV_SYNC_POS_VEL_LATE
  Real_t *fieldData[6] ;

/* wait for async device update to complete */
#pragma acc wait
#endif
  //[DEBUG by Seyong Lee] Below definitions are moved out of
  //the above #ifdef macro region.
  volatile Index_t numNode = m_numNode;
  Real_t *x = m_x;
  Real_t *y = m_y;
  Real_t *z = m_z;
  Real_t *xd = m_xd;
  Real_t *yd = m_yd;
  Real_t *zd = m_zd;

  /* calculate nodal forces, accelerations, velocities, positions, with
   * applied boundary conditions and slide surface considerations */
  LagrangeNodal();

#pragma acc data present(x[numNode], \
                         y[numNode], \
                         z[numNode], \
                         xd[numNode], \
                         yd[numNode], \
                         zd[numNode])
{
#if USE_MPI   

#ifdef SEDOV_SYNC_POS_VEL_EARLY
/* wait for async device update to complete (in LagrangeNodal) */
#pragma acc wait
#endif

#ifdef SEDOV_SYNC_POS_VEL_LATE

  /* asynchronously update on host before MPI comm */
  volatile int up = 1;
#pragma acc update host(x[numNode], \
                        y[numNode], \
                        z[numNode], \
                        xd[numNode], \
                        yd[numNode], \
                        zd[numNode]) \
                   async(up)
#endif
#endif

  /* calculate element quantities (i.e. velocity gradient & q), and update
   * material states */
  LagrangeElements(numElem);

// update values for CalcTimeConstraintsForElems as early as possible
#ifdef _OPENACC
  Real_t *ss = m_ss;
  Real_t *vdov = m_vdov;
  Real_t *arealg = m_arealg;
#pragma acc data present(ss[numElem], \
                         vdov[numElem], \
                         arealg[numElem])
{
#pragma acc update host(ss[numElem], \
                        vdov[numElem], \
                        arealg[numElem]) \
                   async
}
#endif


#if USE_MPI   
#ifdef SEDOV_SYNC_POS_VEL_LATE

#pragma acc wait(up)

  CommRecv(MSG_SYNC_POS_VEL, 6,
      m_sizeX + 1, m_sizeY + 1, m_sizeZ + 1,
      false, false) ;

  fieldData[0] = x;
  fieldData[1] = y;
  fieldData[2] = z;
  fieldData[3] = xd;
  fieldData[4] = yd;
  fieldData[5] = zd;

  CommSend(MSG_SYNC_POS_VEL, 6, fieldData,
           m_sizeX + 1, m_sizeY + 1, m_sizeZ + 1,
           false, false) ;
#endif
#endif   

  CalcTimeConstraintsForElems();

#if USE_MPI   
#ifdef SEDOV_SYNC_POS_VEL_LATE
  CommSyncPosVel() ;
#pragma acc update device(x[numNode], \
                          y[numNode], \
                          z[numNode], \
                          xd[numNode], \
                          yd[numNode], \
                          zd[numNode]) \
                   async
#endif
#endif   
} // end acc data
}


/******************************************/

int main(int argc, char *argv[])
{
  int numRanks ;
  int myRank ;
  struct cmdLineOpts opts;
#if USE_MPI
  char name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
#endif

#if USE_MPI   
  Real_t *nodalMass;
   
  MPI_Init(&argc, &argv) ;
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

  MPI_Get_processor_name(name, &name_len);

  printf("[%s:%d] [%d/%d] [%s]\n", __FILE__, __LINE__, myRank, numRanks, name);
#else
  numRanks = 1;
  myRank = 0;
#endif   

#if LULESH_DUMP_OUTPUT
  FILE *fp;
  int i;
#endif

  /* Set defaults that can be overridden by command line opts */
  opts.its = 9999999;
  opts.nx  = 30;
  opts.numReg = 11;
  opts.numFiles = (int)(numRanks+10)/9;
  opts.showProg = 0;
  opts.quiet = 0;
  opts.viz = 0;
  opts.balance = 1;
  opts.cost = 1;

  ParseCommandLineOptions(argc, argv, myRank, &opts);

  if ((myRank == 0) && (opts.quiet == 0)) {
    printf("Running problem size %d^3 per domain until completion\n", opts.nx);
    printf("Num processors: %d\n", numRanks);
#if !defined(_OPENACC) && defined(_OPENMP)
    printf("Num threads: %d\n", omp_get_max_threads());
#endif
    printf("Total number of elements: %d\n\n", numRanks*opts.nx*opts.nx*opts.nx);
    printf("To run other sizes, use -s <integer>.\n");
    printf("To run a fixed number of iterations, use -i <integer>.\n");
    printf("To run a more or less balanced region set, use -b <integer>.\n");
    printf("To change the relative costs of regions, use -c <integer>.\n");
    printf("To print out progress, use -p\n");
    printf("To write an output file for VisIt, use -v\n");
    printf("See help (-h) for more options\n\n");
  }

  // Set up the mesh and decompose. Assumes regular cubes for now
  Int_t col, row, plane, side;
  InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

  // Build the main data structure and initialize it
  Domain(numRanks, col, row, plane, opts.nx,
      side, opts.numReg, opts.balance, opts.cost) ;


#if USE_MPI   
  nodalMass = m_nodalMass;

  // Initial domain boundary communication 
  CommRecv(MSG_COMM_SBN, 1,
           m_sizeX + 1, m_sizeY + 1, m_sizeZ + 1,
           true, false) ;
  CommSend(MSG_COMM_SBN, 1, &nodalMass,
           m_sizeX + 1, m_sizeY + 1, m_sizeZ +  1,
           true, false) ;
  CommSBN(1, &nodalMass) ;


  // End initialization
  MPI_Barrier(MPI_COMM_WORLD);
#endif   
   
  // BEGIN timestep to solution */
  Real_t start;
#if USE_MPI   
  start = MPI_Wtime();
#else
  start = clock();
#endif
  
  /* tmp region-based arrays */
  int maxRegSize = 0;
  Int_t r;
  for (r=0 ; r < m_numReg ; r++) {
    maxRegSize = MAX(maxRegSize, m_regElemSize[r]);
  }
  AllocateRegionTmps(maxRegSize);

#ifdef _OPENACC
  Index_t numElem = m_numElem;
  Index_t numElem8 = numElem * 8;
  Index_t numNode = m_numNode;
  Index_t size = m_sizeX;
  Index_t numNodeBC = (size+1)*(size+1) ;
  Index_t allElem = numElem +  /* local elem */
    2*m_sizeX*m_sizeY + /* plane ghosts */
    2*m_sizeX*m_sizeZ + /* row ghosts */
    2*m_sizeY*m_sizeZ ; /* col ghosts */

  Real_t *fx = m_fx;
  Real_t *fy = m_fy;
  Real_t *fz = m_fz;

  // load tmp arrays
  Real_t *fx_elem = m_fx_elem;
  Real_t *fy_elem = m_fy_elem;
  Real_t *fz_elem = m_fz_elem;
  Real_t *dvdx = m_dvdx;
  Real_t *dvdy = m_dvdy;
  Real_t *dvdz = m_dvdz;
  Real_t *x8n = m_x8n;
  Real_t *y8n = m_y8n;
  Real_t *z8n = m_z8n;
  Real_t *sigxx = m_sigxx;
  Real_t *sigyy = m_sigyy;
  Real_t *sigzz = m_sigzz;
  Real_t *determ = m_determ;
  Real_t *dxx = m_dxx;
  Real_t *dyy = m_dyy;
  Real_t *dzz = m_dzz;
  Real_t *vnew = m_vnew;
  Real_t *delv_xi = m_delv_xi;
  Real_t *delv_eta = m_delv_eta;
  Real_t *delv_zeta = m_delv_zeta;
  Real_t *delx_xi = m_delx_xi;
  Real_t *delx_eta = m_delx_eta;
  Real_t *delx_zeta = m_delx_zeta;
  Real_t *e_old = m_e_old ;
  Real_t *delvc = m_delvc ;
  Real_t *p_old = m_p_old ;
  Real_t *q_old = m_q_old ;
  Real_t *compression = m_compression ;
  Real_t *compHalfStep = m_compHalfStep ;
  Real_t *qq_old = m_qq_old ;
  Real_t *ql_old = m_ql_old ;
  Real_t *work = m_work ;
  Real_t *p_new = m_p_new ;
  Real_t *e_new = m_e_new ;
  Real_t *q_new = m_q_new ;
  Real_t *bvc = m_bvc ;
  Real_t *pbvc = m_pbvc ;

  Real_t *x = m_x;
  Real_t *y = m_y;
  Real_t *z = m_z;

  Real_t *xd = m_xd;
  Real_t *yd = m_yd;
  Real_t *zd = m_zd;

  Real_t *xdd = m_xdd;
  Real_t *ydd = m_ydd;
  Real_t *zdd = m_zdd;

  Real_t *v = m_v;
  Real_t *volo = m_volo;
  Real_t *delv = m_delv;
  Real_t *vdov = m_vdov;
  Real_t *arealg = m_arealg;
  
#if !USE_MPI
  /* nodalMass already defined if USE_MPI */
  Real_t *nodalMass = m_nodalMass;
#endif
  Real_t *elemMass = m_elemMass;
  Real_t *ss = m_ss;

  Index_t *lxim = m_lxim;
  Index_t *lxip = m_lxip;
  Index_t *letam = m_letam;
  Index_t *letap = m_letap;
  Index_t *lzetam = m_lzetam;
  Index_t *lzetap = m_lzetap;

  Real_t *p = m_p;
  Real_t *e = m_e;
  Real_t *q = m_q;
  Real_t *qq = m_qq;
  Real_t *ql = m_ql;

  Index_t *symmX = m_symmX;
  Index_t *symmY = m_symmY;
  Index_t *symmZ = m_symmZ;

  Index_t *nodelist = m_nodelist;
  Index_t *nodeElemCount = m_nodeElemCount;
  Index_t *nodeElemStart = m_nodeElemStart;
  Index_t *nodeElemCornerList = m_nodeElemCornerList;
  Index_t *elemBC = m_elemBC;

  Index_t nCorner = nodeElemStart[numNode-1] + nodeElemCount[numNode-1];

  /* Since these are only found in pragmas they'll be optimized out -- this
     forces them to remain. If we instead switch all of these pointers to
     volatile some crashes continue happening, so this seems to work best
     for now. */
  volatile Index_t dummyI = nodelist[numElem8-1] + nodeElemCount[numNode-1] +
                            nodeElemStart[numNode-1] + nodeElemCornerList[nCorner-1] +
                            lxim[numElem-1] + lxip[numElem-1] + letam[numElem-1] +
                            letap[numElem-1] + lzetam[numElem-1] + lzetap[numElem-1] +
                            elemBC[numElem-1];
  if(!m_symmXempty)
    dummyI += symmX[numNodeBC-1];
  if(!m_symmYempty)
    dummyI += symmY[numNodeBC-1];
  if(!m_symmZempty)
    dummyI += symmZ[numNodeBC-1];

  volatile Real_t dummyR = x[numNode-1] + y[numNode-1] + z[numNode-1] +
                           xd[numNode-1] + yd[numNode-1] + zd[numNode-1] +
                           xdd[numNode-1] + ydd[numNode-1] + zdd[numNode-1] +
                           fx[numNode-1] + fy[numNode-1] + fz[numNode-1] +
                           fx_elem[numElem8-1] + fy_elem[numElem8-1] + fz_elem[numElem8-1] +
                           dvdx[numElem8-1] + dvdy[numElem8-1] + dvdz[numElem8-1] +
                           x8n[numElem8-1] + y8n[numElem8-1] + z8n[numElem8-1] +
                           sigxx[numElem-1] + sigyy[numElem-1] + sigzz[numElem-1] +
                           dxx[numElem-1] + dyy[numElem-1] + dzz[numElem-1] +
                           determ[numElem-1] + vnew[numElem-1] +
                           delv_xi[allElem-1] + delv_xi[allElem-1] + delv_eta[allElem-1] +
                           delv_zeta[allElem-1] + delx_xi[allElem-1] + delx_eta[allElem-1] +
                           delx_zeta[allElem-1] +
                           v[numElem-1] + volo[numElem-1] + delv[numElem-1] +
                           arealg[numElem-1] + vdov[numElem-1] + ss[numElem-1] +
                           p[numElem-1] + e[numElem-1] + q[numElem-1] +
                           qq[numElem-1] + ql[numElem-1] +
                           elemMass[numElem-1] + nodalMass[numNode-1] +
                           e_old[maxRegSize-1] + delvc[maxRegSize-1] + 
                           p_old[maxRegSize-1] + q_old[maxRegSize-1] + 
                           compression[maxRegSize-1] + compHalfStep[maxRegSize-1] + 
                           qq_old[maxRegSize-1] + ql_old[maxRegSize-1] + 
                           work[maxRegSize-1] + p_new[maxRegSize-1] + 
                           e_new[maxRegSize-1] + q_new[maxRegSize-1] + 
                           bvc[maxRegSize-1] + pbvc[maxRegSize-1];

  if(myRank == 0) {
    printf("Copying data to device...");
    fflush(stdout);
  }

#pragma acc data create(fx[numNode], \
                        fy[numNode], \
                        fz[numNode], \
                        fx_elem[numElem8], \
                        fy_elem[numElem8], \
                        fz_elem[numElem8], \
                        dvdx[numElem8], \
                        dvdy[numElem8], \
                        dvdz[numElem8], \
                        x8n[numElem8], \
                        y8n[numElem8], \
                        z8n[numElem8], \
                        sigxx[numElem], \
                        sigyy[numElem], \
                        sigzz[numElem], \
                        determ[numElem], \
                        dxx[numElem], \
                        dyy[numElem], \
                        dzz[numElem], \
                        vnew[numElem], \
                        delx_xi[allElem], \
                        delx_eta[allElem], \
                        delx_zeta[allElem], \
                        delv_xi[allElem], \
                        delv_eta[allElem], \
                        delv_zeta[allElem], \
                        e_old[maxRegSize],  \
                        delvc[maxRegSize],  \
                        p_old[maxRegSize],  \
                        q_old[maxRegSize],  \
                        compression[maxRegSize],  \
                        compHalfStep[maxRegSize],  \
                        qq_old[maxRegSize],  \
                        ql_old[maxRegSize],  \
                        work[maxRegSize],  \
                        p_new[maxRegSize],  \
                        e_new[maxRegSize],  \
                        q_new[maxRegSize],  \
                        bvc[maxRegSize],  \
                        pbvc[maxRegSize]) \
                 copy(x[numNode], \
                      y[numNode], \
                      z[numNode], \
                      xd[numNode], \
                      yd[numNode], \
                      zd[numNode], \
                      p[numElem], \
                      e[numElem]) \
                 copyin(symmX[numNodeBC], \
                        symmY[numNodeBC], \
                        symmZ[numNodeBC], \
                        xdd[numNode], \
                        ydd[numNode], \
                        zdd[numNode], \
                        v[numElem], \
                        volo[numElem], \
                        delv[numElem], \
                        arealg[numElem], \
                        vdov[numElem], \
                        ss[numElem],       \
                        q[numElem], \
                        qq[numElem], \
                        ql[numElem], \
                        nodalMass[numNode], \
                        elemMass[numElem], \
                        lxim[numElem], \
                        lxip[numElem], \
                        letam[numElem], \
                        letap[numElem], \
                        lzetam[numElem], \
                        lzetap[numElem], \
                        nodelist[numElem8], \
                        nodeElemCount[numNode], \
                        nodeElemStart[numNode], \
                        nodeElemCornerList[nCorner], \
                        elemBC[numElem])
#endif
{
#ifdef _OPENACC
  if(myRank == 0) {
    printf("done.\n");
    fflush(stdout);
  }
#endif
  while((m_time < m_stoptime) && (m_cycle < opts.its)) {
    TimeIncrement() ;
    LagrangeLeapFrog() ;

    if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
      printf("cycle = %d, time = %e, dt=%e\n",
          m_cycle, (double)(m_time), (double)(m_deltatime) ) ;
    }
  }
} // end acc data

  // Use reduced max elapsed time
  Real_t elapsed_time;
#if USE_MPI   
  elapsed_time = MPI_Wtime() - start;
#else
  elapsed_time = (clock() - start) / CLOCKS_PER_SEC;
#endif
  double elapsed_timeG;
#if USE_MPI   
  MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
  elapsed_timeG = elapsed_time;
#endif

#if LULESH_DUMP_OUTPUT
  fp = fopen("lulesh.dump", "w");
  for (i = 0; i < numNode; i++) fprintf(fp, "%.6f\n", x[i]);
  for (i = 0; i < numNode; i++) fprintf(fp, "%.6f\n", y[i]);
  for (i = 0; i < numNode; i++) fprintf(fp, "%.6f\n", z[i]);
  fclose(fp);
#endif

  if ((myRank == 0) && (opts.quiet == 0)) {
    VerifyAndWriteFinalOutput(elapsed_timeG, opts.nx, numRanks);
  }

#if USE_MPI
  MPI_Finalize() ;
#endif

  // OpenACC - release device ptrs
  ReleaseDeviceMem();

  return 0 ;
}
