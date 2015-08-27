#ifndef _LULESH_H_
#define _LULESH_H_

#if !defined(USE_MPI)
# error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif

// OpenMP will be compiled in if this flag is set to 1 AND the compiler beging
// used supports it (i.e. the _OPENMP symbol is defined)
//#define USE_OMP 1

#if USE_MPI
#include <mpi.h>

/*
  define one of these three symbols:

  SEDOV_SYNC_POS_VEL_NONE
  SEDOV_SYNC_POS_VEL_EARLY
  SEDOV_SYNC_POS_VEL_LATE
*/

#define SEDOV_SYNC_POS_VEL_LATE 1
#endif // if USE_MPI

#ifdef _OPENACC
#include "openacc.h"
#endif

#include <math.h>

//**************************************************
// Allow flexibility for arithmetic representations 
//**************************************************

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

#define bool    int
#define false   0
#define true    1

// Could also support fixed point and interval arithmetic types
typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;  // 10 bytes on x86

typedef int    Index_t ; // array subscript and loop index
typedef real8  Real_t ;  // floating point representation
typedef int    Int_t ;   // integer representation

enum { VolumeError = -1, QStopError = -2 } ;

inline real8  SQRT(real8  arg) { return sqrt(arg) ; }

inline real8  CBRT(real8  arg) { return cbrt(arg) ; }

inline real8  FABS(real8  arg) { return fabs(arg) ; }

// Stuff needed for boundary conditions
// 2 BCs on each of 6 hexahedral faces (12 bits)
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
  (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

//////////////////////////////////////////////////////
// Primary data structure
//////////////////////////////////////////////////////

//class Domain {

//  public:

  // Constructor
  void Domain(Int_t numRanks, Index_t colLoc,
      Index_t rowLoc, Index_t planeLoc,
      Index_t nx, int tp, int nr, int balance, int cost);

  void AllocateRegionTmps(Int_t numElem);
  void AllocateGradients(Int_t numElem);
  void DeallocateGradients();

  // Node-centered

#if USE_MPI   
  // Communication Work space 
  extern Real_t *commDataSend ;
  extern Real_t *commDataRecv ;

  // Maximum number of block neighbors 
  extern MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners 
  extern MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners 
#endif

  // OpenACC
  void ReleaseDeviceMem();
  extern int m_numDevs;

  void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
  void SetupThreadSupportStructures();
  void CreateRegionIndexSets(Int_t nreg, Int_t balance);
  void SetupCommBuffers(Int_t edgeNodes);
  void SetupSymmetryPlanes(Int_t edgeNodes);
  void SetupElementConnectivities(Int_t edgeElems);
  void SetupBoundaryConditions(Int_t edgeElems);

  /* Node-centered */
  extern Real_t* m_x ;  /* coordinates */
  extern Real_t* m_y ;
  extern Real_t* m_z ;

  extern Real_t* m_xd ; /* velocities */
  extern Real_t* m_yd ;
  extern Real_t* m_zd ;

  extern Real_t* m_xdd ; /* accelerations */
  extern Real_t* m_ydd ;
  extern Real_t* m_zdd ;

  extern Real_t* m_fx ;  /* forces */
  extern Real_t* m_fy ;
  extern Real_t* m_fz ;

  /* tmp arrays that are allocated globally for OpenACC */
  extern Real_t* m_fx_elem ;
  extern Real_t* m_fy_elem ;
  extern Real_t* m_fz_elem ;
  extern Real_t* m_dvdx ;
  extern Real_t* m_dvdy ;
  extern Real_t* m_dvdz ;
  extern Real_t* m_x8n ;
  extern Real_t* m_y8n ;
  extern Real_t* m_z8n ;
  extern Real_t* m_sigxx ;
  extern Real_t* m_sigyy ;
  extern Real_t* m_sigzz ;
  extern Real_t* m_determ ;
  extern Real_t* m_e_old ;
  extern Real_t* m_delvc ;
  extern Real_t* m_p_old ;
  extern Real_t* m_q_old ;
  extern Real_t* m_compression ;
  extern Real_t* m_compHalfStep ;
  extern Real_t* m_qq_old ;
  extern Real_t* m_ql_old ;
  extern Real_t* m_work ;
  extern Real_t* m_p_new ;
  extern Real_t* m_e_new ;
  extern Real_t* m_q_new ;
  extern Real_t* m_bvc ;
  extern Real_t* m_pbvc ;

  extern Real_t* m_nodalMass ;  /* mass */

  extern Index_t* m_symmX;  /* symmetry plane nodesets */
  extern Index_t* m_symmY;
  extern Index_t* m_symmZ;

  extern bool m_symmXempty;
  extern bool m_symmYempty;
  extern bool m_symmZempty;

  // Element-centered

  // Region information
  extern Int_t    m_numReg ;
  extern Int_t    m_cost; //imbalance cost
  extern Int_t   *m_regElemSize ;   // Size of region sets
  extern Index_t *m_regNumList ;    // Region number per domain element
  extern Index_t **m_regElemlist ;  // region indexset 

  extern Index_t*  m_matElemlist ;  /* material indexset */
  extern Index_t*  m_nodelist ;     /* elemToNode connectivity */

  extern Index_t*  m_lxim ;  /* element connectivity across each face */
  extern Index_t*  m_lxip ;
  extern Index_t*  m_letam ;
  extern Index_t*  m_letap ;
  extern Index_t*  m_lzetam ;
  extern Index_t*  m_lzetap ;

  extern Int_t*    m_elemBC ;  /* symmetry/free-surface flags for each elem face */

  extern Real_t* m_dxx ;  /* principal strains -- temporary */
  extern Real_t* m_dyy ;
  extern Real_t* m_dzz ;

  extern Real_t* m_delv_xi ;    /* velocity gradient -- temporary */
  extern Real_t* m_delv_eta ;
  extern Real_t* m_delv_zeta ;

  extern Real_t* m_delx_xi ;    /* coordinate gradient -- temporary */
  extern Real_t* m_delx_eta ;
  extern Real_t* m_delx_zeta ;

  extern Real_t* m_e ;   /* energy */

  extern Real_t* m_p ;   /* pressure */
  extern Real_t* m_q ;   /* q */
  extern Real_t* m_ql ;  /* linear term for q */
  extern Real_t* m_qq ;  /* quadratic term for q */

  extern Real_t* m_v ;     /* relative volume */
  extern Real_t* m_volo ;  /* reference volume */
  extern Real_t* m_vnew ;  /* new relative volume -- temporary */
  extern Real_t* m_delv ;  /* m_vnew - m_v */
  extern Real_t* m_vdov ;  /* volume derivative over volume */

  extern Real_t* m_arealg ;  /* characteristic length of an element */

  extern Real_t* m_ss ;      /* "sound speed" */

  extern Real_t* m_elemMass ;  /* mass */

  // Cutoffs (treat as constants)
  extern Real_t  m_e_cut ;             // energy tolerance 
  extern Real_t  m_p_cut ;             // pressure tolerance 
  extern Real_t  m_q_cut ;             // q tolerance 
  extern Real_t  m_v_cut ;             // relative volume tolerance 
  extern Real_t  m_u_cut ;             // velocity tolerance 

  // Other constants (usually setable, but hardcoded in this proxy app)

  extern Real_t  m_hgcoef ;            // hourglass control 
  extern Real_t  m_ss4o3 ;
  extern Real_t  m_qstop ;             // excessive q indicator 
  extern Real_t  m_monoq_max_slope ;
  extern Real_t  m_monoq_limiter_mult ;
  extern Real_t  m_qlc_monoq ;         // linear term coef for q 
  extern Real_t  m_qqc_monoq ;         // quadratic term coef for q 
  extern Real_t  m_qqc ;
  extern Real_t  m_eosvmax ;
  extern Real_t  m_eosvmin ;
  extern Real_t  m_pmin ;              // pressure floor 
  extern Real_t  m_emin ;              // energy floor 
  extern Real_t  m_dvovmax ;           // maximum allowable volume change 
  extern Real_t  m_refdens ;           // reference density 

  // Variables to keep track of timestep, simulation time, and cycle
  extern Real_t  m_dtcourant ;         // courant constraint 
  extern Real_t  m_dthydro ;           // volume change constraint 
  extern Int_t   m_cycle ;             // iteration count for simulation 
  extern Real_t  m_dtfixed ;           // fixed time increment 
  extern Real_t  m_time ;              // current time 
  extern Real_t  m_deltatime ;         // variable time increment 
  extern Real_t  m_deltatimemultlb ;
  extern Real_t  m_deltatimemultub ;
  extern Real_t  m_dtmax ;             // maximum allowable time increment 
  extern Real_t  m_stoptime ;          // end time for simulation 


  extern Int_t   m_numRanks ;


  extern Index_t m_colLoc ;
  extern Index_t m_rowLoc ;
  extern Index_t m_planeLoc ;
  extern Index_t m_tp ;

  extern Index_t m_sizeX ;
  extern Index_t m_sizeY ;
  extern Index_t m_sizeZ ;
  extern Index_t m_numElem ;
  extern Index_t m_numNode ;

  extern Index_t m_maxPlaneSize ;
  extern Index_t m_maxEdgeSize ;

  // OMP hack 
  extern Index_t *m_nodeElemCount ;
  extern Index_t *m_nodeElemStart ;
  extern Index_t *m_nodeElemCornerList ;

  // Used in setup
  extern Index_t m_rowMin, m_rowMax;
  extern Index_t m_colMin, m_colMax;
  extern Index_t m_planeMin, m_planeMax ;

//} ; /* end of class Domain */

struct cmdLineOpts {
  int its; // -i 
  int nx;  // -s 
  int numReg; // -r 
  int numFiles; // -f
  int showProg; // -p
  int quiet; // -q
  int viz; // -v 
  Int_t cost; // -c
  int balance; // -b
};



// Function Prototypes

// lulesh-par
Real_t CalcElemVolume( const Real_t x[8],
    const Real_t y[8],
    const Real_t z[8]);

// lulesh-util
void ParseCommandLineOptions(int argc, char *argv[],
    int myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
    Int_t nx,
    Int_t numRanks);

// lulesh-viz
//void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

// lulesh-comm
void CommRecv(int msgType, Index_t xferFields,
    Index_t dx, Index_t dy, Index_t dz,
    bool doRecv, bool planeOnly);
void CommSend(int msgType,
    Index_t xferFields, Real_t **fieldData,
    Index_t dx, Index_t dy, Index_t dz,
    bool doSend, bool planeOnly);
void CommSBN(int xferFields, Real_t **fieldData);
void CommSyncPosVel();
void CommMonoQ();

// lulesh-init
void InitMeshDecomp(int numRanks, int myRank,
    int* col, int* row, int* plane, int* side);

#endif /* _LULESH_H_ */
