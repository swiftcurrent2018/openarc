#include <math.h>
#if USE_MPI
#include <mpi.h>
#endif
#if _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "lulesh.h"

/////////////////////////////////////////////////////////////////////
void Domain(Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, int nr, int balance, Int_t cost)
  {
  m_e_cut=(Real_t)(1.0e-7);
  m_p_cut=(Real_t)(1.0e-7);
  m_q_cut=(Real_t)(1.0e-7);
  m_v_cut=(Real_t)(1.0e-10);
  m_u_cut=(Real_t)(1.0e-7);
  m_hgcoef=(Real_t)(3.0);
  m_ss4o3=(Real_t)(4.0)/(Real_t)(3.0);
  m_qstop=(Real_t)(1.0e+12);
  m_monoq_max_slope=(Real_t)(1.0);
  m_monoq_limiter_mult=(Real_t)(2.0);
  m_qlc_monoq=(Real_t)(0.5);
  m_qqc_monoq=(Real_t)(2.0)/(Real_t)(3.0);
  m_qqc=(Real_t)(2.0);
  m_eosvmax=(Real_t)(1.0e+9);
  m_eosvmin=(Real_t)(1.0e-9);
  m_pmin=(Real_t)(0.);
  m_emin=(Real_t)(-1.0e+15);
  m_dvovmax=(Real_t)(0.1);
  m_refdens=(Real_t)(1.0);

  Index_t edgeElems = nx ;
  Index_t edgeNodes = edgeElems+1 ;
  m_cost = cost;

  m_tp       = tp ;
  m_numRanks = numRanks ;

  ///////////////////////////////
  //   Initialize Sedov Mesh
  ///////////////////////////////

  m_symmXempty = true;
  m_symmYempty = true;
  m_symmZempty = true;

  // construct a uniform box for this processor

  m_colLoc   =   colLoc ;
  m_rowLoc   =   rowLoc ;
  m_planeLoc = planeLoc ;

  m_sizeX = edgeElems ;
  m_sizeY = edgeElems ;
  m_sizeZ = edgeElems ;
  m_numElem = edgeElems*edgeElems*edgeElems ;

  m_numNode = edgeNodes*edgeNodes*edgeNodes ;

#ifdef USE_UNIFIEDMEM
  m_regNumList = (Index_t*) acc_create_unified(NULL, m_numElem * sizeof(Index_t)) ;  // material indexset
  memset(m_regNumList, 0,  m_numElem * sizeof(Index_t)) ;

  m_nodelist = (Index_t*) acc_create_unified(NULL, 8*m_numElem * sizeof(Index_t));
  memset(m_nodelist, 0, 8*m_numElem * sizeof(Index_t));

  // elem connectivities through face 
  m_lxim = (Index_t*) acc_create_unified(NULL, m_numElem * sizeof(Index_t));
  memset(m_lxim, 0, m_numElem * sizeof(Index_t));
  m_lxip = (Index_t*) acc_create_unified(NULL, m_numElem * sizeof(Index_t));
  memset(m_lxip, 0, m_numElem * sizeof(Index_t));
  m_letam = (Index_t*) acc_create_unified(NULL, m_numElem * sizeof(Index_t));
  memset(m_letam, 0, m_numElem * sizeof(Index_t));
  m_letap = (Index_t*) acc_create_unified(NULL, m_numElem * sizeof(Index_t));
  memset(m_letap, 0, m_numElem * sizeof(Index_t));
  m_lzetam = (Index_t*) acc_create_unified(NULL, m_numElem * sizeof(Index_t));
  memset(m_lzetam, 0, m_numElem * sizeof(Index_t));
  m_lzetap = (Index_t*) acc_create_unified(NULL, m_numElem * sizeof(Index_t));
  memset(m_lzetap, 0, m_numElem * sizeof(Index_t));

  m_elemBC = (Int_t*) acc_create_unified(NULL, m_numElem * sizeof(Int_t));
  memset(m_elemBC, 0, m_numElem * sizeof(Int_t));

  m_e = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_e, 0, m_numElem * sizeof(Real_t));
  m_p = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_p, 0, m_numElem * sizeof(Real_t));

  m_q = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_q, 0, m_numElem * sizeof(Real_t));
  m_ql = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_ql, 0, m_numElem * sizeof(Real_t));
  m_qq = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_qq, 0, m_numElem * sizeof(Real_t));

  m_v = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_v, 0, m_numElem * sizeof(Real_t));

  m_volo = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_volo, 0, m_numElem * sizeof(Real_t));
  m_delv = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_delv, 0, m_numElem * sizeof(Real_t));
  m_vdov = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_vdov, 0, m_numElem * sizeof(Real_t));

  m_arealg = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_arealg, 0, m_numElem * sizeof(Real_t));

  m_ss = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_ss, 0, m_numElem * sizeof(Real_t));

  m_elemMass = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_elemMass, 0, m_numElem * sizeof(Real_t));

  // Node-centered 

  m_x = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t)); // coordinates 
  memset(m_x, 0, m_numNode * sizeof(Real_t)); // coordinates 
  m_y = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));
  memset(m_y, 0, m_numNode * sizeof(Real_t));
  m_z = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));
  memset(m_z, 0, m_numNode * sizeof(Real_t));

  m_xd = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t)); // velocities 
  memset(m_xd, 0, m_numNode * sizeof(Real_t)); // velocities 
  m_yd = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));
  memset(m_yd, 0, m_numNode * sizeof(Real_t));
  m_zd = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));
  memset(m_zd, 0, m_numNode * sizeof(Real_t));

  m_xdd = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t)); // accelerations 
  memset(m_xdd, 0, m_numNode * sizeof(Real_t)); // accelerations 
  m_ydd = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));
  memset(m_ydd, 0, m_numNode * sizeof(Real_t));
  m_zdd = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));
  memset(m_zdd, 0, m_numNode * sizeof(Real_t));

  m_fx = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));  // forces 
  memset(m_fx, 0, m_numNode * sizeof(Real_t));  // forces 
  m_fy = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));
  memset(m_fy, 0, m_numNode * sizeof(Real_t));
  m_fz = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));
  memset(m_fz, 0, m_numNode * sizeof(Real_t));

  // Allocate tmp arrays
  m_fx_elem = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_fx_elem, 0, m_numElem*8 * sizeof(Real_t));
  m_fy_elem = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_fy_elem, 0, m_numElem*8 * sizeof(Real_t));
  m_fz_elem = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_fz_elem, 0, m_numElem*8 * sizeof(Real_t));
  m_dvdx = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_dvdx, 0, m_numElem*8 * sizeof(Real_t));
  m_dvdy = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_dvdy, 0, m_numElem*8 * sizeof(Real_t));
  m_dvdz = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_dvdz, 0, m_numElem*8 * sizeof(Real_t));
  m_x8n = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_x8n, 0, m_numElem*8 * sizeof(Real_t));
  m_y8n = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_y8n, 0, m_numElem*8 * sizeof(Real_t));
  m_z8n = (Real_t*) acc_create_unified(NULL, m_numElem*8 * sizeof(Real_t));
  memset(m_z8n, 0, m_numElem*8 * sizeof(Real_t));
  m_sigxx = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_sigxx, 0, m_numElem * sizeof(Real_t));
  m_sigyy = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_sigyy, 0, m_numElem * sizeof(Real_t));
  m_sigzz = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_sigzz, 0, m_numElem * sizeof(Real_t));
  m_determ = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_determ, 0, m_numElem * sizeof(Real_t));
  m_dxx = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_dxx, 0, m_numElem * sizeof(Real_t));
  m_dyy = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_dyy, 0, m_numElem * sizeof(Real_t));
  m_dzz = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_dzz, 0, m_numElem * sizeof(Real_t));
  m_vnew = (Real_t*) acc_create_unified(NULL, m_numElem * sizeof(Real_t));
  memset(m_vnew, 0, m_numElem * sizeof(Real_t));
#else
  m_regNumList = (Index_t*) calloc(m_numElem, sizeof(Index_t)) ;  // material indexset

  m_nodelist = (Index_t*) calloc(8*m_numElem, sizeof(Index_t));

  // elem connectivities through face 
  m_lxim = (Index_t*) calloc(m_numElem, sizeof(Index_t));
  m_lxip = (Index_t*) calloc(m_numElem, sizeof(Index_t));
  m_letam = (Index_t*) calloc(m_numElem, sizeof(Index_t));
  m_letap = (Index_t*) calloc(m_numElem, sizeof(Index_t));
  m_lzetam = (Index_t*) calloc(m_numElem, sizeof(Index_t));
  m_lzetap = (Index_t*) calloc(m_numElem, sizeof(Index_t));

  m_elemBC = (Int_t*) calloc(m_numElem, sizeof(Int_t));

  m_e = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_p = (Real_t*) calloc(m_numElem, sizeof(Real_t));

  m_q = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_ql = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_qq = (Real_t*) calloc(m_numElem, sizeof(Real_t));

  m_v = (Real_t*) calloc(m_numElem, sizeof(Real_t));

  m_volo = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_delv = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_vdov = (Real_t*) calloc(m_numElem, sizeof(Real_t));

  m_arealg = (Real_t*) calloc(m_numElem, sizeof(Real_t));

  m_ss = (Real_t*) calloc(m_numElem, sizeof(Real_t));

  m_elemMass = (Real_t*) calloc(m_numElem, sizeof(Real_t));

  // Node-centered 

  m_x = (Real_t*) calloc(m_numNode, sizeof(Real_t)); // coordinates 
  m_y = (Real_t*) calloc(m_numNode, sizeof(Real_t));
  m_z = (Real_t*) calloc(m_numNode, sizeof(Real_t));

  m_xd = (Real_t*) calloc(m_numNode, sizeof(Real_t)); // velocities 
  m_yd = (Real_t*) calloc(m_numNode, sizeof(Real_t));
  m_zd = (Real_t*) calloc(m_numNode, sizeof(Real_t));

  m_xdd = (Real_t*) calloc(m_numNode, sizeof(Real_t)); // accelerations 
  m_ydd = (Real_t*) calloc(m_numNode, sizeof(Real_t));
  m_zdd = (Real_t*) calloc(m_numNode, sizeof(Real_t));

  m_fx = (Real_t*) calloc(m_numNode, sizeof(Real_t));  // forces 
  m_fy = (Real_t*) calloc(m_numNode, sizeof(Real_t));
  m_fz = (Real_t*) calloc(m_numNode, sizeof(Real_t));

  // Allocate tmp arrays
  m_fx_elem = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_fy_elem = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_fz_elem = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_dvdx = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_dvdy = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_dvdz = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_x8n = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_y8n = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_z8n = (Real_t*) calloc(m_numElem*8, sizeof(Real_t));
  m_sigxx = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_sigyy = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_sigzz = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_determ = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_dxx = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_dyy = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_dzz = (Real_t*) calloc(m_numElem, sizeof(Real_t));
  m_vnew = (Real_t*) calloc(m_numElem, sizeof(Real_t));
#endif
  Index_t allElem = m_numElem +  /* local elem */
    2*m_sizeX*m_sizeY + /* plane ghosts */
    2*m_sizeX*m_sizeZ + /* row ghosts */
    2*m_sizeY*m_sizeZ ; /* col ghosts */
  AllocateGradients(allElem);

#ifdef USE_UNIFIEDMEM
  m_nodalMass = (Real_t*) acc_create_unified(NULL, m_numNode * sizeof(Real_t));  // mass 
  memset(m_nodalMass, 0, m_numNode * sizeof(Real_t));  // mass 
#else
  m_nodalMass = (Real_t*) calloc(m_numNode, sizeof(Real_t));  // mass 
#endif

  SetupCommBuffers(edgeNodes);

  Index_t i, lnode, j;
  // Note - v initializes to 1.0, not 0.0!
  for (i=0; i<m_numElem; ++i) {
    m_v[i] = (Real_t)(1.0);
  }

  // OpenACC - init device ptrs
#ifdef _OPENACC
  m_numDevs = acc_get_num_devices(acc_device_nvidia);

  printf("M_NUMDEVS [%d]\n", m_numDevs);
#if USE_MPI
  if(m_numDevs > 1) {
    //printf("%d Nvidia accelerators found.\n\n", m_numDevs);
    int numRanks;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

    /* Explicitly set device number if using MPI and >1 device */
    if(numRanks > 1) {
      //acc_set_device_num((myRank % m_numDevs) + 1, acc_device_nvidia);
      acc_set_device_num(0, acc_device_nvidia);
    }
  }
#endif
#endif

  BuildMesh(nx, edgeNodes, edgeElems);

#if 1
  SetupThreadSupportStructures();
#else
  // These arrays are not used if we're not threaded
  m_nodeElemStart = NULL;
  m_nodeElemCount = NULL;
  m_nodeElemCornerList = NULL;
#endif

  // Setup region index sets. For now, these are constant sized
  // throughout the run, but could be changed every cycle to 
  // simulate effects of ALE on the lagrange solver
  CreateRegionIndexSets(nr, balance);

  // Setup symmetry nodesets
  SetupSymmetryPlanes(edgeNodes);

  // Setup element connectivities
  SetupElementConnectivities(edgeElems);

  // Setup symmetry planes and free surface boundary arrays
  SetupBoundaryConditions(edgeElems);


  // Setup defaults

  // These can be changed (requires recompile) if you want to run
  // with a fixed timestep, or to a different end time, but it's
  // probably easier/better to just run a fixed number of timesteps
  // using the -i flag in 2.x

  m_dtfixed = (Real_t)(-1.0e-6) ; // Negative means use courant condition
  m_stoptime  = (Real_t)(1.0e-2); // *Real_t(edgeElems*tp/45.0) ;

  // Initial conditions
  m_deltatimemultlb = (Real_t)(1.1) ;
  m_deltatimemultub = (Real_t)(1.2) ;
  m_dtcourant = (Real_t)(1.0e+20) ;
  m_dthydro   = (Real_t)(1.0e+20) ;
  m_dtmax     = (Real_t)(1.0e-2) ;
  m_time    = (Real_t)(0.) ;
  m_cycle   = 0 ;

  // initialize field data 
  for (i=0; i<m_numElem; ++i) {
    Real_t x_local[8], y_local[8], z_local[8] ;
    Index_t *elemToNode = &m_nodelist[8*i] ;
    for(lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemToNode[lnode];
      x_local[lnode] = m_x[gnode];
      y_local[lnode] = m_y[gnode];
      z_local[lnode] = m_z[gnode];
    }

    // volume calculations
    Real_t volume = CalcElemVolume(x_local, y_local, z_local );
    m_volo[i] = volume ;
    m_elemMass[i] = volume ;
    for (j=0; j<8; ++j) {
      Index_t idx = elemToNode[j] ;
      m_nodalMass[idx] += volume / (Real_t)(8.0) ;
    }
  }

  // deposit initial energy
  // An energy of 3.948746e+7 is correct for a problem with
  // 45 zones along a side - we need to scale it
  const Real_t ebase = 3.948746e+7;
  Real_t scale = (nx*m_tp)/45.0;
  Real_t einit = ebase*scale*scale*scale;
  if (m_rowLoc + m_colLoc + m_planeLoc == 0) {
    // Dump into the first zone (which we know is in the corner)
    // of the domain that sits at the origin
    m_e[0] = einit;
  }

  // Initialize deltatime
  if(m_dtfixed > (Real_t)(0.)) {
    m_deltatime = m_dtfixed;
    printf("Using fixed timestep of %12.6e\n\n", m_deltatime);
  }
  else {
    //set initial deltatime base on analytic CFL calculation
    m_deltatime = (.5*cbrt(m_volo[0]))/sqrt(2*einit);
  }

} // End constructor


////////////////////////////////////////////////////////////////////////////////
void
BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)
{
  Index_t meshEdgeElems = m_tp*nx ;
  Index_t plane, row, col;

  // initialize nodal coordinates 
  Int_t nidx = 0 ;
  Real_t tz = (Real_t)(1.125)*(Real_t)(m_planeLoc*nx)/(Real_t)(meshEdgeElems) ;
  for (plane=0; plane<edgeNodes; ++plane) {
     Real_t ty = (Real_t)(1.125)*(Real_t)(m_rowLoc*nx)/(Real_t)(meshEdgeElems) ;
     for (row=0; row<edgeNodes; ++row) {
        Real_t tx = (Real_t)(1.125)*(Real_t)(m_colLoc*nx)/(Real_t)(meshEdgeElems) ;
        for (col=0; col<edgeNodes; ++col) {
           m_x[nidx] = tx ;
           m_y[nidx] = ty ;
           m_z[nidx] = tz ;
           ++nidx ;
           // tx += ds ; // may accumulate roundoff... 
           tx = (Real_t)(1.125)*(Real_t)(m_colLoc*nx+col+1)/(Real_t)(meshEdgeElems) ;
        }
        // ty += ds ;  // may accumulate roundoff... 
        ty = (Real_t)(1.125)*(Real_t)(m_rowLoc*nx+row+1)/(Real_t)(meshEdgeElems) ;
     }
     // tz += ds ;  // may accumulate roundoff... 
     tz = (Real_t)(1.125)*(Real_t)(m_planeLoc*nx+plane+1)/(Real_t)(meshEdgeElems) ;
  }


  // embed hexehedral elements in nodal point lattice 
  Int_t zidx = 0 ;
  nidx = 0 ;
  for (plane=0; plane<edgeElems; ++plane) {
    for (row=0; row<edgeElems; ++row) {
      for (col=0; col<edgeElems; ++col) {
	Index_t *localNode = &m_nodelist[8*zidx] ;
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
}


////////////////////////////////////////////////////////////////////////////////
void
SetupThreadSupportStructures()
{
#if _OPENMP
   Index_t numthreads = omp_get_max_threads();
#else
   Index_t numthreads = 1;
#endif

   Index_t i, j;

   // These structures are always needed if using OpenACC, so just always
   // allocate them
   if (1 /*numthreads > 1*/) {
     // set up node-centered indexing of elements 
#ifdef USE_UNIFIEDMEM
     m_nodeElemCount = (Index_t*) acc_create_unified(NULL, m_numNode * sizeof(Index_t)) ;
     memset(m_nodeElemCount, 0, m_numNode * sizeof(Index_t)) ;
#else
     m_nodeElemCount = (Index_t*) calloc(m_numNode, sizeof(Index_t)) ;
#endif

     for (i=0; i<m_numNode; ++i) {
       m_nodeElemCount[i] = 0 ;
     }

     for (i=0; i<m_numElem; ++i) {
       Index_t *nl = &m_nodelist[8*i] ;
       for (j=0; j < 8; ++j) {
         ++(m_nodeElemCount[nl[j]] );
       }
     }

#ifdef USE_UNIFIEDMEM
     m_nodeElemStart = (Index_t*) acc_create_unified(NULL, m_numNode * sizeof(Index_t)) ;
     memset(m_nodeElemStart, 0, m_numNode * sizeof(Index_t)) ;
#else
     m_nodeElemStart = (Index_t*) calloc(m_numNode, sizeof(Index_t)) ;
#endif

     m_nodeElemStart[0] = 0;

     for (i=1; i < m_numNode; ++i) {
       m_nodeElemStart[i] =
         m_nodeElemStart[i-1] + m_nodeElemCount[i-1] ;
     }


#ifdef USE_UNIFIEDMEM
     m_nodeElemCornerList =
       (Index_t*) acc_create_unified(NULL, m_nodeElemStart[m_numNode-1] + m_nodeElemCount[m_numNode-1] * sizeof(Index_t));
     memset(m_nodeElemCornerList, 0, m_nodeElemStart[m_numNode-1] + m_nodeElemCount[m_numNode-1] * sizeof(Index_t));
#else
     m_nodeElemCornerList =
       (Index_t*) calloc(m_nodeElemStart[m_numNode-1] + m_nodeElemCount[m_numNode-1], sizeof(Index_t));
#endif

     for (i=0; i < m_numNode; ++i) {
       m_nodeElemCount[i] = 0;
     }

     for (i=0; i < m_numElem; ++i) {
       Index_t *nl = &m_nodelist[i*8] ;
       for (j=0; j < 8; ++j) {
         Index_t m = nl[j];
         Index_t k = i*8 + j ;
         Index_t offset = m_nodeElemStart[m] +
           m_nodeElemCount[m] ;
         m_nodeElemCornerList[offset] = k;
         ++(m_nodeElemCount[m]) ;
       }
     }

     Index_t clSize = m_nodeElemStart[m_numNode-1] +
       m_nodeElemCount[m_numNode-1] ;
     for (i=0; i < clSize; ++i) {
       Index_t clv = m_nodeElemCornerList[i] ;
       if ((clv < 0) || (clv > m_numElem*8)) {
         fprintf(stderr,
             "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
#if USE_MPI
         MPI_Abort(MPI_COMM_WORLD, -1);
#else
         exit(-1);
#endif
       }
     }
   }
   else {
     // These arrays are not used if we're not threaded
     m_nodeElemStart = NULL;
     m_nodeElemCount = NULL;
     m_nodeElemCornerList = NULL;
   }
}


////////////////////////////////////////////////////////////////////////////////
void
SetupCommBuffers(Int_t edgeNodes)
{
  // allocate a buffer large enough for nodal ghost data 
  Index_t maxEdgeSize = MAX(m_sizeX, MAX(m_sizeY, m_sizeZ))+1 ;
  m_maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSize*maxEdgeSize) ;
  m_maxEdgeSize = CACHE_ALIGN_REAL(maxEdgeSize) ;

  // assume communication to 6 neighbors by default 
  m_rowMin = (m_rowLoc == 0)        ? 0 : 1;
  m_rowMax = (m_rowLoc == m_tp-1)     ? 0 : 1;
  m_colMin = (m_colLoc == 0)        ? 0 : 1;
  m_colMax = (m_colLoc == m_tp-1)     ? 0 : 1;
  m_planeMin = (m_planeLoc == 0)    ? 0 : 1;
  m_planeMax = (m_planeLoc == m_tp-1) ? 0 : 1;

#if USE_MPI   
  // account for face communication 
  Index_t comBufSize =
    (m_rowMin + m_rowMax + m_colMin + m_colMax + m_planeMin + m_planeMax) *
    m_maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for edge communication 
  comBufSize +=
    ((m_rowMin & m_colMin) + (m_rowMin & m_planeMin) + (m_colMin & m_planeMin) +
     (m_rowMax & m_colMax) + (m_rowMax & m_planeMax) + (m_colMax & m_planeMax) +
     (m_rowMax & m_colMin) + (m_rowMin & m_planeMax) + (m_colMin & m_planeMax) +
     (m_rowMin & m_colMax) + (m_rowMax & m_planeMin) + (m_colMax & m_planeMin)) *
    m_maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for corner communication 
  // factor of 16 is so each buffer has its own cache line 
  comBufSize += ((m_rowMin & m_colMin & m_planeMin) +
		 (m_rowMin & m_colMin & m_planeMax) +
		 (m_rowMin & m_colMax & m_planeMin) +
		 (m_rowMin & m_colMax & m_planeMax) +
		 (m_rowMax & m_colMin & m_planeMin) +
		 (m_rowMax & m_colMin & m_planeMax) +
		 (m_rowMax & m_colMax & m_planeMin) +
		 (m_rowMax & m_colMax & m_planeMax)) * CACHE_COHERENCE_PAD_REAL ;

  commDataSend = (Real_t*) calloc(comBufSize, sizeof(Real_t)) ;
  commDataRecv = (Real_t*) calloc(comBufSize, sizeof(Real_t)) ;
  // prevent floating point exceptions 
  memset(commDataSend, 0, comBufSize*sizeof(Real_t)) ;
  memset(commDataRecv, 0, comBufSize*sizeof(Real_t)) ;
#endif   

  // Boundary nodesets
#ifdef USE_UNIFIEDMEM
  m_symmX = (Index_t*) acc_create_unified(NULL, edgeNodes*edgeNodes * sizeof(Index_t));
  memset(m_symmX, 0, edgeNodes*edgeNodes * sizeof(Index_t));
  m_symmY = (Index_t*) acc_create_unified(NULL, edgeNodes*edgeNodes * sizeof(Index_t));
  memset(m_symmY, 0, edgeNodes*edgeNodes * sizeof(Index_t));
  m_symmZ = (Index_t*) acc_create_unified(NULL, edgeNodes*edgeNodes * sizeof(Index_t));
  memset(m_symmZ, 0, edgeNodes*edgeNodes * sizeof(Index_t));
#else
  m_symmX = (Index_t*) calloc(edgeNodes*edgeNodes, sizeof(Index_t));
  m_symmY = (Index_t*) calloc(edgeNodes*edgeNodes, sizeof(Index_t));
  m_symmZ = (Index_t*) calloc(edgeNodes*edgeNodes, sizeof(Index_t));
#endif

  if (m_colLoc == 0) {
    m_symmXempty = false;
  }
  if (m_rowLoc == 0) {
    m_symmYempty = false;
  }
  if (m_planeLoc == 0) {
    m_symmZempty = false;
  }
}


////////////////////////////////////////////////////////////////////////////////
void
CreateRegionIndexSets(Int_t nr, Int_t balance)
{
#if USE_MPI   
  Index_t myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
  srand(myRank);
#else
  srand(0);
  Index_t myRank = 0;
#endif
  m_numReg = nr;
  m_regElemSize = (Int_t*) calloc(m_numReg, sizeof(Int_t));
  m_regElemlist = (Int_t**) calloc(m_numReg, sizeof(Int_t*));
  Index_t nextIndex = 0;
  //if we only have one region just fill it
  // Fill out the regNumList with material numbers, which are always
  // the region index plus one 
  Index_t i;
  if(m_numReg == 1) {
    while (nextIndex < m_numElem) {
      m_regNumList[nextIndex] = 1;
      nextIndex++;
    }
    m_regElemSize[0] = 0;
  }
  //If we have more than one region distribute the elements.
  else {
    Int_t regionNum;
    Int_t regionVar;
    Int_t lastReg = -1;
    Int_t binSize;
    Int_t elements;
    Index_t runto = 0;
    Int_t costDenominator = 0;
    Int_t* regBinEnd = (Int_t*) calloc(m_numReg, sizeof(Int_t));
    //Determine the relative weights of all the regions.
    for (i=0 ; i<m_numReg ; ++i) {
      m_regElemSize[i] = 0;
      costDenominator += pow((i+1), balance);  //Total cost of all regions
      regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
    }
    //Until all elements are assigned
    while (nextIndex < m_numElem) {
      //pick the region
      regionVar = rand() % costDenominator;
      Index_t i = 0;
      while(regionVar >= regBinEnd[i])
        i++;
      //rotate the regions based on MPI rank.  Rotation is Rank % NumRegions
      regionNum = ((i + myRank) % m_numReg) + 1;
      // make sure we don't pick the same region twice in a row
      while(regionNum == lastReg) {
        regionVar = rand() % costDenominator;
        i = 0;
        while(regionVar >= regBinEnd[i])
          i++;
        regionNum = ((i + myRank) % m_numReg) + 1;
      }
      //Pick the bin size of the region and determine the number of elements.
      binSize = rand() % 1000;
      if(binSize < 773) {
        elements = rand() % 15 + 1;
      }
      else if(binSize < 937) {
        elements = rand() % 16 + 16;
      }
      else if(binSize < 970) {
        elements = rand() % 32 + 32;
      }
      else if(binSize < 974) {
        elements = rand() % 64 + 64;
      } 
      else if(binSize < 978) {
        elements = rand() % 128 + 128;
      }
      else if(binSize < 981) {
        elements = rand() % 256 + 256;
      }
      else
        elements = rand() % 1537 + 512;
      runto = elements + nextIndex;
      //Store the elements.  If we hit the end before we run out of elements then just stop.
      while (nextIndex < runto && nextIndex < m_numElem) {
        m_regNumList[nextIndex] = regionNum;
        nextIndex++;
      }
      lastReg = regionNum;
    } 
  }
  // Convert regNumList to region index sets
  // First, count size of each region 
  for (i=0 ; i<m_numElem ; ++i) {
    int r = m_regNumList[i]-1; // region index == regnum-1
    m_regElemSize[r]++;
  }
  // Second, allocate each region index set
  for (i=0 ; i<m_numReg ; ++i) {
#ifdef USE_UNIFIEDMEM
    m_regElemlist[i] = (Index_t*) acc_create_unified(NULL, m_regElemSize[i] * sizeof(Index_t));
    memset(m_regElemlist[i], 0, m_regElemSize[i] * sizeof(Index_t));
#else
    m_regElemlist[i] = (Index_t*) calloc(m_regElemSize[i], sizeof(Index_t));
#endif
    m_regElemSize[i] = 0;
  }
  // Third, fill index sets
  for (i=0 ; i<m_numElem ; ++i) {
    Index_t r = m_regNumList[i]-1;       // region index == regnum-1
    Index_t regndx = m_regElemSize[r]++; // Note increment
    m_regElemlist[r][regndx] = i;
  }

}

/////////////////////////////////////////////////////////////
void 
SetupSymmetryPlanes(Int_t edgeNodes)
{
  Int_t nidx = 0 ;
  Index_t i, j;
  for (i=0; i<edgeNodes; ++i) {
    Index_t planeInc = i*edgeNodes*edgeNodes ;
    Index_t rowInc   = i*edgeNodes ;
    for (j=0; j<edgeNodes; ++j) {
      if (m_planeLoc == 0) {
        m_symmZ[nidx] = rowInc   + j ;
      }
      if (m_rowLoc == 0) {
        m_symmY[nidx] = planeInc + j ;
      }
      if (m_colLoc == 0) {
        m_symmX[nidx] = planeInc + j*edgeNodes ;
      }
      ++nidx ;
    }
  }
}



/////////////////////////////////////////////////////////////
void
SetupElementConnectivities(Int_t edgeElems)
{
  Index_t i;
  m_lxim[0] = 0 ;
  for (i=1; i<m_numElem; ++i) {
    m_lxim[i]   = i-1 ;
    m_lxip[i-1] = i ;
  }
  m_lxip[m_numElem-1] = m_numElem-1 ;

  for (i=0; i<edgeElems; ++i) {
    m_letam[i] = i ; 
    m_letap[m_numElem-edgeElems+i] = m_numElem-edgeElems+i ;
  }
  for (i=edgeElems; i<m_numElem; ++i) {
    m_letam[i] = i-edgeElems ;
    m_letap[i-edgeElems] = i ;
  }

  for (i=0; i<edgeElems*edgeElems; ++i) {
    m_lzetam[i] = i ;
    m_lzetap[m_numElem-edgeElems*edgeElems+i] = m_numElem-edgeElems*edgeElems+i ;
  }
  for (i=edgeElems*edgeElems; i<m_numElem; ++i) {
    m_lzetam[i] = i - edgeElems*edgeElems ;
    m_lzetap[i-edgeElems*edgeElems] = i ;
  }
}

/////////////////////////////////////////////////////////////
void
SetupBoundaryConditions(Int_t edgeElems) 
{
  Index_t i, j;
  Index_t ghostIdx[6] ;  // offsets to ghost locations

  // set up boundary condition information
  memset(m_elemBC, 0, m_numElem*sizeof(Int_t));

  for (i=0; i<6; ++i) {
    ghostIdx[i] = INT_MIN ;
  }

  Int_t pidx = m_numElem ;
  if (m_planeMin != 0) {
    ghostIdx[0] = pidx ;
    pidx += m_sizeX*m_sizeY ;
  }

  if (m_planeMax != 0) {
    ghostIdx[1] = pidx ;
    pidx += m_sizeX*m_sizeY ;
  }

  if (m_rowMin != 0) {
    ghostIdx[2] = pidx ;
    pidx += m_sizeX*m_sizeZ ;
  }

  if (m_rowMax != 0) {
    ghostIdx[3] = pidx ;
    pidx += m_sizeX*m_sizeZ ;
  }

  if (m_colMin != 0) {
    ghostIdx[4] = pidx ;
    pidx += m_sizeY*m_sizeZ ;
  }

  if (m_colMax != 0) {
    ghostIdx[5] = pidx ;
  }

  // symmetry plane or free surface BCs 
  for (i=0; i<edgeElems; ++i) {
    Index_t planeInc = i*edgeElems*edgeElems ;
    Index_t rowInc   = i*edgeElems ;
    for (j=0; j<edgeElems; ++j) {
      if (m_planeLoc == 0) {
        m_elemBC[rowInc+j] |= ZETA_M_SYMM ;
      }
      else {
        m_elemBC[rowInc+j] |= ZETA_M_COMM ;
        m_lzetam[rowInc+j] = ghostIdx[0] + rowInc + j ;
      }

      if (m_planeLoc == m_tp-1) {
        m_elemBC[rowInc+j+m_numElem-edgeElems*edgeElems] |=
          ZETA_P_FREE;
      }
      else {
        m_elemBC[rowInc+j+m_numElem-edgeElems*edgeElems] |=
          ZETA_P_COMM ;
        m_lzetap[rowInc+j+m_numElem-edgeElems*edgeElems] =
          ghostIdx[1] + rowInc + j ;
      }

      if (m_rowLoc == 0) {
        m_elemBC[planeInc+j] |= ETA_M_SYMM ;
      }
      else {
        m_elemBC[planeInc+j] |= ETA_M_COMM ;
        m_letam[planeInc+j] = ghostIdx[2] + rowInc + j ;
      }

      if (m_rowLoc == m_tp-1) {
        m_elemBC[planeInc+j+edgeElems*edgeElems-edgeElems] |= 
          ETA_P_FREE ;
      }
      else {
        m_elemBC[planeInc+j+edgeElems*edgeElems-edgeElems] |= 
          ETA_P_COMM ;
        m_letap[planeInc+j+edgeElems*edgeElems-edgeElems] =
          ghostIdx[3] +  rowInc + j ;
      }

      if (m_colLoc == 0) {
        m_elemBC[planeInc+j*edgeElems] |= XI_M_SYMM ;
      }
      else {
        m_elemBC[planeInc+j*edgeElems] |= XI_M_COMM ;
        m_lxim[planeInc+j*edgeElems] = ghostIdx[4] + rowInc + j ;
      }

      if (m_colLoc == m_tp-1) {
        m_elemBC[planeInc+j*edgeElems+edgeElems-1] |= XI_P_FREE ;
      }
      else {
        m_elemBC[planeInc+j*edgeElems+edgeElems-1] |= XI_P_COMM ;
        m_lxip[planeInc+j*edgeElems+edgeElems-1] =
          ghostIdx[5] + rowInc + j ;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////
void
AllocateGradients(Int_t numElem)
{
#ifdef USE_UNIFIEDMEM
  // Velocity gradients
  m_delv_xi = (Real_t*) acc_create_unified(NULL, numElem * sizeof(Real_t));
  memset(m_delv_xi, 0, numElem * sizeof(Real_t));
  m_delv_eta = (Real_t*) acc_create_unified(NULL, numElem * sizeof(Real_t));
  memset(m_delv_eta, 0, numElem * sizeof(Real_t));
  m_delv_zeta= (Real_t*) acc_create_unified(NULL, numElem * sizeof(Real_t));
  memset(m_delv_zeta, 0, numElem * sizeof(Real_t));

  // Position gradients
  m_delx_xi = (Real_t*) acc_create_unified(NULL, numElem * sizeof(Real_t));
  memset(m_delx_xi, 0, numElem * sizeof(Real_t));
  m_delx_eta = (Real_t*) acc_create_unified(NULL, numElem * sizeof(Real_t));
  memset(m_delx_eta, 0, numElem * sizeof(Real_t));
  m_delx_zeta= (Real_t*) acc_create_unified(NULL, numElem * sizeof(Real_t));
  memset(m_delx_zeta, 0, numElem * sizeof(Real_t));
#else
  // Velocity gradients
  m_delv_xi = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_delv_eta = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_delv_zeta= (Real_t*) calloc(numElem, sizeof(Real_t));

  // Position gradients
  m_delx_xi = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_delx_eta = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_delx_zeta= (Real_t*) calloc(numElem, sizeof(Real_t));
#endif
}

///////////////////////////////////////////////////////////////////////////
void
AllocateRegionTmps(Int_t numElem)
{
#ifdef USE_UNIFIEDMEM
  m_e_old = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_e_old, 0, numElem *  sizeof(Real_t));
  m_delvc = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_delvc, 0, numElem *  sizeof(Real_t));
  m_p_old = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_p_old, 0, numElem *  sizeof(Real_t));
  m_q_old = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_q_old, 0, numElem *  sizeof(Real_t));
  m_compression = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_compression, 0, numElem *  sizeof(Real_t));
  m_compHalfStep = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_compHalfStep, 0, numElem *  sizeof(Real_t));
  m_qq_old = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_qq_old, 0, numElem *  sizeof(Real_t));
  m_ql_old = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_ql_old, 0, numElem *  sizeof(Real_t));
  m_work = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_work, 0, numElem *  sizeof(Real_t));
  m_p_new = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_p_new, 0, numElem *  sizeof(Real_t));
  m_e_new = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_e_new, 0, numElem *  sizeof(Real_t));
  m_q_new = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_q_new, 0, numElem *  sizeof(Real_t));
  m_bvc = (Real_t*) acc_create_unified(NULL, numElem *  sizeof(Real_t));
  memset(m_bvc, 0, numElem *  sizeof(Real_t));
  m_pbvc = (Real_t*) acc_create_unified(NULL, numElem * sizeof(Real_t));
  memset(m_pbvc, 0, numElem * sizeof(Real_t));
#else
  m_e_old = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_delvc = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_p_old = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_q_old = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_compression = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_compHalfStep = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_qq_old = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_ql_old = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_work = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_p_new = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_e_new = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_q_new = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_bvc = (Real_t*) calloc(numElem, sizeof(Real_t));
  m_pbvc = (Real_t*) calloc(numElem, sizeof(Real_t));
#endif
}

///////////////////////////////////////////////////////////////////////////
void
DeallocateGradients()
{
  /*
  m_delx_zeta.erase(m_delx_zeta.begin(), m_delx_zeta.end()) ;
  m_delx_eta.erase(m_delx_eta.begin(), m_delx_eta.end()) ;
  m_delx_xi.erase(m_delx_xi.begin(), m_delx_xi.end()) ;

  m_delv_zeta.erase(m_delv_zeta.begin(), m_delv_zeta.end()) ;
  m_delv_eta.erase(m_delv_eta.begin(), m_delv_eta.end()) ;
  m_delv_xi.erase(m_delv_xi.begin(), m_delv_xi.end()) ;
  */
}

///////////////////////////////////////////////////////////////////////////
void InitMeshDecomp(int numRanks, int myRank,
                    int* col, int* row, int* plane, int* side)
{
  int testProcs;
  int dx, dy, dz;
  int myDom;

  // Assume cube processor layout for now 
  testProcs = (int) (cbrt((Real_t)(numRanks))+0.5) ;
  if (testProcs*testProcs*testProcs != numRanks) {
    printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n") ;
#if USE_MPI      
    MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
    exit(-1);
#endif
  }
  if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
    printf("MPI operations only support float and double right now...\n");
#if USE_MPI      
    MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
    exit(-1);
#endif
  }
  if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
    printf("corner element comm buffers too small.  Fix code.\n") ;
#if USE_MPI      
    MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
    exit(-1);
#endif
  }

  dx = testProcs ;
  dy = testProcs ;
  dz = testProcs ;

  // temporary test
  if (dx*dy*dz != numRanks) {
    printf("error -- must have as many domains as procs\n") ;
#if USE_MPI      
    MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
    exit(-1);
#endif
  }
  int remainder = dx*dy*dz % numRanks ;
  if (myRank < remainder) {
    myDom = myRank*( 1+ (dx*dy*dz / numRanks)) ;
  }
  else {
    myDom = remainder*( 1+ (dx*dy*dz / numRanks)) +
      (myRank - remainder)*(dx*dy*dz/numRanks) ;
  }

  *col = myDom % dx ;
  *row = (myDom / dx) % dy ;
  *plane = myDom / (dx*dy) ;
  *side = testProcs;

  return;
}

void
ReleaseDeviceMem()
{
#ifdef _OPENACC
  acc_shutdown(acc_device_nvidia);
#endif
}

