#include "srad.h"

extern int grid_rows, grid_cols, size_I, size_R, niter;
extern int r1, r2, c1, c2;
extern float lambda;
extern int nthreads;

#if SWI_UPDATE 

// iN, iS, jE, jW unused in this version
void mainComp(float J[_SIZE_I_], int iN[_ROWS_], int iS[_ROWS_],
    int jE[_COLS_], int jW[_COLS_], float dN[_SIZE_I_], float dS[_SIZE_I_],
    float dW[_SIZE_I_], float dE[_SIZE_I_], float c[_SIZE_I_])
{

  float sum=0, sum2=0;     

  // CPU computation
  for (int i = r1; i <= r2; i++) {
    for (int j = c1; j <= c2; j++) {
      float tmp   = J[i * grid_cols + j];
      sum  += tmp ;
      sum2 += tmp*tmp;
    }
  }

  float meanROI = sum / size_R;
  float varROI  = (sum2 / size_R) - meanROI*meanROI;
  float q0sqr   = varROI / (meanROI*meanROI);

  #pragma acc parallel num_gangs(1) num_workers(1) \
      copyin(q0sqr)  present(J, dN, dS, dE, dW, c)
  {
    #pragma acc loop collapse(2)
    #pragma openarc transform window(J[0:_ROWS_*_COLS_], c[0:_ROWS_*_COLS_])
    for (int i = 0 ; i < grid_rows ; i++) {
      for (int j = 0; j < _COLS_; j++) { 

        int k = i * _COLS_ + j;
        float Jc = J[i*_COLS_ + j];

        // directional derivates
        if( i == 0 ) {
          dN[k] = J[i*_COLS_ + j] - Jc;
          dS[k] = J[i*_COLS_ + j + _COLS_] - Jc;
        } else if( i == (grid_rows-1) ) {
          dN[k] = J[(i-1) * _COLS_ + j] - Jc;
          dS[k] = J[i*_COLS_ + j] - Jc;
        } else {
          dN[k] = J[(i-1) * _COLS_ + j] - Jc;
          dS[k] = J[(i+1) * _COLS_ + j] - Jc;
        }
        if( j == 0 ) {    
          dW[k] = J[i * _COLS_ + j] - Jc;
          dE[k] = J[i * _COLS_ + j + 1] - Jc;
        } else if( j == (_COLS_-1) ) {
          dW[k] = J[i * _COLS_ + (j-1)] - Jc;
          dE[k] = J[i * _COLS_ + j] - Jc;
        } else {
          dW[k] = J[i * _COLS_ + (j-1)] - Jc;
          dE[k] = J[i * _COLS_ + (j+1)] - Jc;
        }

        float G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
            + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

        float L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

        float num  = (0.5F*G2) - ((1.0F/16.0F)*(L*L)) ;
        float den  = 1 + (.25F*L);
        float qsqr = num/(den*den);

        // diffusion coefficent (equ 33)
        den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
        c[k] = 1.0F / (1.0F+den) ;

        // saturate diffusion coefficent
        if (c[k] < 0) {c[k] = 0;}
        else if (c[k] > 1) {c[k] = 1;}

      }
    }
  } // acc

  #pragma acc parallel num_gangs(1) num_workers(1) \
      present(J, dN, dS, dE, dW, c) 
  {
    #pragma acc loop collapse(2)
    #pragma openarc transform window(c[0:_ROWS_*_COLS_], J[0:_ROWS_*_COLS_]) 
    for (int i = 0; i < grid_rows; i++) {
      for (int j = 0; j < _COLS_; j++) {        

        // current index
        int k = i * _COLS_ + j;

        float cN, cS, cW, cE;

        // diffusion coefficent
        cN = c[i*_COLS_ + j];
        if( i == (grid_rows-1) ) {
          cS = c[i*_COLS_ + j];
        } else {
          cS = c[(i+1) * _COLS_ + j];
        }
        cW = c[i*_COLS_ + j];
        if( j == (_COLS_-1) ) {
          cE = c[i*_COLS_ + j];
        } else {
          cE = c[i * _COLS_ + j +1];
        }


        // divergence (equ 58)
        float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

        // image update (equ 61)
        J[k] = J[k] + 0.25F*lambda*D;
      }
    }
  } // acc

}
#endif
