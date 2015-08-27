#include <stdio.h>

#define _BSIZE_ 16
#ifdef _OPENARC_
#pragma openarc #define _BSIZE_ 16
#endif

#ifndef _M_SIZE
#define _M_SIZE 4096
#ifdef _OPENARC_
#pragma openarc #define _M_SIZE 4096
#endif
#endif

extern int omp_num_threads;

void lud_omp(float * m, int matrix_dim)
{
     int i,j,k;
	 int offset;
	 int gx, gy, wx, wy;
     float sum;
	 //printf("num of threads = %d\n", omp_num_threads);
#pragma acc data copy(m[0:_M_SIZE])
	 {
     for (offset=0; offset <matrix_dim-_BSIZE_; offset+= _BSIZE_){
//lud_diagonal(m, matrix_dim, offset);
#pragma acc kernels loop gang(1) 
		for(gx=0; gx<1; gx++) {
			float shadow[_BSIZE_][_BSIZE_];
#pragma acc loop worker(_BSIZE_) private(wx, i, j)
			for(wx=0; wx<_BSIZE_; wx++) {
				int array_offset = offset*matrix_dim+offset;
				for(i=0; i<_BSIZE_; i++) {
					shadow[i][wx] = m[array_offset+wx];
					array_offset += matrix_dim;
				}
#pragma acc barrier
				for(i=0; i<_BSIZE_-1; i++) {
					if( wx>i ) {
						for(j=0; j<i; j++) {
							shadow[wx][i] -= shadow[wx][j]*shadow[j][i];
						}
						shadow[wx][i] /= shadow[i][i];
					}	
#pragma acc barrier
					if( wx>i ) {
						for(j=0; j<i+1; j++) {
							shadow[i+1][wx] -= shadow[i+1][j]*shadow[j][wx];
						}
					}
#pragma acc barrier
				}
				array_offset = (offset+1)*matrix_dim+offset;
				for(i=1; i<_BSIZE_; i++) {
					m[array_offset+wx]=shadow[i][wx];
					array_offset += matrix_dim;
				}
			} 	
		}

//lud_perimeter(m, matrix, offset);
#pragma acc kernels loop gang((matrix_dim-offset)/_BSIZE_-1)
		for(gx=0; gx<((matrix_dim-offset)/_BSIZE_-1); gx++) {
			float dia[_BSIZE_][_BSIZE_];
			float peri_row[_BSIZE_][_BSIZE_];
			float peri_col[_BSIZE_][_BSIZE_];
#pragma acc loop worker(_BSIZE_*2) private(i,j)
			for(wx=0; wx<_BSIZE_*2; wx++) {
				int array_offset, idx;
				if( wx<_BSIZE_ ) {
					idx = wx;
					array_offset = offset*matrix_dim+offset;
					for (i=0; i < _BSIZE_/2; i++){
      					dia[i][idx]=m[array_offset+idx];
      					array_offset += matrix_dim;
    				}   
    
    				array_offset = offset*matrix_dim+offset;
    				for (i=0; i < _BSIZE_; i++) {
      					peri_row[i][idx]=m[array_offset+(gx+1)*_BSIZE_+idx];
      					array_offset += matrix_dim;
    				}   

  				} else {
    				idx = wx-_BSIZE_;
    
    				array_offset = (offset+_BSIZE_/2)*matrix_dim+offset;
    				for (i=_BSIZE_/2; i < _BSIZE_; i++){
      					dia[i][idx]=m[array_offset+idx];
      					array_offset += matrix_dim;
    				}   
    
    				array_offset = (offset+(gx+1)*_BSIZE_)*matrix_dim+offset;
    				for (i=0; i < _BSIZE_; i++) {
      					peri_col[i][idx] = m[array_offset+idx];
      					array_offset += matrix_dim;
    				}   
				}
#pragma acc barrier

  				if (wx < _BSIZE_) { //peri-row
    				idx=wx;
    				for(i=1; i < _BSIZE_; i++){
      					for (j=0; j < i; j++)
        					peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    				}   
  				} else { //peri-col
    				idx=wx - _BSIZE_;
    				for(i=0; i < _BSIZE_; i++){
      					for(j=0; j < i; j++)
        					peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      					peri_col[idx][i] /= dia[i][i];
    				}   
    			}   
#pragma acc barrier
    
  				if (wx < _BSIZE_) { //peri-row
    				idx=wx;
    				array_offset = (offset+1)*matrix_dim+offset;
    				for(i=1; i < _BSIZE_; i++){
      					m[array_offset+(gx+1)*_BSIZE_+idx] = peri_row[i][idx];
      					array_offset += matrix_dim;
    				}   
  				} else { //peri-col
    				idx=wx - _BSIZE_;
    				array_offset = (offset+(gx+1)*_BSIZE_)*matrix_dim+offset;
    				for(i=0; i < _BSIZE_; i++){
      					m[array_offset+idx] =  peri_col[i][idx];
      					array_offset += matrix_dim;
    				}
  				}
			}
		}

//lud_internal(m, matrix_dim, offset);
#pragma acc kernels loop gang((matrix_dim-offset)/_BSIZE_-1)
		for(gy=0; gy<((matrix_dim-offset)/_BSIZE_-1); gy++) {
#pragma acc loop gang((matrix_dim-offset)/_BSIZE_-1)
			for(gx=0; gx<((matrix_dim-offset)/_BSIZE_-1); gx++) {
  				float peri_row[_BSIZE_][_BSIZE_];
  				float peri_col[_BSIZE_][_BSIZE_];
#pragma acc loop worker(_BSIZE_) 
				for(wy=0; wy<_BSIZE_; wy++) {
#pragma acc loop worker(_BSIZE_) 
					for(wx=0; wx<_BSIZE_; wx++) {
  						int i;
  						float sum;

  						int global_row_id = offset + (gy+1)*_BSIZE_;
  						int global_col_id = offset + (gx+1)*_BSIZE_;

  						peri_row[wy][wx] = m[(offset+wy)*matrix_dim+global_col_id+wx];
  						peri_col[wy][wx] = m[(global_row_id+wy)*matrix_dim+offset+wx];
#pragma acc barrier

  						sum = 0;
  						for (i=0; i < _BSIZE_; i++)
    						sum += peri_col[wy][i] * peri_row[i][wx];
  						m[(global_row_id+wy)*matrix_dim+global_col_id+wx] -= sum;
					}
				}
			}
		}
	} //end of offset loop

//lud_diagonal(m, matrix_dim, offset);
#pragma acc kernels loop gang(1) 
		for(gx=0; gx<1; gx++) {
			float shadow[_BSIZE_][_BSIZE_];
#pragma acc loop worker(_BSIZE_) private(wx, i, j)
			for(wx=0; wx<_BSIZE_; wx++) {
				int array_offset = offset*matrix_dim+offset;
				for(i=0; i<_BSIZE_; i++) {
					shadow[i][wx] = m[array_offset+wx];
					array_offset += matrix_dim;
				}
#pragma acc barrier
				for(i=0; i<_BSIZE_-1; i++) {
					if( wx>i ) {
						for(j=0; j<i; j++) {
							shadow[wx][i] -= shadow[wx][j]*shadow[j][i];
						}
						shadow[wx][i] /= shadow[i][i];
					}	
#pragma acc barrier
					if( wx>i ) {
						for(j=0; j<i+1; j++) {
							shadow[i+1][wx] -= shadow[i+1][j]*shadow[j][wx];
						}
					}
#pragma acc barrier
				}
				array_offset = (offset+1)*matrix_dim+offset;
				for(i=1; i<_BSIZE_; i++) {
					m[array_offset+wx]=shadow[i][wx];
					array_offset += matrix_dim;
				}
			} 	
		}
	}

}
