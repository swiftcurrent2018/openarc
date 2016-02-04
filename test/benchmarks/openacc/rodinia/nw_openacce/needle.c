#define LIMIT -999
#define TRACE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define DEBUG

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

#ifndef HOST_MEM_ALIGNMENT
#define HOST_MEM_ALIGNMENT 1
#endif

#if HOST_MEM_ALIGNMENT == 1
#define AOCL_ALIGNMENT 64
#endif

#ifndef _MAX_ROWS_
#define _MAX_ROWS_	2049
#ifdef _OPENARC_
#pragma openarc #define _MAX_ROWS_ 2049
#endif
#endif

#ifndef _BSIZE_
#define _BSIZE_ 16
#ifdef _OPENARC_
#pragma openarc #define _BSIZE_ 16
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
int maximum( int a,
				int b,
				int c){

		int k;
		if( a <= b )
				k = b;
		else 
				k = a;

		if( k <=c )
				return(c);
		else
				return(k);
}


int blosum62[24][24] = {
		{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
		{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
		{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
		{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
		{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
		{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
		{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
		{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
		{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
		{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
		{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
		{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
		{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
		{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
		{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
		{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
		{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
		{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
		{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
		{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
		{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
		{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
		{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
		{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

int max_rows, max_cols, penalty;
int omp_num_threads;

double gettime() {
		struct timeval t;
		gettimeofday(&t,0);
		return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
		int
main( int argc, char** argv) 
{
		double start_time, end_time;
		start_time = gettime();
		runTest( argc, argv);

		end_time = gettime();
		printf("Total Execution Time %lf sec. \n", end_time - start_time);
		return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
		fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads>\n", argv[0]);
		fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
		fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
		fprintf(stderr, "\t<num_threads>    - no. of threads\n");
		exit(1);
}

void mainComp(int input_itemsets[_MAX_ROWS_*_MAX_ROWS_], int referrence[_MAX_ROWS_*_MAX_ROWS_]) 
{
		int i;
		int bx, tx;
		int block_width = (max_cols - 1)/_BSIZE_;
#pragma acc data \
		copy(input_itemsets[0:_MAX_ROWS_*_MAX_ROWS_]) \
		copyin(referrence[0:_MAX_ROWS_*_MAX_ROWS_])
		{
				for( i = 1 ; i <= block_width ; i++){
#pragma acc kernels loop gang(i) 
						for( bx = 0 ; bx < i ; bx++){
								int temp[_BSIZE_+1][_BSIZE_+1]; 
								int ref[_BSIZE_][_BSIZE_];
#pragma acc loop worker(_BSIZE_) 
								for( tx = 0 ; tx < _BSIZE_ ; tx++){
										int b_index_x = bx; 
										int b_index_y = i - 1 - bx; 

										int index   = max_cols * _BSIZE_ * b_index_y + _BSIZE_ * b_index_x + tx + ( max_cols + 1 );
										int index_n   = max_cols * _BSIZE_ * b_index_y + _BSIZE_ * b_index_x + tx + ( 1 );  int index_w   = max_cols * _BSIZE_ * b_index_y + _BSIZE_ * b_index_x + ( max_cols );
										int index_nw =  max_cols * _BSIZE_ * b_index_y + _BSIZE_ * b_index_x;


										if (tx == 0)
												temp[tx][0] = input_itemsets[index_nw];


										for ( int ty = 0 ; ty < _BSIZE_ ; ty++)
												ref[ty][tx] = referrence[index + max_cols * ty];

#pragma acc barrier

										temp[tx + 1][0] = input_itemsets[index_w + max_cols * tx];

#pragma acc barrier

										temp[0][tx + 1] = input_itemsets[index_n];

#pragma acc barrier


										for( int m = 0 ; m < _BSIZE_ ; m++){

												if ( tx <= m ){

														int t_index_x =  tx + 1;
														int t_index_y =  m - tx + 1;

														temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
																		temp[t_index_y][t_index_x-1]  - penalty, 
																		temp[t_index_y-1][t_index_x]  - penalty);



												}

#pragma acc barrier

										}

										for( int m = _BSIZE_ - 2 ; m >=0 ; m--){

												if ( tx <= m){

														int t_index_x =  tx + _BSIZE_ - m ;
														int t_index_y =  _BSIZE_ - tx;

														temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
																		temp[t_index_y][t_index_x-1]  - penalty,
																		temp[t_index_y-1][t_index_x]  - penalty);

												}

#pragma acc barrier
										}

										for ( int ty = 0 ; ty < _BSIZE_ ; ty++)
												input_itemsets[index + ty * max_cols] = temp[ty+1][tx+1];

								}
						}
				}
				printf("Processing bottom-right matrix\n");
				//Compute bottom-right matrix 
				for( i = block_width - 1 ; i >= 1 ; i--){
#pragma acc kernels loop gang(i) 
						for( bx = 0 ; bx < i ; bx++){
								int temp[_BSIZE_+1][_BSIZE_+1];
								int ref[_BSIZE_][_BSIZE_];
#pragma acc loop worker(_BSIZE_) 
								for( tx = 0 ; tx < _BSIZE_ ; tx++){
										int b_index_x = bx + block_width - i  ;
										int b_index_y = block_width - bx -1; 

										int index   = max_cols * _BSIZE_ * b_index_y + _BSIZE_ * b_index_x + tx + ( max_cols + 1 );
										int index_n   = max_cols * _BSIZE_ * b_index_y + _BSIZE_ * b_index_x + tx + ( 1 );  int index_w   = max_cols * _BSIZE_ * b_index_y + _BSIZE_ * b_index_x + ( max_cols );
										int index_nw =  max_cols * _BSIZE_ * b_index_y + _BSIZE_ * b_index_x;


										for ( int ty = 0 ; ty < _BSIZE_ ; ty++)
												ref[ty][tx] = referrence[index + max_cols * ty];

#pragma acc barrier

										if (tx == 0)
												temp[tx][0] = input_itemsets[index_nw];


										temp[tx + 1][0] = input_itemsets[index_w + max_cols * tx];

#pragma acc barrier

										temp[0][tx + 1] = input_itemsets[index_n];

#pragma acc barrier


										for( int m = 0 ; m < _BSIZE_ ; m++){

												if ( tx <= m ){

														int t_index_x =  tx + 1;
														int t_index_y =  m - tx + 1;

														temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
																		temp[t_index_y][t_index_x-1]  - penalty, 
																		temp[t_index_y-1][t_index_x]  - penalty);     

												}

#pragma acc barrier

										}


										for( int m = _BSIZE_ - 2 ; m >=0 ; m--){

												if ( tx <= m){

														int t_index_x =  tx + _BSIZE_ - m ;
														int t_index_y =  _BSIZE_ - tx;

														temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
																		temp[t_index_y][t_index_x-1]  - penalty,
																		temp[t_index_y-1][t_index_x]  - penalty);


												}

#pragma acc barrier
										}


										for ( int ty = 0 ; ty < _BSIZE_ ; ty++)
												input_itemsets[index + ty * max_cols] = temp[ty+1][tx+1];

								}
						}
				}
		}

}



void mainCompCPU(int input_itemsets[_MAX_ROWS_*_MAX_ROWS_], int referrence[_MAX_ROWS_*_MAX_ROWS_]) 
{
		int i, idx, index;
		/////////////////////////////////
		// Used for inlining maximum() //
		/////////////////////////////////
		int a, b, c, k;
		for( i = 0 ; i < max_cols-2 ; i++){
#ifdef _OPENMP
				//omp_set_num_threads(omp_num_threads);
#pragma omp parallel for shared(input_itemsets) firstprivate(i,max_cols,penalty) private(idx, index) 
#endif 
				for( idx = 0 ; idx <= i ; idx++){
						index = (idx + 1) * max_cols + (i + 1 - idx);
						//	input_itemsets[index]= maximum( input_itemsets[index-1-max_cols]+ referrence[index], 
						//					input_itemsets[index-1]         - penalty, 
						//					input_itemsets[index-max_cols]  - penalty);
						a = input_itemsets[index-1-max_cols]+ referrence[index]; 
						b = input_itemsets[index-1] - penalty; 
						c = input_itemsets[index-max_cols] - penalty;
						if( a <= b )
								k = b;
						else 
								k = a;

						if( k <=c )
								input_itemsets[index] = c; 
						else
								input_itemsets[index] = k; 

				}
		}
		//Compute bottom-right matrix 
		for( i = max_cols - 4 ; i >= 0 ; i--){
#ifdef _OPENMP	
				//omp_set_num_threads(omp_num_threads);
#pragma omp parallel for shared(input_itemsets) firstprivate(i,max_cols,penalty) private(idx, index) 
#endif 
				for( idx = 0 ; idx <= i ; idx++){
						index =  ( max_cols - idx - 2 ) * max_cols + idx + max_cols - i - 2 ;
						//input_itemsets[index]= maximum( input_itemsets[index-1-max_cols]+ referrence[index], 
						//				input_itemsets[index-1]         - penalty, 
						//				input_itemsets[index-max_cols]  - penalty);
						a = input_itemsets[index-1-max_cols]+ referrence[index]; 
						b = input_itemsets[index-1] - penalty; 
						c = input_itemsets[index-max_cols] - penalty;
						if( a <= b )
								k = b;
						else 
								k = a;

						if( k <=c )
								input_itemsets[index] = c; 
						else
								input_itemsets[index] = k; 
				}

		}

}
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
		void
runTest( int argc, char** argv) 
{
		int *input_itemsets, *output_itemsets, *referrence;
		int i,j;
#ifdef DEBUG
		double start_time, end_time;
#endif
#ifdef TRACE
		FILE *fp;
#endif

#if HOST_MEM_ALIGNMENT == 1
        void *p;
#endif

		// the lengths of the two sequences should be able to divided by 16.
		// And at current stage  max_rows needs to equal max_cols
		if (argc == 4)
		{
				max_rows = atoi(argv[1]);
				max_cols = atoi(argv[1]);
				penalty = atoi(argv[2]);
				omp_num_threads = atoi(argv[3]);
				if( max_rows != (_MAX_ROWS_-1) ) {
						printf("Wrong value (%d) for macro, _MAX_ROWS_!\n", _MAX_ROWS_);
						return;
				}
		}
		else{
				usage(argc, argv);
		}

		max_rows = max_rows + 1;
		max_cols = max_cols + 1;
#if HOST_MEM_ALIGNMENT == 1
        posix_memalign(&p, AOCL_ALIGNMENT, max_rows*max_cols*sizeof(int));
        referrence = (int *)p;
        posix_memalign(&p, AOCL_ALIGNMENT, max_rows*max_cols*sizeof(int));
        input_itemsets = (int *)p;
        posix_memalign(&p, AOCL_ALIGNMENT, max_rows*max_cols*sizeof(int));
        output_itemsets = (int *)p;
#else
		referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
		input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
		output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
#endif


		if (!input_itemsets)
				fprintf(stderr, "error: can not allocate memory");

		srand ( 7 );

		for (i = 0 ; i < max_cols; i++){
				for (j = 0 ; j < max_rows; j++){
						input_itemsets[i*max_cols+j] = 0;
				}
		}

		printf("Start Needleman-Wunsch\n");

		for( i=1; i< max_rows ; i++){    //please define your own sequence. 
				input_itemsets[i*max_cols] = rand() % 10 + 1;
		}
		for( j=1; j< max_cols ; j++){    //please define your own sequence.
				input_itemsets[j] = rand() % 10 + 1;
		}


		for (i = 1 ; i < max_cols; i++){
				for (j = 1 ; j < max_rows; j++){
						referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
				}
		}

		for( i = 1; i< max_rows ; i++)
				input_itemsets[i*max_cols] = -i * penalty;
		for( j = 1; j< max_cols ; j++)
				input_itemsets[j] = -j * penalty;



		//Compute top-left matrix 
		printf("Num of threads: %d\n", omp_num_threads);
		printf("Processing top-left matrix\n");

#ifdef DEBUG
		start_time = gettime();
#endif

		mainComp(input_itemsets, referrence); 

#ifdef DEBUG
		end_time = gettime();
		printf("Accelerator Elapsed Time = %lf sec. \n", end_time - start_time);
#endif

		if(VERIFICATION) {
				int *input_itemsets_CPU;
				double deltaL2Norm = 0;
				double nonAccL2Norm = 0;
				double L2Norm;

				input_itemsets_CPU = (int *)malloc( max_rows * max_cols * sizeof(int) );

				srand ( 7 );

				for (i = 0 ; i < max_cols; i++){
						for (j = 0 ; j < max_rows; j++){
								input_itemsets_CPU[i*max_cols+j] = 0;
						}
				}


				for( i=1; i< max_rows ; i++){    //please define your own sequence. 
						input_itemsets_CPU[i*max_cols] = rand() % 10 + 1;
				}
				for( j=1; j< max_cols ; j++){    //please define your own sequence.
						input_itemsets_CPU[j] = rand() % 10 + 1;
				}



				for( i = 1; i< max_rows ; i++)
						input_itemsets_CPU[i*max_cols] = -i * penalty;
				for( j = 1; j< max_cols ; j++)
						input_itemsets_CPU[j] = -j * penalty;

#ifdef DEBUG
				start_time = gettime();
#endif

				mainCompCPU(input_itemsets_CPU, referrence); 

#ifdef DEBUG
				end_time = gettime();
				printf("Main Comp. Time CPU = %lf sec. \n", end_time - start_time);
#endif

				for (i = 0; i < max_rows * max_cols; ++i) {
						double d = input_itemsets_CPU[i] - input_itemsets[i];
						deltaL2Norm += d * d;
						nonAccL2Norm += input_itemsets_CPU[i] * input_itemsets_CPU[i];
				}

				L2Norm = sqrt(deltaL2Norm / nonAccL2Norm);

				if (L2Norm < 1e-6) {
						printf("Verification: Successful\n");
				} else {
						printf("Verification: Failed\n");		
				}
				printf("L2Norm = %lf\n", L2Norm);


				free(input_itemsets_CPU);

		}

#ifdef TRACE

		printf("print traceback value CPU:\n");
		if( (fp = fopen("nwTrace.txt", "w")) == 0 ) {
				printf("Can not open %s\n", "nwTrace.txt");
				return;
		}

		//int i, j;
		for (i = j = max_rows - 2; i>=0, j>=0;){

				int nw, n, w, traceback;

				if ( i == max_rows - 2 && j == max_rows - 2 )
						fprintf(fp, "%d ", input_itemsets[ i * max_cols + j]); //print the first element


				if ( i == 0 && j == 0 )
						break;


				if ( i > 0 && j > 0 ){
						nw = input_itemsets[(i - 1) * max_cols + j - 1];
						w  = input_itemsets[ i * max_cols + j - 1 ];
						n  = input_itemsets[(i - 1) * max_cols + j];
				}
				else if ( i == 0 ){
						nw = n = LIMIT;
						w  = input_itemsets[ i * max_cols + j - 1 ];
				}
				else if ( j == 0 ){
						nw = w = LIMIT;
						n  = input_itemsets[(i - 1) * max_cols + j];
				}
				else{
				}

				traceback = maximum(nw, w, n);

				fprintf(fp, "%d ", traceback);

				if(traceback == nw )
				{i--; j--; continue;}

				else if(traceback == w )
				{j--; continue;}

				else if(traceback == n )
				{i--; continue;}

				else
						;
		}

		fprintf(fp, "\n");
		fclose(fp);

#endif


}



