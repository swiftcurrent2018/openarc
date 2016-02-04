#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "fft_config.h"

#ifndef DELAY_ELEMENTS_ON_LOCAL
#define DELAY_ELEMENTS_ON_LOCAL 1
#endif

// Function prototypes
static void test_fft(int iterations, int inverse);
static void fft1d(flt2 * src, flt2 * dest, int count, int inverse);
static int coord(int iteration, int i);
static void fourier_transform_gold(int inverse, int lognr_points, dbl2 * data);
static void fourier_stage(int lognr_points, dbl2 * data);
double my_timer();

// Host memory buffers
flt2 *h_inData, *h_outData;
dbl2 *h_verify;

int main( int argc, char** argv ) {
	int iterations = 100;
	if( argc > 1 ) {
		iterations = atoi(argv[1]);
	}

	// Allocate host memory
	posix_memalign((void **)(&h_inData), AOCL_ALIGNMENT, sizeof(flt2)*N*iterations);
	posix_memalign((void **)(&h_outData), AOCL_ALIGNMENT, sizeof(flt2)*N*iterations);
	posix_memalign((void **)(&h_verify), AOCL_ALIGNMENT, sizeof(dbl2)*N*iterations);
	if (!(h_inData && h_outData && h_verify)) {
		printf("ERROR: Couldn't create host buffers\n");
		return _false;
	}

	test_fft(iterations, _false);
	test_fft(iterations, _true);

	// Free the resources allocated
	free(h_inData);
	free(h_outData);
	free(h_verify);

	return 0;
}

void test_fft(int iterations, int inverse) {
	double time;
	double totaltime;
	int i, j;
	double gpoints_per_sec;
	double gflops;
	double fpga_snr = 200;
	double mag_sum = 0;
	double noise_sum = 0;
	double magnitude;
	double noise;
	double db;

	printf("Launching");
	if (inverse)
		printf(" inverse");
	printf(" FFT transform for %d iterations\n", iterations);

	// Initialize input and produce verification data
	for (i = 0; i < iterations; i++) {
		for (j = 0; j < N; j++) {
			h_verify[coord(i, j)].x = h_inData[coord(i, j)].x = (float)((double)rand() / (double)RAND_MAX);
			h_verify[coord(i, j)].y = h_inData[coord(i, j)].y = (float)((double)rand() / (double)RAND_MAX);
		}
	}

	totaltime = my_timer();	
	#pragma acc data copyin(h_inData[0:N*iterations]) copyout(h_outData[0:N*iterations])
	{
		printf(inverse ? "\tInverse FFT" : "\tFFT");

		// Get the iterationstamp to evaluate performance
		time = my_timer();	
		
		fft1d(h_inData, h_outData, iterations, inverse);

		// Record execution time
		time = my_timer() - time;
	}

	totaltime = my_timer() - totaltime;
	printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));
	printf("\tMain execution time = %.4fms\n", (float)(totaltime * 1E3));
	gpoints_per_sec = ((double) iterations * N / time) * 1E-9;
	gflops = 5 * N * (log((float)N)/log((float)2))/(time / iterations * 1E9);
	printf("\tThroughput = %.4f Gpoints / sec (%.4f Gflops)\n", gpoints_per_sec, gflops);

	// Pick randomly a few iterations and check SNR

	for (i = 0; i < iterations; i+= rand() % 20 + 1) {
		fourier_transform_gold(inverse, LOGN, h_verify + coord(i, 0));
		mag_sum = 0;
		noise_sum = 0;
		for (j = 0; j < N; j++) {
			magnitude = (double)h_verify[coord(i, j)].x * (double)h_verify[coord(i, j)].x + (double)h_verify[coord(i, j)].y * (double)h_verify[coord(i, j)].y;
			noise = (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) * (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) + (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y) * (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y);

			mag_sum += magnitude;
			noise_sum += noise;
		}
		db = 10 * log(mag_sum / noise_sum) / log(10.0);
		// find minimum SNR across all iterations
		if (db < fpga_snr) fpga_snr = db;
	}
	printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", fpga_snr, fpga_snr > 125 ? "PASSED" : "FAILED");
}

static void fft1d(flt2 * src, flt2 * dest, int count, int inverse) {
	/* The FFT engine requires a sliding window array for data reordering; data 
	* stored in this array is carried across loop iterations and shifted by one
	* element every iteration; all loop dependencies derived from the uses of 
	* this array are simple transfers between adjacent array elements
	*/
	int i;

#if DELAY_ELEMENTS_ON_LOCAL == 0
	flt2 fft_delay_elements[N + 8 * (LOGN - 2)];
#endif

	/* This is the main loop. It runs 'count' back-to-back FFT transforms
	* In addition to the 'count * (N / 8)' iterations, it runs 'N / 8 - 1'
	* additional iterations to drain the last outputs
	* (see comments attached to the FFT engine)
	*
	* The compiler leverages pipeline parallelism by overlapping the
	* iterations of this loop - launching one iteration every clock cycle
	*/

#if DELAY_ELEMENTS_ON_LOCAL == 1
	#pragma acc parallel num_gangs(1) num_workers(1) present(src[0:N*count], dest[0:N*count]) copyin(count, inverse) 
#else
	#pragma acc parallel num_gangs(1) num_workers(1) present(src[0:N*count], dest[0:N*count]) copyin(count, inverse) create(fft_delay_elements[0: N + 8 * (LOGN -2)])
#endif
	{
#if DELAY_ELEMENTS_ON_LOCAL == 1
	flt2 fft_delay_elements[N + 8 * (LOGN - 2)];
#endif
	#pragma acc loop seq 
	for (i = 0; i < count * (N / 8) + N / 8 - 1; i++) {

		/* As required by the FFT engine, gather input data from 8 distinct
		* segments of the input buffer; for simplicity, this implementation 
		* does not attempt to coalesce memory accesses and this leads to
		* higher resource utilization (see the fft2d example for advanced
		* memory access techniques)
		*/

		int base = (i / (N / 8)) * N;
		int offset = i % (N / 8);

		flt2x8 data;
		// Perform memory transfers only when reading data in range
		if (i < count * (N / 8)) {
			data.i0 = src[base + offset];
			data.i1 = src[base + 4 * N / 8 + offset];
			data.i2 = src[base + 2 * N / 8 + offset];
			data.i3 = src[base + 6 * N / 8 + offset];
			data.i4 = src[base + N / 8 + offset];
			data.i5 = src[base + 5 * N / 8 + offset];
			data.i6 = src[base + 3 * N / 8 + offset];
			data.i7 = src[base + 7 * N / 8 + offset];
		} else {
			data.i0.x = data.i1.x = data.i2.x = data.i3.x = data.i4.x = data.i5.x = data.i6.x = data.i7.x = 0.0F;
			data.i0.y = data.i1.y = data.i2.y = data.i3.y = data.i4.y = data.i5.y = data.i6.y = data.i7.y = 0.0F;
		}

		// Perform one step of the FFT engine
		data = fft_step(data, i % (N / 8), fft_delay_elements, inverse, LOGN); 
	
		/* Store data back to memory. FFT engine outputs are delayed by
		* N / 8 - 1 steps, hence gate writes accordingly
		*/

		if (i >= N / 8 - 1) {
			int base = 8 * (i - (N / 8 - 1));

			// These consecutive accesses will be coalesced by the compiler
			dest[base] = data.i0;
			dest[base + 1] = data.i1;
			dest[base + 2] = data.i2;
			dest[base + 3] = data.i3;
			dest[base + 4] = data.i4;
			dest[base + 5] = data.i5;
			dest[base + 6] = data.i6;
			dest[base + 7] = data.i7;
		}
	}
	}
}

/////// HELPER FUNCTIONS ///////

// provides a linear offset in the input array
int coord(int iteration, int i) {
  return iteration * N + i;
}


// Reference Fourier transform
void fourier_transform_gold(int inverse, const int lognr_points, dbl2 *data) {
	int i,j;
	double tmp;
	int fwd;
	int bit_rev;
	const int nr_points = 1 << lognr_points;

	// The inverse requires swapping the real and imaginary component

	if (inverse) {
		for (i = 0; i < nr_points; i++) {
			tmp = data[i].x;
			data[i].x = data[i].y;
			data[i].y = tmp;;
		}
	}
	// Do a FT recursively
	fourier_stage(lognr_points, data);

	// The inverse requires swapping the real and imaginary component
	if (inverse) {
		for (i = 0; i < nr_points; i++) {
			tmp = data[i].x;
			data[i].x = data[i].y;
			data[i].y = tmp;;
		}
	}

	// Do the bit reversal

	dbl2 *temp = (dbl2 *)alloca(sizeof(dbl2) * nr_points);
	for (i = 0; i < nr_points; i++) temp[i] = data[i];
	for (i = 0; i < nr_points; i++) {
		fwd = i;
		bit_rev = 0;
		for (j = 0; j < lognr_points; j++) {
			bit_rev <<= 1;
			bit_rev |= fwd & 1;
			fwd >>= 1;
		}
		data[i] = temp[bit_rev];
	}
}

void fourier_stage(int lognr_points, dbl2 *data) {
	int i;
	int nr_points = 1 << lognr_points;
	if (nr_points == 1) return;
	dbl2 *half1 = (dbl2 *)alloca(sizeof(dbl2) * nr_points / 2);
	dbl2 *half2 = (dbl2 *)alloca(sizeof(dbl2) * nr_points / 2);
	for (i = 0; i < nr_points / 2; i++) {
		half1[i] = data[2 * i];
		half2[i] = data[2 * i + 1];
	}
	fourier_stage(lognr_points - 1, half1);
	fourier_stage(lognr_points - 1, half2);
	for (i = 0; i < nr_points / 2; i++) {
		data[i].x = half1[i].x + cos (2 * M_PI * i / nr_points) * half2[i].x + sin (2 * M_PI * i / nr_points) * half2[i].y;
		data[i].y = half1[i].y - sin (2 * M_PI * i / nr_points) * half2[i].x + cos (2 * M_PI * i / nr_points) * half2[i].y;
		data[i + nr_points / 2].x = half1[i].x - cos (2 * M_PI * i / nr_points) * half2[i].x - sin (2 * M_PI * i / nr_points) * half2[i].y;
		data[i + nr_points / 2].y = half1[i].y + sin (2 * M_PI * i / nr_points) * half2[i].x - cos (2 * M_PI * i / nr_points) * half2[i].y;
	}
}

double my_timer ()
{
    struct timeval time;
    gettimeofday (&time, 0); 
    return time.tv_sec + time.tv_usec / 1000000.0;
}
