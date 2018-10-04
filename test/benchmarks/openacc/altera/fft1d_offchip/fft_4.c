// Copyright (C) 2013-2015 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

// 1D complex floating-point feed-forward radix-4 FFT / iFFT
// Processes 4 points in parallel.
// See bigger comments in fft_8.cl for more information.
// the input is ordered, the output is bit reversed

//#include "twid_radix_2_2.cl"

typedef struct {
  float x;
  float y;
} flt2;

// convenience struct representing the 4 elements processed in parallel
typedef union {
  struct {
    flt2 i0;
    flt2 i1;
    flt2 i2;
    flt2 i3;
  };
  flt2 d[4];
} flt2x4;

typedef struct {
    flt2 i0;
    flt2 i1;
    flt2 i2;
} flt2x3;

typedef struct {
    flt2 i0;
    flt2 i1;
} flt2x2;

typedef struct {
  flt2 i0;
  flt2 i1;
  flt2 i2;
} flt2x3;

flt2x4 butterfly(flt2x4 data) {
   flt2x4 res;
   res.i0 = data.i0 + data.i1;
   res.i1 = data.i0 - data.i1;
   res.i2 = data.i2 + data.i3;
   res.i3 = data.i2 - data.i3;
   return res;
}

flt2x4 swap_complex(flt2x4 data) {
   flt2x4 res;
   res.i0.x = data.i0.y;
   res.i0.y = data.i0.x;
   res.i1.x = data.i1.y;
   res.i1.y = data.i1.x;
   res.i2.x = data.i2.y;
   res.i2.y = data.i2.x;
   res.i3.x = data.i3.y;
   res.i3.y = data.i3.x;
   return res;
}

flt2x4 trivial_rotate(flt2x4 data) {
   flt2 tmp = data.i3;
   data.i3.x = tmp.y;
   data.i3.y = -tmp.x;
   return data;
}

flt2x4 swap(flt2x4 data) {
   flt2 tmp = data.i1;
   data.i1 = data.i2;
   data.i2 = tmp;
   return data;
}

flt2x2 delay_data(flt2x2 data, const int depth, const int depth_mod_mask, 
                 flt2x2 *shift_reg, int inv_count) {
   int read_addr  = (0 + inv_count)     & depth_mod_mask; 
   int write_addr = (depth + inv_count) & depth_mod_mask; 
   shift_reg[write_addr] = data;
   return shift_reg[read_addr];
}

flt2x4 reorder_data(flt2x4 data, const int depth, const int depth_mod_mask,
       flt2 *delay1, 
       flt2 *delay2, 
      int inv_count, const int stage, int toggle) {
   // Use disconnected segments of length 'depth + 1' elements starting at 
   // 'shift_reg' to implement the delay elements. At the end of each FFT step, 
   // the contents of the entire buffer is shifted by 1 element
   flt2x2 t;
   t.i0 = data.i1;
   t.i1 = data.i3;
   t = delay_data(t, depth, depth_mod_mask, ( flt2x2*)delay1, inv_count);
   data.i1 = t.i0;
   data.i3 = t.i1;
   
   if (toggle) {
      flt2 tmp = data.i0;
      data.i0 = data.i1;
      data.i1 = tmp;
      tmp = data.i2;
      data.i2 = data.i3;
      data.i3 = tmp;
   }

   t.i0 = data.i0;
   t.i1 = data.i2;
   t = delay_data(t, depth, depth_mod_mask, ( flt2x2*)delay2, inv_count);
   data.i0 = t.i0;
   data.i2 = t.i1;
   
   return data;
}

flt2 comp_mult(flt2 a, flt2 b) {
   flt2 res;
   res.x = a.x * b.x - a.y * b.y;
   res.y = a.x * b.y + a.y * b.x;
   return res;
}

flt2 twiddle(int index, int stage, int log_size, int stream) {
   flt2 twid;
   
   const float * twiddles_cos[TWID_STAGES][3] = {{tc00, tc01, tc02}, 
                                                    {tc10, tc11, tc12}, 
                                                    {tc20, tc21, tc22}, 
                                                    {tc30, tc31, tc32}, 
                                                    {tc40, tc41, tc42}};
                                                    
   const float * twiddles_sin[TWID_STAGES][3] = {{ts00, ts01, ts02}, 
                                                    {ts10, ts11, ts12}, 
                                                    {ts20, ts21, ts22}, 
                                                    {ts30, ts31, ts32}, 
                                                    {ts40, ts41, ts42}};

   // use the hardcoded twiddle fators, if available - otherwise, compute them
   int twid_stage = stage >> 1;
   if (log_size <= (TWID_STAGES * 2 + 2)) {
      int index_mult = 1 << (TWID_STAGES * 2 + 2 - log_size);
      twid.x = twiddles_cos[twid_stage][stream][index * index_mult];
      twid.y = twiddles_sin[twid_stage][stream][index * index_mult];

   }
   return twid;
}

flt2x3 get_twid(int index, int stage, int log_size) {
  flt2x3 result;
  result.i0 = twiddle(index, stage, log_size, 0);
  result.i1 = twiddle(index, stage, log_size, 1);
  result.i2 = twiddle(index, stage, log_size, 2);
  return result;
}

flt2x4 complex_rotate_given_twid(flt2x4 data, flt2x3 twid) {
   data.i1 = comp_mult(data.i1, twid.i0);
   data.i2 = comp_mult(data.i2, twid.i1);
   data.i3 = comp_mult(data.i3, twid.i2);
   return data;
}

// FFT complex rotation building block
flt2x4 complex_rotate(flt2x4 data, int index, int stage, int log_size, int bypass) {
  flt2x3 twid = get_twid(index, stage, log_size);
  if (bypass) {
    twid.i0.x = 1.0f; twid.i0.y = 0.0f;
    twid.i1.x = 1.0f; twid.i1.y = 0.0f;
    twid.i2.x = 1.0f; twid.i2.y = 0.0f;
  }
  return complex_rotate_given_twid(data, twid);
}

typedef struct {
  flt2x4 data;
  int size;
  int logM;
  int step;
  int inv_count;
  int two_to_logM_m_stage;
  int delay;
  int delay_mod_mask;
} interstage_data;

interstage_data do_single_stage (const int stage, interstage_data d, 
                                  flt2 *delay1,
                                  flt2 *delay2)
{
  int complex_stage = stage & 1;
  int process_stage = (stage < (d.logM - 1));

  int data_index = d.step; 

  if (process_stage) d.data = butterfly(d.data);

  if (complex_stage) {
    d.data = complex_rotate(d.data, data_index, stage, d.logM, !process_stage);
  }

  if (process_stage) d.data = swap(d.data);

  // Reordering multiplexers must toggle every 'delay' steps
  int toggle = data_index & d.delay;
  

  // Assign unique sections of the buffer for the set of delay elements at
  // each stage
  int delay_arg = process_stage ? d.delay : 0;
  int toggle_arg = process_stage ? toggle : 0;

  d.data = reorder_data(d.data, delay_arg, d.delay_mod_mask, delay1, delay2, d.inv_count, stage, toggle_arg);

  if (!complex_stage && process_stage) {
    d.data = trivial_rotate(d.data);
  }

  // update for next call
  d.two_to_logM_m_stage >>= 1;
  d.delay >>= 1;
  d.delay_mod_mask >>= 1;
  
  return d;
}


// process 4 input points towards and a FFT/iFFT of size N, N >= 4
flt2x4 fft_step(flt2x4 data, int inv_count, int step, 
                   flt2 *delay1,  flt2 *delay11,
                   flt2 *delay2,  flt2 *delay21,
                   flt2 *delay3,  flt2 *delay31,
                   flt2 *delay4,  flt2 *delay41,
                   flt2 *delay5,  flt2 *delay51,
                   flt2 *delay6,  flt2 *delay61,
                   flt2 *delay7,  flt2 *delay71,
                   flt2 *delay8,  flt2 *delay81,
                   flt2 *delay9,  flt2 *delay91,
                   flt2 *delayA,  flt2 *delayA1,
                  int inverse, const int logN, const int logM, const int size) {


    const int sizeN = 1 << logN;
    const int logD = (logN - logM);

    // Swap real and imaginary components if doing an inverse transform
    if (inverse) {
       data = swap_complex(data);
    }

    // Stage 0 of feed-forward FFT
    data = butterfly(data);
    data = trivial_rotate(data);
    data = swap(data);


    // next logN - 1 stages alternate two computation patterns
    interstage_data d;
    d.data = data;
    d.size = size;
    d.logM = logM;
    d.step = step;
    d.inv_count = inv_count;
    d.two_to_logM_m_stage = size >> 3;
    d.delay = size >> 3;
    d.delay_mod_mask = 2 * d.delay - 1;
    
    d = do_single_stage (1, d, delay1, delay11);
    
#if LOGN>3
    d = do_single_stage (2, d, delay2, delay21);
#if LOGN>4
    d = do_single_stage (3, d, delay3, delay31);
#if LOGN>5
    d = do_single_stage (4, d, delay4, delay41);
#if LOGN>6
    d = do_single_stage (5, d, delay5, delay51);
#if LOGN>7
    d = do_single_stage (6, d, delay6, delay61);
#if LOGN>8
    d = do_single_stage (7, d, delay7, delay71);
#if LOGN>9
    d = do_single_stage (8, d, delay8, delay81);
#if LOGN>10
    d = do_single_stage (9, d, delay9, delay91);
#if LOGN>11
    d = do_single_stage (10, d, delayA, delayA1);
#if LOGN>12
  #error "Maximum supported value of LOGN is 12.\nYou need to regenerate constant twiddle factores to support larger sizes."
#endif
#endif 
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif

    data = d.data;

    // stage logN - 1
    data = butterfly(data);

    if (inverse) {
       data = swap_complex(data);
    }

    return data;
}

