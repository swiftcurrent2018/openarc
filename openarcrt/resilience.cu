#ifndef __RESILIENCE_CUDA__

#define __RESILIENCE_CUDA__

#include "resilience.h"

////////////////////////////////////////
// Functions used for resilience test //
////////////////////////////////////////
static __device__ void dev__HI_ftinjection_int8b(type8b * target, int ftinject,  long int epos, type8b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
    }   
}

static __device__ void dev__HI_ftinjection_int16b(type16b * target, int ftinject,  long int epos, type16b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
    }   
}

static __device__ void dev__HI_ftinjection_int32b(type32b * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
    }   
}

static __device__ void dev__HI_ftinjection_int64b(type64b * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
    }   
}

static __device__ void dev__HI_ftinjection_float(float * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
		FloatBits val;
        val.f = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.f;
    }   
}

static __device__ void dev__HI_ftinjection_double(double * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
		DoubleBits val;
        val.d = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.d;
    }   
}

#endif
