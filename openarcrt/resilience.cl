#ifndef __RESILIENCE_OPENCL__

#define __RESILIENCE_OPENCL__

#include "resilience.h"

////////////////////////////////////////
// Functions used for resilience test //
////////////////////////////////////////
static void dev__HI_ftinjection_int8b(type8b __global * target, int ftinject,  long int epos, type8b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
    }   
}

static void dev__HI_ftinjection_int16b(type16b __global * target, int ftinject,  long int epos, type16b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
    }   
}

static void dev__HI_ftinjection_int32b(type32b __global * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
    }   
}

static void dev__HI_ftinjection_int64b(type64b __global * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
    }   
}

static void dev__HI_ftinjection_float(float __global * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
		FloatBits val;
        val.f = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.f;
    }   
}

static void dev__HI_ftinjection_double(double __global * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
		DoubleBits val;
        val.d = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.d;
    }   
}

#endif
