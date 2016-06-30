//////////////////////////////////////////////////////////////////////
// This header file contains macros used for fault-injection tests. //
// These macros may need to be customized for each benchmark.       //
// (Any undefined macro will be evaluated to 0.)                    //
//////////////////////////////////////////////////////////////////////

//Decide target GPU thread to inject faults.
#if _FTTHREAD == 1
#pragma openarc #define FTTHREAD ftthread(0)
#else
#pragma openarc #define FTTHREAD 
#endif

//Decide at which resilience region fault will be injected.
//If multiple resilience regions exist, users may have to create
//multiple macros to control each region separately.
#if RES_REGION0 ==   0
#pragma openarc #define RES_REGION0 0
#else
#pragma openarc #define RES_REGION0 1
#endif
#if RES_REGION1 ==   0
#pragma openarc #define RES_REGION1 0
#else
#pragma openarc #define RES_REGION1 1
#endif
#if RES_REGION2 ==   0
#pragma openarc #define RES_REGION2 0
#else
#pragma openarc #define RES_REGION2 1
#endif
#if RES_REGION3 ==   0
#pragma openarc #define RES_REGION3 0
#else
#pragma openarc #define RES_REGION3 1
#endif

//Decide variables to inject faults
#if _FTVAR0 == 0
#pragma openarc #define FTVAR0  input_units[0:I_SIZE]
#elif _FTVAR0 == 1
#pragma openarc #define FTVAR0  input_weights[0:H_SIZE][0:I_SIZE]
#endif
#if _FTVAR1 == 0
#pragma openarc #define FTVAR1  hidden_units[0:H_SIZE]
#elif _FTVAR1 == 1
#pragma openarc #define FTVAR1  hidden_weights[0:O_SIZE][0:H_SIZE]
#endif
#if _FTVAR2 == 0
#pragma openarc #define FTVAR2  hidden_units[0:H_SIZE]
#elif _FTVAR2 == 1
#pragma openarc #define FTVAR2  hidden_weights[0:O_SIZE][0:H_SIZE]
#elif _FTVAR2 == 2
#pragma openarc #define FTVAR2  hidden_prev_weights[0:O_SIZE][0:H_SIZE]
#elif _FTVAR2 == 3
#pragma openarc #define FTVAR2  output_delta[0:O_SIZE]
#endif
#if _FTVAR3 == 0
#pragma openarc #define FTVAR3  input_units[0:I_SIZE]
#elif _FTVAR3 == 1
#pragma openarc #define FTVAR3  input_weights[0:H_SIZE][0:I_SIZE]
#elif _FTVAR3 == 2
#pragma openarc #define FTVAR3  input_prev_weights[0:H_SIZE][0:I_SIZE]
#elif _FTVAR3 == 3
#pragma openarc #define FTVAR3  hidden_delta[0:H_SIZE]
#endif

//Decide total number of faults to be injected.
#if TOTAL_NUM_FAULTS == 0
#pragma openarc #define TOTAL_NUM_FAULTS    0
#elif TOTAL_NUM_FAULTS == 1
#pragma openarc #define TOTAL_NUM_FAULTS    1
#elif TOTAL_NUM_FAULTS == 2
#pragma openarc #define TOTAL_NUM_FAULTS    2
#elif TOTAL_NUM_FAULTS == 4
#pragma openarc #define TOTAL_NUM_FAULTS    4
#elif TOTAL_NUM_FAULTS == 8
#pragma openarc #define TOTAL_NUM_FAULTS    8
#elif TOTAL_NUM_FAULTS == 16
#pragma openarc #define TOTAL_NUM_FAULTS    16
#elif TOTAL_NUM_FAULTS == 128
#pragma openarc #define TOTAL_NUM_FAULTS    128
#elif TOTAL_NUM_FAULTS == 1024
#pragma openarc #define TOTAL_NUM_FAULTS    1024
#else
#pragma openarc #define TOTAL_NUM_FAULTS    1
#endif

//Decide total number of faulty bits to be changed per fault injection.
#if NUM_FAULTYBITS ==   0
#pragma openarc #define NUM_FAULTYBITS  0
#elif NUM_FAULTYBITS ==   1
#pragma openarc #define NUM_FAULTYBITS  1
#elif NUM_FAULTYBITS == 2
#pragma openarc #define NUM_FAULTYBITS  2
#elif NUM_FAULTYBITS == 4
#pragma openarc #define NUM_FAULTYBITS  4
#elif NUM_FAULTYBITS == 8
#pragma openarc #define NUM_FAULTYBITS  8
#elif NUM_FAULTYBITS == 16
#pragma openarc #define NUM_FAULTYBITS  16
#else
#pragma openarc #define NUM_FAULTYBITS  1
#endif

//Decide number of repeating of a target kernel; with this, the kernel
//execution will be repeated as specified in the clause.
#if NUM_REPEATS ==   0
#pragma openarc #define NUM_REPEATS 0
#elif NUM_REPEATS ==   1
#pragma openarc #define NUM_REPEATS 1
#elif NUM_REPEATS ==   128
#pragma openarc #define NUM_REPEATS 128
#elif NUM_REPEATS ==   1024
#pragma openarc #define NUM_REPEATS 1024
#else
#pragma openarc #define NUM_REPEATS 1
#endif
