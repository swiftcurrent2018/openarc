#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Multi-threaded 
#define ND_UPDATE 0
#define ND_REDUCE 0

// Single-threaded
#define SWI_UPDATE 0
#define SWI_REDUCE 1

#define SWI_UPDATE_PIPE 0

#define DEBUG
#define	ITERATION

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

#ifndef _ROWS_
#define _ROWS_ 256 
#endif
#ifndef _COLS_
#define _COLS_ 256
#endif

#define _SIZE_I_ (_ROWS_ * _COLS_)

#ifdef _OPENARC_

#if _ROWS_ == 256
#pragma openarc #define _ROWS_ 256 
#elif _ROWS_ == 1024
#pragma openarc #define _ROWS_ 1024
#elif _ROWS_ == 2048
#pragma openarc #define _ROWS_ 2048
#elif _ROWS_ == 4096
#pragma openarc #define _ROWS_ 4096
#endif

#if _COLS_ == 256 
#pragma openarc #define _COLS_ 256
#elif _COLS_ == 1024
#pragma openarc #define _COLS_ 1024
#elif _COLS_ == 2048
#pragma openarc #define _COLS_ 2048
#elif _COLS_ == 4096
#pragma openarc #define _COLS_ 4096
#endif

#pragma openarc #define _SIZE_I_ (_ROWS_ * _COLS_)

#endif
