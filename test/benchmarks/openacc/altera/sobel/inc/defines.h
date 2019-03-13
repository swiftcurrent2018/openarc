
// Directive versions
#define WINDOW   0
#define ND_RANGE 1

#define AOCL_ALIGNMENT 64

#define ROWS 1080
#ifdef _OPENARC_
#pragma openarc #define ROWS 1080
#endif

#define COLS 1920
#ifdef _OPENARC_
#pragma openarc #define COLS 1920 
#endif

#ifndef MAX 
#ifndef __cplusplus
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#endif

#ifndef MIN 
#ifndef __cplusplus
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif
#endif

#ifndef  __cplusplus
#define abs(x) ((x)<0 ? -(x) : (x))
#endif
