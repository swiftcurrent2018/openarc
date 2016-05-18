#ifndef __TYPE_H__
#define __TYPE_H__

typedef int logical;
#define false   0
#define true    1

typedef struct { 
  double real;
  double imag;
} dcomplex;


#define min(x,y)    ((x) < (y) ? (x) : (y))
#define max(x,y)    ((x) > (y) ? (x) : (y))

#endif //__TYPE_H__
