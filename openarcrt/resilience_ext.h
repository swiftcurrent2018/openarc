#ifndef __RESILIENCE_EXT_HEADER__

#define __RESILIENCE_EXT_HEADER__
#include "resilience.h"
#include <stddef.h>
#include <sys/time.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <map>
#include <cstring>
#include <list>


extern void HI_checksum_sum_register(void * target, long int nElems, size_t typeSize, int isIntType, double confVal);
extern void HI_checksum_xor_register(void * target, long int nElems, size_t typeSize, int isIntType, double confVal);
extern void HI_checksum_register(void * target, long int nElems, size_t typeSize, int isIntType, int option, double confVal);
extern void HI_checksum_set(void * target);
extern int HI_checksum_check(void * target);

extern void HI_checkpoint_inmemory_register(void * target, long int nElems, size_t typeSize, int isIntType, double confVal);
extern void HI_checkpoint_register(void * target, long int nElems, size_t typeSize, int isIntType, int option, double confVal);
extern void HI_checkpoint_backup(void * target);
extern void HI_checkpoint_restore(void * target);

///////////////////////////////////////////////
// Unified APIs for fault detection/recovery //
///////////////////////////////////////////////
extern void HI_rsmodule_register(const char *mName, void (*registerF)(void *,long int, size_t, int, double), void (*setF)(void *), int (*checkF)(void *), void (*recoverF)(void *));
extern void HI_rsmodule_REPentry(const char *mName, double R, double E, double P, double C);
extern char * HI_rsmodule_get(const char *mType, double R, double E, double P);
extern void HI_rsdata_register(const char *mName, void *target, long int nElems, size_t typeSize, int isIntType, double C);
extern void HI_rsdata_set(const char *mName, void *target);
extern int HI_rsdata_check(const char *mName, void *target);
extern void HI_rsdata_recover(const char *mName, void *target);

#endif
