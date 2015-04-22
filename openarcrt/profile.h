#ifndef __PROFILE_HEADER__

#define __PROFILE_HEADER__
#include <stdarg.h>
#include <stdlib.h>
#include <string>
#include "stdio.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <map>
#include <list>
#include <stack>
#include <set>
#define ACCRTNOINDUCTION AcCnOnE
#define ACCRTREGIONTAG _ACCRT_REGION_
/**typedef enum {
	HI_profile_all = 0;
	HI_profile_occupancy       = 1;
	HI_profile_memorytransfer  = 2;
	HI_profile_instructions    = 3;
	HI_profile_throughput      = 4;
	HI_profile_caching         = 5;
} HI_profilemode_t;

typedef enum {
  HI_none    = 0
  HI_verbose = 1
} HI_profileverbosity_t;
*/
/**
typedef enum {
	HI_uevent_cashmiss = 0;
} HI_userevent_t;
*/
////////////////////////////////////
//Functions used for profile test //
////////////////////////////////////
extern void HI_profile_init(const char* label, const char* fileName);
extern void HI_profile_start(const char* label);
extern void HI_profile_stop(const char* label);
extern void HI_profile_track(const char* label, const char* metricsOfInterest, const char* inductionVariables,bool isRegion);
//extern double HI_profile_get_userevent(HI_userevent_t uevent);
extern void HI_profile_measure_userevent(const char* label, double value);
extern void HI_profile_shutdown(const char* progName);

//[NTD] We need another API to pass track event information to the TAU analysis pass.

#endif
