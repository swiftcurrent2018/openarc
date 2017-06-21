////////////////////////////////////////
// Functions used for resilience test //
////////////////////////////////////////
#include "resilience_ext.h"

#define _DEBUG_FTPRINT_ON_ 1

typedef std::map<void *, int> cs_optionmap_t;
typedef std::map<void *, long> cs_countermap_t;
typedef std::map<void *, type64bS> cs_intchecksummap_t;
typedef std::map<void *, double> cs_floatchecksummap_t;
typedef std::map<void *, type64b> cs_xorchecksummap_t;
typedef std::map<void *, void *> cp_checkpointmap_t;
typedef std::map<const char *, void (*)(void *, long int, size_t, int, double)> rsmodule_registermap_t;
typedef std::map<const char *, void (*)(void *)> rsmodule_setmap_t;
typedef std::map<const char *, int (*)(void *)> rsmodule_checkmap_t;
typedef std::map<const char *, void (*)(void *)> rsmodule_recovermap_t;
typedef std::map<void *, size_t> rsdata_datatypesizemap_t;
typedef std::map<void *, double> rsdata_confvalmap_t;

static int HI_srand_set = 0;
//interal maps for checksum
static cs_optionmap_t cs_optionmap;
static cs_countermap_t cs_sizemap;
static cs_countermap_t cs_strtidxmap;
static cs_intchecksummap_t cs_intchecksummap;
static cs_floatchecksummap_t cs_sumchecksummap;
static cs_xorchecksummap_t cs_xorchecksummap;
//interal maps for checkpoint
static cs_optionmap_t cp_optionmap;
static cs_countermap_t cp_sizemap;
static cp_checkpointmap_t cp_checkpointmap;
//internal maps for unified fault detection/recovery
static rsmodule_registermap_t rsmodule_registermap;
static rsmodule_setmap_t rsmodule_setmap;
static rsmodule_checkmap_t rsmodule_checkmap;
static rsmodule_recovermap_t rsmodule_recovermap;
static rsdata_datatypesizemap_t rsdata_datatypesizemap;
static cs_optionmap_t rsdata_datatypemap;
static rsdata_confvalmap_t rsdata_confvalmap;

//Data and functions used for unified fault detection/recovery backend.
enum policy_t {MAXR, MAXP, MINE, DECREASEE, INCREASER, INCREASEP, DECREASEP};

typedef class REP_Model {
	protected:
	std::map<int, double> R;
	std::map<int, double> E;
	std::map<int, double> P;
	std::map<int, double> C;
	double currentR;
	double currentE;
	double currentP;
	double currentC;
	int current_PL;
	policy_t current_policy;

	int findMinLoc(std::map<int, double>* inMap) {
		int minLoc = INT_MIN;
		double minVal = DBL_MAX;
		for( std::map<int, double>::iterator it=inMap->begin(); it!=inMap->end(); ++it ) {
			if( it->second < minVal ) {
				minLoc = it->first;
				minVal = it->second;
			}	
		}
		return minLoc;
	}

	int findMaxLoc(std::map<int, double>* inMap) {
		int maxLoc = INT_MIN;
		double maxVal = DBL_MIN;
		for( std::map<int, double>::iterator it=inMap->begin(); it!=inMap->end(); ++it ) {
			if( it->second > maxVal ) {
				maxLoc = it->first;
				maxVal = it->second;
			}	
		}
		return maxLoc;
	}

	public:
	void add_REP_entry(int pl, double tR, double tE, double tP) {
		R[pl] = tR;
		E[pl] = tE;
		P[pl] = tP;
		maximize_P();
	}

	void add_REP_entry(int pl, double tR, double tE, double tP, double tC) {
		R[pl] = tR;
		E[pl] = tE;
		P[pl] = tP;
		C[pl] = tC;
		maximize_P();
	}

	void printStatusTable() {
		printf("PL	R	E	P	C\n");
		for( std::map<int, double>::iterator it=R.begin(); it!=R.end(); ++it ) {
			if( C.count(it->first) > 0 ) {
				printf("%d\t%lf\t%lf\t%lf\t%lf\n", it->first, it->second, E[it->first], P[it->first], C[it->first]);
			} else {
				printf("%d\t%lf\t%lf\t%lf\n", it->first, it->second, E[it->first], P[it->first]);
			}
		}
	}

	int maximize_R() {
		int tPL = findMaxLoc(&R);
		update_state(tPL);
		return tPL;
	}

	int maximize_P() {
		int tPL = findMaxLoc(&P);
		update_state(tPL);
		return tPL;
	}

	int minimize_E() {
		int tPL = findMinLoc(&E);
		update_state(tPL);
		return tPL;
	}

	int change_state(const char * action, std::map<int, double>* inMap) {
		double current_val = inMap->at(current_PL);
		int newPL = current_PL;
		double new_val = 0.0; //default value in chapel
		if( strcmp(action, "INCREASE") == 0 ) {
			for( std::map<int, double>::iterator it=inMap->begin(); it!=inMap->end(); ++it ) {
				if( it->second > current_val) {
					if( newPL == current_PL ) {
						newPL = it->first;
						new_val = it->second;
					} else if( it->second < new_val ) {
						newPL = it->first;
						new_val = it->second;
					}
				}
			}
		} else if( strcmp(action, "DECREASE") == 0 ) {
			for( std::map<int, double>::iterator it=inMap->begin(); it!=inMap->end(); ++it ) {
				if( it->second < current_val) {
					if( newPL == current_PL ) {
						newPL = it->first;
						new_val = it->second;
					} else if( it->second > new_val ) {
						newPL = it->first;
						new_val = it->second;
					}
				}
			}
		}
		update_state(newPL);
		return newPL;
	}

	int get_PL(double tR, double tE, double tP) {
		if( tR == DBL_MAX ) {
			return maximize_R();
		} else if( tP == DBL_MAX ) {
			return maximize_P();
		} else if( tE == DBL_MIN ) {
			return minimize_E();
		} else {
			return maximize_P();
		}
	}

	int get_PL() {
		return current_PL;
	}

	int get_R() {
		return currentR;
	}

	int get_E() {
		return currentE;
	}

	int get_P() {
		return currentP;
	}

	void update_state(int tPL) {
		current_PL = tPL;
		currentR = R[tPL];
		currentE = E[tPL];
		currentP = P[tPL];
		currentC = C[tPL];
	}
} REP_Model_t;

typedef std::map<const char *, REP_Model_t> rsmodule_pgrmodulemap_t;
typedef std::map<const char *, std::list<REP_Model_t *> > rsmodule_moduletypemap_t;
static rsmodule_pgrmodulemap_t pgrModules;
static rsmodule_moduletypemap_t moduleTypes;

void HI_set_srand() {
    struct timeval time;
    gettimeofday(&time, 0); 
    unsigned int seed = time.tv_sec*time.tv_usec;
    srand(seed);
	//printf("execute HI_set_srand() with seed %d\n", seed);
	HI_srand_set = 1;
}

type8b HI_genbitvector8b(int numFaults) {
    int j;
    unsigned int bit;
    type8b bitVector = 0;
	type8b tbitvec = 1;
    double numBits = 8.0;
	if( HI_srand_set == 0 ) {
		HI_set_srand();
	}
    for( j=0; j<numFaults; j++ ) { 
        bit = (unsigned int)(numBits * rand()/(RAND_MAX + 1.0));
        bitVector |= (tbitvec << bit);
    }   
    return bitVector;
}

type16b HI_genbitvector16b(int numFaults) {
    int j;
    unsigned int bit;
    type16b bitVector = 0;
	type16b tbitvec = 1;
    double numBits = 16.0;
	if( HI_srand_set == 0 ) {
		HI_set_srand();
	}
    for( j=0; j<numFaults; j++ ) { 
        bit = (unsigned int)(numBits * rand()/(RAND_MAX + 1.0));
        bitVector |= (tbitvec << bit);
    }   
    return bitVector;
}

type32b HI_genbitvector32b(int numFaults) {
    int j;
    unsigned int bit;
    type32b bitVector = 0;
	type32b tbitvec = 1;
    double numBits = 32.0;
	if( HI_srand_set == 0 ) {
		HI_set_srand();
	}
    for( j=0; j<numFaults; j++ ) { 
        bit = (unsigned int)(numBits * rand()/(RAND_MAX + 1.0));
        bitVector |= (tbitvec << bit);
    }   
    return bitVector;
}

type64b HI_genbitvector64b(int numFaults) {
    int j;
    unsigned int bit;
    type64b bitVector = 0;
	type64b tbitvec = 1;
    double numBits = 64.0;
	if( HI_srand_set == 0 ) {
		HI_set_srand();
	}
    for( j=0; j<numFaults; j++ ) { 
        bit = (unsigned int)(numBits * rand()/(RAND_MAX + 1.0));
        bitVector |= (tbitvec << bit);
    }   
    return bitVector;
}

unsigned long int HI_genrandom_int(unsigned long int Range) {
    unsigned long int rInt;
    double dRange = (double)Range;
	if( HI_srand_set == 0 ) {
		printf("set srand in HI_genrandom_int()\n");
		HI_set_srand();
	}
    rInt = (unsigned long int)(dRange * rand()/(RAND_MAX + 1.0));
    return rInt;
}

void HI_sort_int( unsigned int* iArray, int iSize ) {
	int i, j;
	unsigned int tmp;
	int middle;
	int left, right;
	for( i=1; i<iSize; ++i ) {
		tmp = iArray[i];
		left = 0;
		right = i;
		while (left < right) {
			middle = (left + right)/2;
			if( tmp >= iArray[middle] ) {
				left = middle + 1;
			} else {
				right = middle;
			}	
		}	
		for ( j=i; j>left; --j ) {
			//swap(j-1, j);
			tmp = iArray[j-1];
			iArray[j-1] = iArray[j];
			iArray[j] = tmp;
		}
	} 
}

void HI_ftinjection_int8b(type8b * target, int ftinject,  long int epos, type8b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int8b data\n");
#endif
    }   
}

void HI_ftinjection_int16b(type16b * target, int ftinject,  long int epos, type16b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int16b data\n");
#endif
    }   
}

void HI_ftinjection_int32b(type32b * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int32b data\n");
#endif
    }   
}

void HI_ftinjection_int64b(type64b * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
        *(target+epos) ^= bitvec; 
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int64b data\n");
#endif
    }   
}

/*
void HI_ftinjection_float(float * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
        type32b val = (type32b)(*(target+epos));
        val ^= bitvec;
        *(target+epos) = (float)val;
    }   
}

void HI_ftinjection_double(double * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
        type64b val = (type64b)(*(target+epos));
        val ^= bitvec;
        *(target+epos) = (double)val;
    }   
}
*/

void HI_ftinjection_float(float * target, int ftinject,  long int epos, type32b bitvec) {
    if( ftinject != 0 ) { 
		FloatBits val;
        val.f = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.f;
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for float data\n");
#endif
    }   
}

void HI_ftinjection_double(double * target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) { 
		DoubleBits val;
        val.d = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.d;
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for double data\n");
#endif
    }   
}

void HI_ftinjection_pointer(void ** target, int ftinject,  long int epos, type64b bitvec) {
    if( ftinject != 0 ) {
		PointerBits val;
        val.p = *(target+epos);
        val.i ^= bitvec;
        *(target+epos) = val.p;
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for pointer data\n");
#endif
    }
}

type8b HI_ftinject_val_int1b(type8b target, int ftinject) {
    if( ftinject != 0 ) {
        target ^= 1;
#ifdef _DEBUG_FTPRINT_ON_
		fprintf(stderr, "====> Fault injected for int1b data\n");
#endif
    }
    return target;
}

type8b HI_ftinject_val_int8b(type8b target, int ftinject,  type8b bitvec) {
    HI_ftinjection_int8b(&target, ftinject, 0, bitvec);
    return target;
}

type16b HI_ftinject_val_int16b(type16b target, int ftinject,  type16b bitvec) {
    HI_ftinjection_int16b(&target, ftinject, 0, bitvec);
    return target;
}

type32b HI_ftinject_val_int32b(type32b target, int ftinject,  type32b bitvec) {
    HI_ftinjection_int32b(&target, ftinject, 0, bitvec);
    return target;
}

type64b HI_ftinject_val_int64b(type64b target, int ftinject,  type64b bitvec) {
    HI_ftinjection_int64b(&target, ftinject, 0, bitvec);
    return target;
}

float HI_ftinject_val_float(float target, int ftinject,  type32b bitvec) {
    HI_ftinjection_float(&target, ftinject, 0, bitvec);
    return target;
}

double HI_ftinject_val_double(double target, int ftinject,  type64b bitvec) {
    HI_ftinjection_double(&target, ftinject, 0, bitvec);
    return target;
}

void *HI_ftinject_val_pointer(void *target, int ftinject,  type64b bitvec) {
    HI_ftinjection_pointer(&target, ftinject, 0, bitvec);
    return target;
}

void HI_checksum_sum_register(void * target, long int nElems, size_t typeSize, int isIntType, double confVal) {
	rsdata_confvalmap[target] = confVal;
	rsdata_datatypemap[target] = isIntType;
	rsdata_datatypesizemap[target] = typeSize;
	cs_optionmap[target] = 0;
	if( (confVal <= 0.0) || (confVal >= 1.0) ) {
		cs_strtidxmap[target] = 0;
		cs_sizemap[target] = nElems;
	} else {
		long int cNElems = (long int)(nElems*confVal);
		unsigned long int numIntervals = 0;
		if( cNElems > 0 ) {
			numIntervals = nElems/cNElems;
		}	
		unsigned long int strtIdx = HI_genrandom_int(numIntervals);
		//strtIdx contains an element index.
		strtIdx = strtIdx*cNElems;
		if( strtIdx >= nElems ) {
			strtIdx = nElems - 1;
		}
		cs_strtidxmap[target] = strtIdx;
		if( strtIdx + cNElems > nElems ) {
			cNElems = nElems - strtIdx -1;
		}
		//cNElems is in element numbers.
		cs_sizemap[target] = cNElems;
	}
	cs_sumchecksummap[target] = 0.0; //set initial value	
}

void HI_checksum_xor_register(void * target, long int nElems, size_t typeSize, int isIntType, double confVal) {
	rsdata_confvalmap[target] = confVal;
	rsdata_datatypemap[target] = isIntType;
	rsdata_datatypesizemap[target] = typeSize;
	cs_optionmap[target] = 1;
	if( (confVal <= 0.0) || (confVal >= 1.0) ) {
		cs_strtidxmap[target] = 0;
		cs_sizemap[target] = nElems;
	} else {
		long int cNElems = (long int)(nElems*confVal);
		unsigned long int numIntervals = 0;
		if( cNElems > 0 ) {
			numIntervals = nElems/cNElems;
		}	
		unsigned long int strtIdx = HI_genrandom_int(numIntervals);
		//strtIdx contains an element index.
		strtIdx = strtIdx*cNElems;
		if( strtIdx >= nElems ) {
			strtIdx = nElems - 1;
		}
		cs_strtidxmap[target] = strtIdx;
		if( strtIdx + cNElems > nElems ) {
			cNElems = nElems - strtIdx -1;
		}
		//cNElems is in element numbers.
		cs_sizemap[target] = cNElems;
	}
	cs_xorchecksummap[target] = 0; //set initial value	
}

void HI_checksum_register(void * target, long int nElems, size_t typeSize, int isIntType, int option, double confVal) {
	if( option < 0 ) {
		//[TODO] checksum mode is selected by runtime or others.
		//option should be updated here.
		option = 0;
	}
	if( option == 0 ) {
		HI_checksum_sum_register(target, nElems, typeSize, isIntType, confVal);
	} else if( option == 1 ) {
		HI_checksum_xor_register(target, nElems, typeSize, isIntType, confVal);
	}
}

template<typename T>
void HI_checksum_set_intT(T target) {
	long int size;
	unsigned long int strtidx;
	int option;
	long int i;
	if( (cs_sizemap.count(target) == 0) || (cs_strtidxmap.count(target) == 0) || (cs_optionmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_set_intT()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		strtidx = cs_strtidxmap[target];
		option = cs_optionmap[target];
		if( option == 0 ) {
			type64b checksum = 0;
			for( i=0; i<size; i++ ) {
				checksum += (type64bS)(*(target+i+strtidx));
			}
			cs_sumchecksummap[target] = (double)checksum;
		} else if( option == 1 ) {
			type64b checksum = 0;
			for( i=0; i<size; i++ ) {
				checksum ^= (type64b)(*(target+i+strtidx));
			}
			cs_xorchecksummap[target] = checksum;
		}
	}
}

template<typename T>
void HI_checksum_set_floatT(T target) {
	long int size;
	unsigned long int strtidx;
	int option;
	long int i;
	if( (cs_sizemap.count(target) == 0) || (cs_strtidxmap.count(target) == 0) || (cs_optionmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_set_floatT()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		strtidx = cs_strtidxmap[target];
		option = cs_optionmap[target];
		if( option == 0 ) {
			double checksum = 0;
			for( i=0; i<size; i++ ) {
				checksum += double(*(target+i+strtidx));
			}
			cs_sumchecksummap[target] = checksum;
		} else if( option == 1 ) {
			type64b checksum = 0;
			DoubleBits val;
			for( i=0; i<size; i++ ) {
        		val.d = (double)(*(target+i+strtidx));
				checksum ^= val.i;
			}
			cs_xorchecksummap[target] = checksum;
		}
	}
}

void HI_checksum_set(void *target) {
	long int size;
	unsigned long int strtidx;
	int option;
	double confVal;
	size_t typeSize;
	int isIntType;
	if( (cs_sizemap.count(target) == 0) || (cs_strtidxmap.count(target) == 0) || (cs_optionmap.count(target) == 0)  || (rsdata_confvalmap.count(target) == 0) ||
(rsdata_datatypemap.count(target) == 0) || (rsdata_datatypesizemap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_set()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		strtidx = cs_strtidxmap[target];
		option = cs_optionmap[target];
		confVal = rsdata_confvalmap[target];
		isIntType = rsdata_datatypemap[target];
		typeSize = rsdata_datatypesizemap[target];
		if( isIntType == 1 ) { //int type
			if( typeSize == 1 ) {
				HI_checksum_set_intT<type8b *>((type8b *) target);	
			} else if( typeSize == 2 ) {
				HI_checksum_set_intT<type16b *>((type16b *) target);	
			} else if( typeSize == 4 ) {
				HI_checksum_set_intT<type32b *>((type32b *) target);	
			} else if( typeSize == 8 ) {
				HI_checksum_set_intT<type64b *>((type64b *) target);	
			}
		} else { //float type
			if( typeSize == 4 ) {
				HI_checksum_set_floatT<float *>((float *) target);	
			} else {
				HI_checksum_set_floatT<double *>((double *) target);	
			}
		}
	}
}

template<typename T>
int HI_checksum_check_intT(T target) {
	int error = 0;
	long int size;
	unsigned long int strtidx;
	int option;
	long int i;
	if( (cs_sizemap.count(target) == 0) || (cs_optionmap.count(target) == 0) ||
		(cs_strtidxmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_check_intT()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		strtidx = cs_strtidxmap[target];
		option = cs_optionmap[target];
		if( option == 0 ) {
			type64bS checksum = 0;
			double checksum_old = 0;
			double checksum_new = 0.0;
			checksum_old = cs_sumchecksummap[target];
			for( i=0; i<size; i++ ) {
				checksum += (type64bS)(*(target+i+strtidx));
			}
			checksum_new = (double)checksum;
			if( checksum_new != checksum_old ) {
				error = 1;
			}
			cs_sumchecksummap[target] = checksum_new;
		} else if( option == 1 ) {
			type64b checksum = 0;
			type64b checksum_old = 0;
			checksum_old = cs_xorchecksummap[target];
			for( i=0; i<size; i++ ) {
				checksum ^= (type64b)(*(target+i+strtidx));
			}
			if( checksum != checksum_old ) {
				error = 1;
			}
			cs_xorchecksummap[target] = checksum;
		}
	}
	return error;
}

template<typename T>
int HI_checksum_check_floatT(T target) {
	int error = 0;
	long int size;
	unsigned long int strtidx;
	int option;
	long int i;
	if( (cs_sizemap.count(target) == 0) || (cs_optionmap.count(target) == 0) ||
		(cs_strtidxmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_check_floatT()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		strtidx = cs_strtidxmap[target];
		option = cs_optionmap[target];
		if( option == 0 ) {
			double checksum = 0;
			double checksum_old = 0;
			checksum_old = cs_sumchecksummap[target];
			for( i=0; i<size; i++ ) {
				checksum += double(*(target+i+strtidx));
			}
			if( checksum != checksum_old ) {
				error = 1;
			}
			cs_sumchecksummap[target] = checksum;
		} else if( option == 1 ) {
			type64b checksum = 0;
			type64b checksum_old = 0;
			DoubleBits val;
			checksum_old = cs_xorchecksummap[target];
			for( i=0; i<size; i++ ) {
        		val.d = (double)(*(target+i+strtidx));
				checksum ^= val.i;
			}
			if( checksum != checksum_old ) {
				error = 1;
			}
			cs_xorchecksummap[target] = checksum;
		}
	}
	return error;
}

int HI_checksum_check(void *target) {
	long int size;
	unsigned long int strtidx;
	int option;
	double confVal;
	size_t typeSize;
	int isIntType;
	int error = 0;
	if( (cs_sizemap.count(target) == 0) || (cs_strtidxmap.count(target) == 0) || (cs_optionmap.count(target) == 0)  || (rsdata_confvalmap.count(target) == 0) ||
(rsdata_datatypemap.count(target) == 0) || (rsdata_datatypesizemap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checksum_check()]\n");
		exit(1);
	} else {
		size = cs_sizemap[target];
		strtidx = cs_strtidxmap[target];
		option = cs_optionmap[target];
		confVal = rsdata_confvalmap[target];
		isIntType = rsdata_datatypemap[target];
		typeSize = rsdata_datatypesizemap[target];
		if( isIntType == 1 ) { //int type
			if( typeSize == 1 ) {
				error = HI_checksum_check_intT<type8b *>((type8b *)target);	
			} else if( typeSize == 2 ) {
				error = HI_checksum_check_intT<type16b *>((type16b *)target);	
			} else if( typeSize == 4 ) {
				error = HI_checksum_check_intT<type32b *>((type32b *)target);	
			} else if( typeSize == 8 ) {
				error = HI_checksum_check_intT<type64b *>((type64b *)target);	
			}
		} else { //float type
			if( typeSize == 4 ) {
				error = HI_checksum_check_floatT<float *>((float *)target);	
			} else {
				error = HI_checksum_check_floatT<double *>((double *)target);	
			}
		}
	}
#ifdef _DEBUG_FTPRINT_ON_
	if( error != 0 ) {
		if( isIntType == 1 ) {
			fprintf(stderr, "====> Checksum Error detected on int data!\n");
		} else {
			fprintf(stderr, "====> Checksum Error detected on float data!\n");
		}
	}
#endif
	return error;
}


void HI_checkpoint_inmemory_register(void * target, long int nElems, size_t typeSize, int isIntType, double confVal) {
	cp_optionmap[target] = 0;
	//size in bytes
	cp_sizemap[target] = nElems*typeSize;
	void *cp_data = malloc(nElems*typeSize);
	cp_checkpointmap[target] = cp_data; //map target to checkpoint data
	//fprintf(stderr, "HI_checkpoint_register() is called\n");
}

void HI_checkpoint_register(void * target, long int nElems, size_t typeSize, int isIntType, int option, double confVal) {
	//size in bytes
	rsdata_confvalmap[target] = confVal;
	rsdata_datatypemap[target] = isIntType;
	rsdata_datatypesizemap[target] = typeSize;
	if( option < 0 ) {
		//[TODO] checkpoint mode is selected by runtime or others.
		//option should be updated here.
		option = 0;
	}
	cp_optionmap[target] = option;
	if( option == 0 ) {
		HI_checkpoint_inmemory_register(target,nElems, typeSize, isIntType, confVal); 
	}
}

void HI_checkpoint_backup(void * target) {
	long int size;
	int option;
	void *cp_data;
	if( (cp_sizemap.count(target) == 0) || (cp_optionmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checkpoint_backup()]\n");
		exit(1);
	} else {
		//fprintf(stderr, "HI_checkpoint_backup() is called\n");
		size = cp_sizemap[target];
		option = cp_optionmap[target];
		cp_data = cp_checkpointmap[target];	
		if( option == 0 ) {
			memcpy(cp_data, target, size);
		}
	}
}

void HI_checkpoint_restore(void * target) {
	long int size;
	int option;
	void *cp_data;
	if( (cp_sizemap.count(target) == 0) || (cp_optionmap.count(target) == 0) ) {
		fprintf(stderr, "[ERROR in HI_checkpoint_restore()]\n");
		exit(1);
	} else {
		//fprintf(stderr, "HI_checkpoint_restore() is called\n");
		size = cp_sizemap[target];
		option = cp_optionmap[target];
		cp_data = cp_checkpointmap[target];	
		if( option == 0 ) {
			memcpy(target, (void *)cp_data, size);
		}
	}
}

///////////////////////////////////////////////
// Unified APIs for fault detection/recovery //
///////////////////////////////////////////////

void HI_rsmodule_register(const char *mName, void (*registerF)(void *,long int, size_t, int, double), void (*setF)(void *), int (*checkF)(void *), void (*recoverF)(void *)) {
	if( mName == NULL ) {
		fprintf(stderr, "[ERROR in HI_rsmodule_register()] NULL for mName; exit\n");
		exit(1);
	}
	if( registerF != NULL ) {
		rsmodule_registermap[mName] = registerF;
	}
	if( registerF != NULL ) {
		rsmodule_setmap[mName] = setF;
	}
	if( registerF != NULL ) {
		rsmodule_checkmap[mName] = checkF;
	}
	if( registerF != NULL ) {
		rsmodule_recovermap[mName] = recoverF;
	}
}

void HI_rsmodule_REPentry(const char *mName, double R, double E, double P, double C) {
	if( mName == NULL ) {
		fprintf(stderr, "[ERROR in HI_rsmodule_REPentry()] NULL for mName; exit\n");
		exit(1);
	}
}

char * HI_rsmodule_get(const char *mType, double R, double E, double P) {
	char * mName = NULL;
	return mName;
} 

void HI_rsdata_register(const char *mName, void *target, long int nElems, size_t typeSize, int isIntType, double C) {
	void (*registerF)(void *, long int, size_t, int, double);
	if( mName == NULL ) {
		fprintf(stderr, "[ERROR in HI_rsdata_register()] NULL for mName; exit\n");
		exit(1);
	}
	if( rsmodule_registermap.count(mName) == 0 ) {
		fprintf(stderr, "[ERROR in HI_rsdata_register()] register function does not exist for module %s; exit\n", mName);
		exit(1);
	} else {
		registerF = rsmodule_registermap[mName];
		registerF(target, nElems, typeSize, isIntType, C);
	}
}

void HI_rsdata_set(const char *mName, void *target) {
	void (*setF)(void *);
	if( mName == NULL ) {
		fprintf(stderr, "[ERROR in HI_rsdata_set()] NULL for mName; exit\n");
		exit(1);
	}
	if( rsmodule_setmap.count(mName) == 0 ) {
		return;
	} else {
		setF = rsmodule_setmap[mName];
		setF(target);
	}
}

int HI_rsdata_check(const char *mName, void *target) {
	int (*checkF)(void *);
	if( mName == NULL ) {
		fprintf(stderr, "[ERROR in HI_rsdata_check()] NULL for mName; exit\n");
		exit(1);
	}
	if( rsmodule_recovermap.count(mName) == 0 ) {
		return 0;
	} else {
		checkF = rsmodule_checkmap[mName];
		return checkF(target);
	}
}

void HI_rsdata_recover(const char *mName, void *target) {
	void (*recoverF)(void *);
	if( mName == NULL ) {
		fprintf(stderr, "[ERROR in HI_rsdata_recover()] NULL for mName; exit\n");
		exit(1);
	}
	if( rsmodule_recovermap.count(mName) == 0 ) {
		return;
	} else {
		recoverF = rsmodule_recovermap[mName];
		recoverF(target);
	}
}

