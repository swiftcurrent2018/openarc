#ifndef __IMPACC_UTILS_H__
#define __IMPACC_UTILS_H__

#include <mpi.h>

namespace impacc {

static size_t MPI_size(int count, MPI_Datatype datatype) {
    size_t bytes = 0;
    switch (datatype) {
        case MPI_CHAR:
        case MPI_SIGNED_CHAR:
        case MPI_UNSIGNED_CHAR:
        case MPI_BYTE:              bytes = sizeof(char);   break;
        case MPI_SHORT:
        case MPI_UNSIGNED_SHORT:    bytes = sizeof(short);  break;
        case MPI_INT:
        case MPI_UNSIGNED:          bytes = sizeof(int);    break;
        case MPI_LONG:
        case MPI_UNSIGNED_LONG:     bytes = sizeof(long);   break;
        case MPI_FLOAT:             bytes = sizeof(float);  break;
        case MPI_DOUBLE:            bytes = sizeof(double); break;
        default: _error("not support type[%x]", datatype);
    }
    return count * bytes;
}

static int MPI_tag(int tag, int src, int dst) {
    return tag == MPI_ANY_TAG ? tag : tag * 7 + src * 11 + dst * 17; 
}

}

#endif
