#ifndef MINOS_ACCEXT_H
#define MINOS_ACCEXT_H

extern "C" {
#include "minos.h"
//#include "utils.h"
}

typedef enum {
    acc_device_none = 0,
    acc_device_default = 1,
    acc_device_host = 2,
    acc_device_not_host = 3,
    acc_device_nvidia = 4,
    acc_device_radeon = 5,
    acc_device_gpu = 6,
    acc_device_xeonphi = 7,
    acc_device_current = 8,
    acc_device_altera = 9 
} acc_device_t;


int		mcl_acc_wait(int asyncID);
int		mcl_acc_test(int asyncID);
int		mcl_on_device(acc_device_t devType);
void	mcl_acc_set_handle(int asyncID, mcl_handle *mclHandle);
int		mcl_load(const char* file, char** src);

#endif
