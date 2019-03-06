#include "mcl_accext.h"
#include <map>
#include <unistd.h>
#include <fcntl.h>
//#include <string>

typedef std::map<int, mcl_handle *> mclhandlemap_t;
static mclhandlemap_t* mclhandlemap = new mclhandlemap_t();

int		mcl_acc_wait(int asyncID) {
	int returnV = -1;
	mcl_handle * mclhandle = mclhandlemap->at(asyncID);
	if( mclhandle != NULL ) {
		returnV = mcl_wait(mclhandle);
	}
/*
	if( returnV == 0 ) {
		mclhandlemap->erase(asyncID);
	}
*/
	return returnV;
}

int		mcl_acc_test(int asyncID) {
	int returnV = -1;
	mcl_handle * mclhandle = mclhandlemap->at(asyncID);
	if( mclhandle != NULL ) {
		returnV = mcl_test(mclhandle);
	}
/*
	if( returnV == 0 ) {
		mclhandlemap->erase(asyncID);
	}
*/
	return returnV;
}

int		mcl_on_device(acc_device_t devType) {
	//TODO: add implementation.
	return 1;
}

void	mcl_acc_set_handle(int asyncID, mcl_handle *mclHandle) {
	mclhandlemap->insert(std::pair<int, mcl_handle *>(asyncID, mclHandle));
}

int mcl_load(const char* file, char** src)
{
    int fd;
    size_t n;

    fd = open(file, O_RDONLY);
    if(fd == -1){
        printf("\n Error opening OpenCL code. Aborting.\n");
        goto err;
    }
    n = lseek(fd, 0, SEEK_END);

    *src = (char*) malloc(n+1);
    if(!src){
        printf("Error allocating memory to store OpenCL code. Aborting.\n");
        goto err_file;
    }

    lseek(fd, 0, SEEK_SET);
    if(read(fd, (void*) *src, n) != n){
        printf("Error loading OpenCL code. Aborting.\n");
        goto err_src;
    }

    (*src)[n] = '\0';
#ifdef VERBOSE
    printf("Loaded source, size %lu\n ", n);
#endif
    return 0;

 err_src:
    free(*src);
 err_file:
    close(fd);
 err:
    return -1;
}

