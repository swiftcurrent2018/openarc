#include <omp.h>
#include <openacc.h>

void IMPACC_init(int argc, char** argv, int numdevs);
void IMPACC_shutdown();
int real_main(int argc, char** argv);

int main(int argc, char** argv) {
    int ret;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret);
    if (ret != MPI_THREAD_MULTIPLE) {
        printf("[%s:%d] ret[%d]\n", __FILE__, __LINE__, ret);
        return EXIT_FAILURE;
    }

#pragma omp parallel
    {
#pragma omp master
        IMPACC_init(argc, argv, omp_get_num_threads());

        acc_set_device_num(omp_get_thread_num(), acc_device_nvidia);

        real_main(argc, argv);

#pragma omp master
        IMPACC_shutdown();
    }
    return EXIT_SUCCESS;
}

