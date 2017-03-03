__global__ void MatrixMultiplication_cuda (float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, int M, int N, int P)
{
float sum;
int lwpriv___ti_100_0;
int lwpriv__i;
int lwpriv__j;
int lwpriv__k;
lwpriv___ti_100_0=(threadIdx.x+(blockIdx.x*32));
if (lwpriv___ti_100_0<(M*N))
{
sum=0.0;
lwpriv__j=(lwpriv___ti_100_0%N);
lwpriv__i=(lwpriv___ti_100_0/N);
for (lwpriv__k=0; lwpriv__k<P; lwpriv__k ++ )
{
sum+=(b[((lwpriv__i*P)+lwpriv__k)]*c[((lwpriv__k*N)+lwpriv__j)]);
}
a[((lwpriv__i*N)+lwpriv__j)]=sum;
}
}

