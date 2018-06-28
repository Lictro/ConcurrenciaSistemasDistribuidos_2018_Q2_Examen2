#include "Utilidades/book.h"

#define N 10

__global__ void add(int *a, int *b, int *c){
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y*N;
    
    if(offset < N)
        c[offset] = a[offset] + b[offset];
    //if(tid < N)
    //    c[tid] = a[tid] + b[tid];
}

int main(){
    int a[N*N], b[N*N], c[N*N];
    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

    for(int i = 0; i < N*N; i++){
        a[i] = i;
        b[i] = i;
    }

    HANDLE_ERROR(cudaMalloc(dev_a, a, N*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMalloc(dev_b, b, N*sizeof(int), cudaMemcpyDeviceToHost));

    add<<<N*N,1>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMalloc(dev_c, c, N*sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < N*N; i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
}