#include "../Utilidades/book.h"

#define N 10

__global__ void add(int *a, int *b, int *c){
    int tidx = blockIdx.x;
    int tidy = blockIdx.y;
    if(tidx < N && tidy < N)
        c[tidx][tidy] = a[tid][tidy] + b[tid][tidy];
}

int main(){
    int a[N][N], b[N][N], c[N][N];
    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

    for(int i = 0; i < N; i++){
        for(int j = 0; i < N; j++){
            a[i][i] = i;
            b[i][j] = i;
        }
    }

    HANDLE_ERROR(cudaMalloc(dev_a,a,N*sizeof(int),cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMalloc(dev_b,b,N*sizeof(int),cudaMemcpyDeviceToHost));

    add<<<N*N,1>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMalloc(dev_c,c,N*sizeof(int),cudaMemcpyDeviceToHost));

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%d + %d = %d\n", a[i][j], b[i][j], c[i][j]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
}