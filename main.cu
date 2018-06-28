#include "Utilidades/book.h"

#define N 10

__global__ void add(int *a, int *b, int *c){
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y*N;
    
    //if(offset < N)
    c[offset] = a[offset] + b[offset];
    //if(tid < N)
    //    c[tid] = a[tid] + b[tid];
}

int main(){
    int matriz_leng = N*N;

    int a[matriz_leng], b[matriz_leng], c[matriz_leng];
    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, matriz_leng*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, matriz_leng*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, matriz_leng*sizeof(int)));

    for(int i = 0; i < matriz_leng; i++){
        a[i] = 2;
        b[i] = 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, matriz_leng*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, matriz_leng*sizeof(int), cudaMemcpyHostToDevice));

    add<<<matriz_leng,1>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, matriz_leng*sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    for(int i = 0; i < matriz_leng; i++){
        //printf("%d + %d = %d\n", a[i], b[i], c[i]);
        if(c[i] != 3){
            printf("FALLO EN LA POSICION: %d", i);
            return 0;
        }
    }

    printf("DONE!");

    return 0;
}