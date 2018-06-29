#include "../Utilidades/book.h"

// Definimos las dimensiones de la matriz
#define N 500
#define M N

// Definimos type para poder probar con mayor cantidad de numeros o diferentes precisiones
// En este caso queremos probar data mas massiva
#define TYPE char

// Hacemos la suma de la matriz en el device
__global__ void add(TYPE* a, TYPE* b, TYPE* c){
    // Inicializamos la MATRIX_SIZE para reusarla multiples veces sin recalcular la multiplicacion
    const unsigned int MATRIX_SIZE = N * M;
    // Nos colocamos en la primera posicion del arreglo correspondiente al bloque y thread que
    //  estamos usando para calcular el resultado
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Repetimos mientras hayan valores que calcular
    while(tid < MATRIX_SIZE){
        // Calculamos el resultado
        c[tid] = a[tid] + b[tid];
        // Saltamos a la misma posicion del siguiente segmento
        tid += gridDim.x * blockDim.x;
    }
}

int main(void){
    // Inicializamos la MATRIX_SIZE para reusarla multiples veces sin recalcular la multiplicacion
    const unsigned int MATRIX_SIZE = N * M;
    TYPE a[MATRIX_SIZE], b[MATRIX_SIZE], c[MATRIX_SIZE];
    TYPE *dev_a, *dev_b, *dev_c;

    // Se puede paralelizar esto con OpenMP?
    // Inicializamos los arreglos
    for(unsigned int i = 0; i < MATRIX_SIZE; i++){
        a[i] = 1;
        b[i] = 2;
    }
    
    // Reservamos el espacio de memoria en la GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, MATRIX_SIZE * sizeof(TYPE)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, MATRIX_SIZE * sizeof(TYPE)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, MATRIX_SIZE * sizeof(TYPE)));

    // Copiamos los arreglos a la RAM de GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, MATRIX_SIZE * sizeof(TYPE), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, MATRIX_SIZE * sizeof(TYPE), cudaMemcpyHostToDevice));

    // Llamamos al Kernel para ejecutar el codigo en la GPU con lo que se recomienda
    //  en el video: 128x128
    add<<<128,128>>>(dev_a, dev_b, dev_c);

    // Copuamos los resultados a RAM de CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, MATRIX_SIZE * sizeof(TYPE), cudaMemcpyDeviceToHost));

    // Se puede paralelizar esto con OpenMP?
    // Revisamos los resultados
    for(unsigned int i = 0; i < MATRIX_SIZE; i++){
        if(c[i] == 3){
            // printf("POSICION: %d { %d }\n", i, c[i]);
            continue;
        }
        printf("FALLO EN LA POSICION: %d { x: %d, y: %d }\n", i, i%N, i/N);

        // for(unsigned int j = 0; j < N; j++){
        //     for(unsigned int k = 0; k < M; k++)
        //         printf("%d\t", c[j*N + k]);
        //     printf("\n");
        // }

        return 0;
    }

    // Liberamos el espacio de la GPU
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    // Mensaje de Confirmacion
    printf("Se completo para una matriz: { x: %d, y: %d }\n", N, M);

    // Retornamos 0 para que un bash no nos tire error en caso de ser ejecutado a travez de tal
    return 0;
}
