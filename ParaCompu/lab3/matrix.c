#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>

void generateRandomMatrix(int8_t **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand();
        }
    }
}

void matrixMultiplySequential(int8_t **A, int8_t **B, int8_t **C, int size) {
    if (size <= 32) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                C[i][j] = 0;
                for (int k = 0; k < size; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return;
    }

    int newSize = size / 2;

    // Allocate submatrices
    int8_t **A11 = A, **A12 = A, **A21 = A + newSize, **A22 = A + newSize;
    int8_t **B11 = B, **B12 = B, **B21 = B + newSize, **B22 = B + newSize;
    int8_t **C11 = C, **C12 = C, **C21 = C + newSize, **C22 = C + newSize;

    // Temporary matrices for intermediate results
    int8_t **T1 = (int8_t **)malloc(newSize * sizeof(int8_t *));
    int8_t **T2 = (int8_t **)malloc(newSize * sizeof(int8_t *));
    for (int i = 0; i < newSize; i++) {
        T1[i] = (int8_t *)malloc(newSize * sizeof(int8_t));
        T2[i] = (int8_t *)malloc(newSize * sizeof(int8_t));
    }

    // Compute C11 = A11*B11 + A12*B21
    matrixMultiplySequential(A11, B11, C11, newSize);
    matrixMultiplySequential(A12, B21, T1, newSize);
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C11[i][j] += T1[i][j];
        }
    }

    // Compute C12 = A11*B12 + A12*B22
    matrixMultiplySequential(A11, B12, C12, newSize);
    matrixMultiplySequential(A12, B22, T1, newSize);
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C12[i][j] += T1[i][j];
        }
    }

    // Compute C21 = A21*B11 + A22*B21
    matrixMultiplySequential(A21, B11, C21, newSize);
    matrixMultiplySequential(A22, B21, T1, newSize);
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C21[i][j] += T1[i][j];
        }
    }

    // Compute C22 = A21*B12 + A22*B22
    matrixMultiplySequential(A21, B12, C22, newSize);
    matrixMultiplySequential(A22, B22, T1, newSize);
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C22[i][j] += T1[i][j];
        }
    }

    // Free temporary matrices
    for (int i = 0; i < newSize; i++) {
        free(T1[i]);
        free(T2[i]);
    }
    free(T1);
    free(T2);
}

void matrixMultiplyParallel(int8_t **A, int8_t **B, int8_t **C, int size, int depth, int maxDepth) {
    if (size <= 32 || depth >= maxDepth) {
        // Base case: perform sequential multiplication for small blocks
        matrixMultiplySequential(A, B, C, size);
        return;
    }

    int newSize = size / 2;

    // Allocate submatrices
    int8_t **A11 = A, **A12 = A, **A21 = A + newSize, **A22 = A + newSize;
    int8_t **B11 = B, **B12 = B, **B21 = B + newSize, **B22 = B + newSize;
    int8_t **C11 = C, **C12 = C, **C21 = C + newSize, **C22 = C + newSize;

    #pragma omp task shared(A11, B11, C11)
    matrixMultiplyParallel(A11, B11, C11, newSize, depth + 1, maxDepth);

    #pragma omp task shared(A12, B21, C11)
    matrixMultiplyParallel(A12, B21, C11, newSize, depth + 1, maxDepth);

    #pragma omp task shared(A11, B12, C12)
    matrixMultiplyParallel(A11, B12, C12, newSize, depth + 1, maxDepth);

    #pragma omp task shared(A12, B22, C12)
    matrixMultiplyParallel(A12, B22, C12, newSize, depth + 1, maxDepth);

    #pragma omp task shared(A21, B11, C21)
    matrixMultiplyParallel(A21, B11, C21, newSize, depth + 1, maxDepth);

    #pragma omp task shared(A22, B21, C21)
    matrixMultiplyParallel(A22, B21, C21, newSize, depth + 1, maxDepth);

    #pragma omp task shared(A21, B12, C22)
    matrixMultiplyParallel(A21, B12, C22, newSize, depth + 1, maxDepth);

    #pragma omp task shared(A22, B22, C22)
    matrixMultiplyParallel(A22, B22, C22, newSize, depth + 1, maxDepth);

    #pragma omp taskwait
}

int main() {
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int n = sizeof(sizes) / sizeof(sizes[0]);
    int epo = 1;

    int numThreads = omp_get_max_threads();
    printf("Number of threads: %d\n", numThreads);
    printf("Number of experiments: %d\n", epo);

    printf("Size\t\tTime\t\tParallel Time\t\tSpeedup\n");
    for (int k = 0; k < n; k++) {
        int size = sizes[k];

        // Allocate matrices
        int8_t **A = (int8_t **)malloc(size * sizeof(int8_t *));
        int8_t **B = (int8_t **)malloc(size * sizeof(int8_t *));
        int8_t **C = (int8_t **)malloc(size * sizeof(int8_t *));
        for (int i = 0; i < size; i++) {
            A[i] = (int8_t *)malloc(size * sizeof(int8_t));
            B[i] = (int8_t *)malloc(size * sizeof(int8_t));
            C[i] = (int8_t *)malloc(size * sizeof(int8_t));
        }

        generateRandomMatrix(A, size);
        generateRandomMatrix(B, size);

        double totalTime = 0.0;
        double totalTimeParallel = 0.0;

        for (int i = 0; i < epo; i++) {
            clock_t start = clock();
            matrixMultiplySequential(A, B, C, size);
            clock_t end = clock();
            totalTime += ((double)(end - start)) / CLOCKS_PER_SEC;

            clock_t startParallel = clock();
            #pragma omp parallel
            {
                #pragma omp single
                matrixMultiplyParallel(A, B, C, size, 0, 8); // Max depth = 8
            }
            clock_t endParallel = clock();
            totalTimeParallel += ((double)(endParallel - startParallel)) / CLOCKS_PER_SEC;
        }

        double speedup = totalTime / totalTimeParallel;
        printf("%d\t\t%.4f\t\t%.4f\t\t\t%.3f\n", size, totalTime / epo, totalTimeParallel / epo, speedup);

        // Free matrices
        for (int i = 0; i < size; i++) {
            free(A[i]);
            free(B[i]);
            free(C[i]);
        }
        free(A);
        free(B);
        free(C);
    }

    return 0;
}