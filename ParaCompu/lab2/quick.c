#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#define MAX_PARALLEL_DEPTH 16

void generateRandomArray(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100000;
    }
}

void quickSortSerial(int *arr, int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        int pi = i + 1;
        quickSortSerial(arr, low, pi - 1);
        quickSortSerial(arr, pi + 1, high);
    }
}

void quickSortParallel(int *arr, int low, int high, int depth) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        int pi = i + 1;

        if (depth < MAX_PARALLEL_DEPTH) {
            #pragma omp task shared(arr)
            quickSortParallel(arr, low, pi - 1, depth + 1);

            #pragma omp task shared(arr)
            quickSortParallel(arr, pi + 1, high, depth + 1);

            #pragma omp taskwait
        } else {
            quickSortSerial(arr, low, pi - 1);
            quickSortSerial(arr, pi + 1, high);
        }
    }
}

void parallelQuickSort(int *arr, int low, int high, int numThreads) {
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp single
        quickSortParallel(arr, low, high, 0);
    }
}

int main() {
    int sizes[] = {100000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000};
    int n = sizeof(sizes) / sizeof(sizes[0]);
    int epo = 3;

    int numThreads = omp_get_max_threads();
    printf("Number of threads: %d\n", numThreads);
    printf("MAX_PARALLEL_DEPTH: %d\n", MAX_PARALLEL_DEPTH);
    printf("Number of experiments: %d\n", epo);

    printf("  Size\t\t Time\t\tParallel Time\tSpeedup\n");
    for (int k = 0; k < n; k++) {
        int size = sizes[k];
        int *arr = (int *)malloc(size * sizeof(int));
        if (!arr) {
            printf("Memory allocation failed.\n");
            return 1;
        }

        double totalTime = 0.0;
        double totalTimeParallel = 0.0;

        for (int i = 0; i < epo; i++) {
            generateRandomArray(arr, size);

            clock_t start = clock();
            quickSortSerial(arr, 0, size - 1);
            clock_t end = clock();

            double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
            totalTime += timeTaken;

            generateRandomArray(arr, size);

            clock_t startParallel = clock();
            parallelQuickSort(arr, 0, size - 1, numThreads);
            clock_t endParallel = clock();
            double timeTakenParallel = ((double)(endParallel - startParallel)) / CLOCKS_PER_SEC;
            totalTimeParallel += timeTakenParallel;
        }

        double speedup = totalTime / totalTimeParallel;
        printf("10^%.2f\t\t%.4f\t\t%.4f\t\t%.3f\n", log10(size), totalTime / epo, totalTimeParallel / epo, speedup);

        free(arr);
    }
    return 0;
}