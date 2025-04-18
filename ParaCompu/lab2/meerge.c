#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

void generateRandomArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        srand(time(NULL));
        arr[i] = rand() % 10000;
    }
}

void parallelMergeSort(int arr[], int left, int right, int numThreads) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if (numThreads > 1) {
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, left, mid, numThreads / 2);

                #pragma omp section
                parallelMergeSort(arr, mid + 1, right, numThreads / 2);
            }
        } else {
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}
// void parallelMergeSort(int arr[], int left, int right, int numThreads) {
//     if (left < right) {
//         int mid = left + (right - left) / 2;

//         if (numThreads > 1) {
//             #pragma omp parallel sections
//             {
//                 #pragma omp section
//                 parallelMergeSort(arr, left, mid, numThreads / 2);

//                 #pragma omp section
//                 parallelMergeSort(arr, mid + 1, right, numThreads / 2);
//             }
//         } else {
//             mergeSort(arr, left, mid);
//             mergeSort(arr, mid + 1, right);
//         }

//         merge(arr, left, mid, right);
//     }
// }
int main() {
    int sizes[]={5000, 10000, 20000, 50000, 100000, 500000, 1000000};
    int n = sizeof(sizes) / sizeof(sizes[0]);
    int epo = 3;
    
    int numThreads = omp_get_max_threads();
    printf("Number of threads: %d\n", numThreads);

    printf("Size\t  Time\t\tParallel Time\tSpeedup\n");
    for(int k=0; k < n; k++){
        int size = sizes[k];
        // printf("Size of the array: %d\n", size);
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
                mergeSort(arr, 0, size - 1);
                clock_t end = clock();

                double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
                // printf("Experiment %d: Time taken = %f seconds\n", i + 1, timeTaken);
                totalTime += timeTaken;

                clock_t startParallel = clock();
                parallelMergeSort(arr, 0, size - 1, numThreads);
                clock_t endParallel = clock();
                double timeTakenParallel = ((double)(endParallel - startParallel)) / CLOCKS_PER_SEC;
                // printf("Parallel Experiment %d: Time taken = %f seconds\n", i + 1, timeTakenParallel);
                totalTimeParallel += timeTakenParallel;
            }
            
            // printf("Average time taken: %f seconds\n", totalTime / epo);
            // printf("Average parallel time taken: %f seconds\n", totalTimeParallel / epo);
            double speedup = totalTime / totalTimeParallel;
            // printf("Speedup: %f\n\n", speedup);
            printf("%d\t%f\t%f\t%f\n", size, totalTime / epo, totalTimeParallel / epo, speedup);

        free(arr);
    }
    return 0;
}