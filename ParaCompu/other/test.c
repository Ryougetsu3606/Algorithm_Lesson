#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Serial bubble sort
void bubble_sort_serial(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Parallel bubble sort
void bubble_sort_parallel(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to generate random array
void generate_array(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 10000;
    }
}

// Main function
int main() {
    int increment = 10000;
    int max_size = 100000; // Adjust as needed
    srand(time(NULL));

    printf("Array Size\tSerial Time (s)\tParallel Time (s)\tSpeedup\n");

    for (int size = increment; size <= max_size; size += increment) {
        int *arr_serial = (int *)malloc(size * sizeof(int));
        int *arr_parallel = (int *)malloc(size * sizeof(int));

        generate_array(arr_serial, size);
        for (int i = 0; i < size; i++) {
            arr_parallel[i] = arr_serial[i];
        }

        // Measure serial bubble sort time
        double start_serial = omp_get_wtime();
        bubble_sort_serial(arr_serial, size);
        double end_serial = omp_get_wtime();
        double serial_time = end_serial - start_serial;

        // Measure parallel bubble sort time
        double start_parallel = omp_get_wtime();
        bubble_sort_parallel(arr_parallel, size);
        double end_parallel = omp_get_wtime();
        double parallel_time = end_parallel - start_parallel;

        // Calculate speedup
        double speedup = serial_time / parallel_time;

        printf("%d\t\t%.6f\t\t%.6f\t\t%.2f\n", size, serial_time, parallel_time, speedup);

        free(arr_serial);
        free(arr_parallel);
    }

    return 0;
}
