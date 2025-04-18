#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <omp.h>

// Function to generate a random vector of given size
std::vector<int> generateRandomVector(size_t size) {
    std::vector<int> vec(size);
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        vec[i] = rand() % 1000000;
    }
    return vec;
}

// Function to measure execution time of a function
template <typename Func>
double measureExecutionTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

// Serial quicksort implementation
void quicksort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int pivot = arr[right];
        int i = left - 1;
        for (int j = left; j < right; ++j) {
            if (arr[j] <= pivot) {
                std::swap(arr[++i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[right]);
        int partitionIndex = i + 1;

        quicksort(arr, left, partitionIndex - 1);
        quicksort(arr, partitionIndex + 1, right);
    }
}

// Parallel quicksort implementation using OpenMP
void parallelQuicksort(std::vector<int>& arr, int left, int right, int depth = 0) {
    if (left < right) {
        int pivot = arr[right];
        int i = left - 1;
        for (int j = left; j < right; ++j) {
            if (arr[j] <= pivot) {
                std::swap(arr[++i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[right]);
        int partitionIndex = i + 1;

        // Parallelize only if depth is below a certain threshold
        if (depth < 4) { // Adjust depth threshold as needed
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelQuicksort(arr, left, partitionIndex - 1, depth + 1);
                #pragma omp section
                parallelQuicksort(arr, partitionIndex + 1, right, depth + 1);
            }
        } else {
            quicksort(arr, left, partitionIndex - 1);
            quicksort(arr, partitionIndex + 1, right);
        }
    }
}

int main() {
    std::cout << "Array Size\tSerial Time (s)\tParallel Time (s)\tSpeedup\n";

    for (size_t size = 100000; size <= 1000000; size += 100000) {
        // Generate random data
        std::vector<int> data = generateRandomVector(size);

        // Measure serial quicksort time
        std::vector<int> serialData = data;
        double serialTime = measureExecutionTime([&]() {
            quicksort(serialData, 0, serialData.size() - 1);
        });

        // Measure parallel quicksort time
        std::vector<int> parallelData = data;
        double parallelTime = measureExecutionTime([&]() {
            parallelQuicksort(parallelData, 0, parallelData.size() - 1);
        });

        // Calculate speedup
        double speedup = serialTime / parallelTime;

        // Output results
        std::cout << size << "\t\t" << serialTime << "\t\t" << parallelTime << "\t\t" << speedup << "\n";
    }

    return 0;
}