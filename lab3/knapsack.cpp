#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <functional>
#include <chrono>
#include <queue>
void DnC(int n, int C, const std::vector<int>& weights, const std::vector<int>& values);
void DP(int n, int C, const std::vector<int>& weights, const std::vector<int>& values);
void Greedy(int n, int C, const std::vector<int>& weights, const std::vector<int>& values);
void Backtracking(int n, int C, const std::vector<int>& weights, const std::vector<int>& values);
void BRnBO(int n, int C, const std::vector<int>& weights, const std::vector<int>& values);
int main() {
    // Array of seeds
    std::vector<int> seeds = {114, 514, 42};
    // Array of n values
    std::vector<int> n_values = {5, 10, 20, 30, 40, 50, 100, 200};

    for (int seed : seeds) {
        std::srand(seed); // Set the random seed

        for (int n : n_values) {
            // Maximum capacity of the knapsack
            int C = 10 * n + std::rand() % (10 * n); // Random capacity between 10n and 20n

            // Vectors to store weights and values of items
            std::vector<int> weights(n);
            std::vector<int> values(n);

            // Generate random weights and values for each item
            for (int i = 0; i < n; ++i) {
                weights[i] = 4 + std::rand() % 37; // Random weight between 4 and 40
                values[i] = 10 + std::rand() % 91; // Random value between 10 and 100
            }

            // Output the current experiment parameters
            std::cout << "Seed: " << seed << ", Number of items: " << n << ", Knapsack capacity: " << C << std::endl;

            // Run the algorithms
            DnC(n, C, weights, values);
            DP(n, C, weights, values);
            Greedy(n, C, weights, values);
            Backtracking(n, C, weights, values);
            BRnBO(n, C, weights, values);
        }
    }

    return 0;
}

void DnC(int n, int C, const std::vector<int>& weights, const std::vector<int>& values) {
    auto start = std::chrono::high_resolution_clock::now();

    // Define a time limit for the divide and conquer process
    const double timeLimit = 10.0;

    // Helper function for divide and conquer
    std::function<int(int, int)> knapsack = [&](int index, int remainingCapacity) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start;

        // Terminate if the time limit is exceeded
        if (elapsed.count() > timeLimit) {
            return 0;
        }

        // Base case: no items left or no remaining capacity
        if (index == n || remainingCapacity <= 0) {
            return 0;
        }

        // If the current item's weight is more than the remaining capacity, skip it
        if (weights[index] > remainingCapacity) {
            return knapsack(index + 1, remainingCapacity);
        }

        // Otherwise, consider both including and excluding the current item
        int exclude = knapsack(index + 1, remainingCapacity);
        int include = values[index] + knapsack(index + 1, remainingCapacity - weights[index]);

        // Return the maximum of both choices
        return std::max(exclude, include);
    };

    // Call the helper function starting from the first item and full capacity
    int optimalValue = knapsack(0, C);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the results
    std::cout << "Divide and Conquer" << std::endl;
    std::cout << "Optimal value: " << optimalValue << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
    std::cout << std::endl;
}

void DP(int n, int C, const std::vector<int>& weights, const std::vector<int>& values) {
    auto start = std::chrono::high_resolution_clock::now();

    // Create a DP table
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(C + 1, 0));

    // Fill the DP table
    for (int i = 1; i <= n; ++i) {
        for (int w = 0; w <= C; ++w) {
            if (weights[i - 1] <= w) {
                dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    // The optimal value is in dp[n][C]
    int optimalValue = dp[n][C];

    // Trace back to find the items included in the optimal solution
    std::vector<int> selectedItems;
    int remainingCapacity = C;
    for (int i = n; i > 0 && remainingCapacity > 0; --i) {
        if (dp[i][remainingCapacity] != dp[i - 1][remainingCapacity]) {
            selectedItems.push_back(i - 1);
            remainingCapacity -= weights[i - 1];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the results
    std::cout << "Dynamic Programming" << std::endl;
    std::cout << "Optimal value: " << optimalValue << std::endl;
    if (n < 20) {
        std::cout << "Selected items (0-based index): ";
        for (int item : selectedItems) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
    std::cout << std::endl;
}

void Greedy(int n, int C, const std::vector<int>& weights, const std::vector<int>& values) {
    auto start = std::chrono::high_resolution_clock::now();

    // Create a vector of items with value-to-weight ratio
    std::vector<std::pair<double, int>> valuePerWeight(n);
    for (int i = 0; i < n; ++i) {
        valuePerWeight[i] = {static_cast<double>(values[i]) / weights[i], i};
    }

    // Sort items by value-to-weight ratio in descending order
    std::sort(valuePerWeight.rbegin(), valuePerWeight.rend());

    // Select items greedily
    int totalValue = 0;
    int remainingCapacity = C;
    std::vector<int> selectedItems;
    for (const auto& item : valuePerWeight) {
        int index = item.second;
        if (weights[index] <= remainingCapacity) {
            selectedItems.push_back(index);
            totalValue += values[index];
            remainingCapacity -= weights[index];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the results
    std::cout << "Greedy Algorithm" << std::endl;
    std::cout << "Optimal value: " << totalValue << std::endl;
    if (n < 20) {
        std::cout << "Selected items (0-based index): ";
        for (int item : selectedItems) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
    std::cout << std::endl;
}

void Backtracking(int n, int C, const std::vector<int>& weights, const std::vector<int>& values) {
    auto start = std::chrono::high_resolution_clock::now();

    int optimalValue = 0;
    std::vector<int> selectedItems;
    std::vector<int> currentItems;
    int currentValue = 0;
    int currentWeight = 0;

    // Define a time limit for the backtracking process
    const double timeLimit = 10.0;

    // Precompute the maximum possible value from the current index onwards
    std::vector<int> maxValueFrom(n + 1, 0);
    for (int i = n - 1; i >= 0; --i) {
        maxValueFrom[i] = maxValueFrom[i + 1] + values[i];
    }

    // Helper function for backtracking with pruning
    std::function<void(int)> backtrack = [&](int index) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start;

        // Terminate if the time limit is exceeded
        if (elapsed.count() > timeLimit) {
            return;
        }

        // If we've considered all items, update the optimal solution
        if (index == n) {
            if (currentValue > optimalValue) {
                optimalValue = currentValue;
                selectedItems = currentItems;
            }
            return;
        }

        // Prune if the current value plus the maximum possible value from here is not better
        if (currentValue + maxValueFrom[index] <= optimalValue) {
            return;
        }

        // Exclude the current item
        backtrack(index + 1);

        // Include the current item if it doesn't exceed capacity
        if (currentWeight + weights[index] <= C) {
            currentItems.push_back(index);
            currentWeight += weights[index];
            currentValue += values[index];

            backtrack(index + 1);

            // Backtrack
            currentItems.pop_back();
            currentWeight -= weights[index];
            currentValue -= values[index];
        }
    };

    // Start backtracking from the first item
    backtrack(0);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the results
    std::cout << "Backtracking" << std::endl;
    std::cout << "Optimal value: " << optimalValue << std::endl;
    if (n < 20) {
        std::cout << "Selected items (0-based index): ";
        for (int item : selectedItems) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
    std::cout << std::endl;
}

void BRnBO(int n, int C, const std::vector<int>& weights, const std::vector<int>& values) {
    auto start = std::chrono::high_resolution_clock::now();

    struct Node {
        int level;
        int value;
        int weight;
        double bound;
        std::vector<int> selectedItems;
    };

    auto calculateBound = [&](const Node& node) -> double {
        if (node.weight >= C) return 0;

        double bound = node.value;
        int totalWeight = node.weight;
        for (int i = node.level + 1; i < n; ++i) {
            if (totalWeight + weights[i] <= C) {
                totalWeight += weights[i];
                bound += values[i];
            } else {
                bound += (C - totalWeight) * static_cast<double>(values[i]) / weights[i];
                break;
            }
        }
        return bound;
    };

    std::vector<std::pair<double, int>> valuePerWeight(n);
    for (int i = 0; i < n; ++i) {
        valuePerWeight[i] = {static_cast<double>(values[i]) / weights[i], i};
    }
    std::sort(valuePerWeight.rbegin(), valuePerWeight.rend());

    std::vector<int> sortedWeights(n), sortedValues(n);
    for (int i = 0; i < n; ++i) {
        sortedWeights[i] = weights[valuePerWeight[i].second];
        sortedValues[i] = values[valuePerWeight[i].second];
    }

    std::priority_queue<Node, std::vector<Node>, std::function<bool(const Node&, const Node&)>> pq(
        [](const Node& a, const Node& b) { return a.bound < b.bound; });

    Node root = { -1, 0, 0, calculateBound({-1, 0, 0, 0, {}}), {} };
    pq.push(root);

    int optimalValue = 0;
    std::vector<int> optimalItems;

    // Define a time limit for the algorithm
    const double timeLimit = 10.0;

    while (!pq.empty()) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start;

        // Terminate if the time limit is exceeded
        if (elapsed.count() > timeLimit) {
            break;
        }

        Node current = pq.top();
        pq.pop();

        // Prune nodes with bound less than or equal to the current optimal value
        if (current.bound <= optimalValue) continue;

        Node next = current;
        next.level++;

        if (next.level < n) {
            // Include the current item
            next.weight = current.weight + sortedWeights[next.level];
            next.value = current.value + sortedValues[next.level];
            next.selectedItems = current.selectedItems;
            next.selectedItems.push_back(valuePerWeight[next.level].second);

            if (next.weight <= C && next.value > optimalValue) {
                optimalValue = next.value;
                optimalItems = next.selectedItems;
            }

            next.bound = calculateBound(next);
            if (next.bound > optimalValue) {
                pq.push(next);
            }

            // Exclude the current item
            next.weight = current.weight;
            next.value = current.value;
            next.selectedItems = current.selectedItems;
            next.bound = calculateBound(next);
            if (next.bound > optimalValue) {
                pq.push(next);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Branch and Bound" << std::endl;
    std::cout << "Optimal value: " << optimalValue << std::endl;
    if (n < 20) {
        std::cout << "Selected items (0-based index): ";
        for (int item : optimalItems) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
    std::cout << std::endl;
}