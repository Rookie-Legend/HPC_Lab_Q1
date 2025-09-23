#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <random>
#include <chrono>  // for timing
#include <iomanip> // for table formatting

// Merges two sorted vectors and keeps either the lower or upper half
std::vector<int> compare_split(std::vector<int>& local_data, std::vector<int>& received_data, bool keep_lower_half) {
    std::vector<int> merged;
    merged.reserve(local_data.size() + received_data.size());
    std::merge(local_data.begin(), local_data.end(),
               received_data.begin(), received_data.end(),
               std::back_inserter(merged));
    if (keep_lower_half) {
        return std::vector<int>(merged.begin(), merged.begin() + local_data.size());
    } else {
        return std::vector<int>(merged.begin() + local_data.size(), merged.end());
    }
}

// Function to check if the array is sorted in non-decreasing order
bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < arr[i - 1]) {
            return false; // Found inversion
        }
    }
    return true;
}

// Function to run one experiment for given array size
double run_experiment(int array_size, int rank, int size) {
    int local_n = array_size / size;
    std::vector<int> local_data(local_n);
    std::vector<int> full_array;

    if (rank == 0) {
        full_array.resize(array_size);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 10000);

        for (int i = 0; i < array_size; ++i) {
            full_array[i] = dis(gen);
        }
    }

    MPI_Scatter(full_array.data(), local_n, MPI_INT,
                local_data.data(), local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    // Each process sorts its local data
    std::sort(local_data.begin(), local_data.end());

    // Odd-Even Transposition Sort
    for (int i = 0; i < size; ++i) {
        if (i % 2 == 0) { // Odd phase
            if (rank % 2 == 1 && rank > 0) {
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
                             received_data.data(), local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, false);
            } else if (rank % 2 == 0 && rank < size - 1) {
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank + 1, 0,
                             received_data.data(), local_n, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, true);
            }
        } else { // Even phase
            if (rank % 2 == 0 && rank > 0) {
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
                             received_data.data(), local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, false);
            } else if (rank % 2 == 1 && rank < size - 1) {
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank + 1, 0,
                             received_data.data(), local_n, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, true);
            }
        }
    }

    MPI_Gather(local_data.data(), local_n, MPI_INT,
               full_array.data(), local_n, MPI_INT,
               0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (rank == 0) {
        if (!is_sorted(full_array)) {
            std::cerr << "Warning: array not sorted for size " << array_size << std::endl;
        }
    }

    return elapsed.count();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << std::setw(15) << "Array Size"
                  << std::setw(20) << "Time (seconds)" << std::endl;
        std::cout << std::string(35, '-') << std::endl;
    }

    // Test for multiple array sizes (multiples of 1000 up to 1e9)
    for (long long array_size = 1000; array_size <= 1000000000; array_size *= 10) {
        double time_taken = run_experiment(array_size, rank, size);

        if (rank == 0) {
            std::cout << std::setw(15) << array_size
                      << std::setw(20) << time_taken << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
