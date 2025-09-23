
#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <random>
#include <chrono>
#include <iomanip>

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

// Function to check if an array is sorted in non-decreasing order
bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

void run_experiment(int array_size, int rank, int size) {
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

    double t_scatter = 0, t_localsort = 0, t_pivot = 0, t_exchange = 0, t_merge = 0, t_gather = 0;

    // Scatter (manual send/recv)
    auto t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            MPI_Send(full_array.data() + i * local_n, local_n, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        std::copy(full_array.begin(), full_array.begin() + local_n, local_data.begin());
    } else {
        MPI_Recv(local_data.data(), local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    t_scatter = std::chrono::duration<double>(t2 - t1).count();

    // Local sort
    t1 = std::chrono::high_resolution_clock::now();
    std::sort(local_data.begin(), local_data.end());
    t2 = std::chrono::high_resolution_clock::now();
    t_localsort = std::chrono::duration<double>(t2 - t1).count();

    // Pivot phase (not used here)
    t_pivot = 0.0;

    // Odd-Even Transposition Sort
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i) {
        if (i % 2 == 0) {
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
        } else {
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
    t2 = std::chrono::high_resolution_clock::now();
    t_exchange = std::chrono::duration<double>(t2 - t1).count();

    // Merge phase (not used here)
    t_merge = 0.0;

    // Gather (manual send/recv)
    t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::copy(local_data.begin(), local_data.end(), full_array.begin());
        for (int i = 1; i < size; ++i) {
            MPI_Recv(full_array.data() + i * local_n, local_n, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_data.data(), local_n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    t2 = std::chrono::high_resolution_clock::now();
    t_gather = std::chrono::duration<double>(t2 - t1).count();

    if (rank == 0) {
        if (!is_sorted(full_array)) {
            std::cerr << "Warning: array not sorted for size " << array_size << std::endl;
        }

        std::cout << std::setw(10) << array_size
                  << std::setw(10) << std::fixed << std::setprecision(2) << t_scatter
                  << std::setw(12) << t_localsort
                  << std::setw(10) << t_pivot
                  << std::setw(12) << t_exchange
                  << std::setw(10) << t_merge
                  << std::setw(10) << t_gather
                  << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << std::setw(10) << "N"
                  << std::setw(10) << "Scatter"
                  << std::setw(12) << "LocalSort"
                  << std::setw(10) << "Pivot"
                  << std::setw(12) << "Exchange"
                  << std::setw(10) << "Merge"
                  << std::setw(10) << "Gather"
                  << std::endl;
    }

    // Try multiple input sizes like in your screenshot
    for (long long n = 100000; n <= 10000000; n *= 10) {
        run_experiment(n, rank, size);
    }

    MPI_Finalize();
    return 0;
}
