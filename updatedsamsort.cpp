
#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <random> // for better random number generation

// Merges two sorted vectors and keeps either the lower or upper half
std::vector<int> compare_split(std::vector<int>& local_data, std::vector<int>& received_data, bool keep_lower_half) {
    std::vector<int> merged;
    merged.reserve(local_data.size() + received_data.size());
    std::merge(local_data.begin(), local_data.end(), received_data.begin(), received_data.end(), std::back_inserter(merged));
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int array_size = 128000; // Total size of the array to sort
    std::vector<int> full_array;
    int local_n = array_size / size; // Number of elements per process
    std::vector<int> local_data(local_n);

    if (rank == 0) {
        full_array.resize(array_size);

        // Improved randomization using C++11 <random>
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 10000);

        for (int i = 0; i < array_size; ++i) {
            full_array[i] = dis(gen);
        }

        std::cout << "Unsorted array: " << std::endl;
        for (int i = 0; i < array_size; ++i) {
            std::cout << full_array[i] << " ";
        }
        std::cout << std::endl;
    }

    // Scatter the initial data to all processes
    MPI_Scatter(full_array.data(), local_n, MPI_INT, local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its local data
    std::sort(local_data.begin(), local_data.end());

    // Odd-Even Transposition Sort main loop
    for (int i = 0; i < size; ++i) {
        // Odd phase
        if (i % 2 == 0) {
            if (rank % 2 == 1 && rank > 0) {
                // Odd-ranked process sends to left neighbor
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
                             received_data.data(), local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, false); // Keep larger half
            }
            else if (rank % 2 == 0 && rank < size - 1) {
                // Even-ranked process sends to right neighbor
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank + 1, 0,
                             received_data.data(), local_n, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, true); // Keep smaller half
            }
        }
        // Even phase
        else {
            if (rank % 2 == 0 && rank > 0) {
                // Even-ranked process sends to left neighbor
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
                             received_data.data(), local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, false); // Keep larger half
            }
            else if (rank % 2 == 1 && rank < size - 1) {
                // Odd-ranked process sends to right neighbor
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank + 1, 0,
                             received_data.data(), local_n, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, true); // Keep smaller half
            }
        }
    }

    // Gather all local sorted chunks back to the master process
    MPI_Gather(local_data.data(), local_n, MPI_INT, full_array.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\nSorted array: " << std::endl;
        for (int i = 0; i < array_size; ++i) {
            std::cout << full_array[i] << " ";
        }
        std::cout << std::endl;

        // Check if the array is sorted correctly
        if (is_sorted(full_array)) {
            std::cout << "The array is correctly sorted." << std::endl;
        } else {
            std::cout << "The array is NOT sorted correctly!" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
