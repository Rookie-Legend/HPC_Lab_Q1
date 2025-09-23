#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <random> // For better random number generation

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

// Function to check if an array is sorted in non-decreasing order
bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < arr[i - 1]) {
            return false; // Found an inversion
        }
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int array_size = 1000000; // Total size of the array to sort
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
    }

    // Step 1: Manually distribute data to processes
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            MPI_Send(full_array.data() + i * local_n, local_n, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        std::copy(full_array.begin(), full_array.begin() + local_n, local_data.begin());
    } else {
        MPI_Recv(local_data.data(), local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Step 2: Sort the local data
    std::sort(local_data.begin(), local_data.end());

    // Odd-Even Transposition Sort main loop
    for (int i = 0; i < size; ++i) {
        // Odd phase
        if (i % 2 == 0) {
            if (rank % 2 == 1 && rank > 0) {
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
                             received_data.data(), local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, false); // Keep larger half
            } else if (rank % 2 == 0 && rank < size - 1) {
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
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
                             received_data.data(), local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, false); // Keep larger half
            } else if (rank % 2 == 1 && rank < size - 1) {
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank + 1, 0,
                             received_data.data(), local_n, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, true); // Keep smaller half
            }
        }
    }

    // Step 3: Manually gather the sorted chunks back to the master process
    if (rank == 0) {
        std::copy(local_data.begin(), local_data.end(), full_array.begin());
        for (int i = 1; i < size; ++i) {
            MPI_Recv(full_array.data() + i * local_n, local_n, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_data.data(), local_n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Step 4: Output the result if rank 0 and test correctness
    if (rank == 0) {
        std::cout << "\nSorted array: " << std::endl;
        // Optional: comment out printing large arrays to avoid huge output
        /*
        for (int i = 0; i < array_size; ++i) {
            std::cout << full_array[i] << " ";
        }
        std::cout << std::endl;
        */

        if (is_sorted(full_array)) {
            std::cout << "The array is correctly sorted." << std::endl;
        } else {
            std::cout << "The array is NOT sorted correctly!" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
