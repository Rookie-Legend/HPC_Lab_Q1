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
            // Odd-ranked process (excluding rank 0)
            if (rank % 2 == 1 && rank > 0) {
                // Send to left neighbor (rank-1) and receive from it
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
                             received_data.data(), local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, false); // Keep larger half
            }
            // Even-ranked process (excluding rank size-1)
            else if (rank % 2 == 0 && rank < size - 1) {
                // Send to right neighbor (rank+1) and receive from it
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank + 1, 0,
                             received_data.data(), local_n, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, true); // Keep smaller half
            }
        }
        // Even phase
        else {
            // Even-ranked process (excluding rank 0)
            if (rank % 2 == 0 && rank > 0) {
                // Send to left neighbor (rank-1) and receive from it
                std::vector<int> received_data(local_n);
                MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
                             received_data.data(), local_n, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = compare_split(local_data, received_data, false); // Keep larger half
            }
            // Odd-ranked process (excluding rank size-1)
            else if (rank % 2 == 1 && rank < size - 1) {
                // Send to right neighbor (rank+1) and receive from it
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

    // Step 4: Output the result if rank 0
    if (rank == 0) {
        std::cout << "\nSorted array: " << std::endl;
        for (int i = 0; i < array_size; ++i) {
            std::cout << full_array[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
