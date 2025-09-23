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

int main(int argc, char** argv) {
MPI_Init(&argc, &argv);

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

const int array_size = 12800; // Total size of the array to sort
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
// Odd-ranked process
if (rank % 2 == 1 && rank < size) {
// Send to left neighbor (rank-1) and receive from it
std::vector<int> received_data(local_n);
MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
received_data.data(), local_n, MPI_INT, rank - 1, 0,
MPI_COMM_WORLD, MPI_STATUS_IGNORE);
local_data = compare_split(local_data, received_data, false); // Keep larger half
}
// Even-ranked process#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper function to compare two integers
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

// Function to perform local sort on each process
void local_sort(int *local_data, int local_size) {
    qsort(local_data, local_size, sizeof(int), compare);
}

// Function to send and receive pivots between processes
void exchange_pivots(int *local_pivots, int *global_pivots, int num_pivots, int rank, int num_processes) {
    MPI_Status status;

    // Step 1: Send pivots to the right process
    if (rank > 0) {
        MPI_Send(local_pivots, num_pivots, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
    }
    if (rank < num_processes - 1) {
        MPI_Send(local_pivots, num_pivots, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    // Step 2: Receive pivots from neighbors
    if (rank > 0) {
        MPI_Recv(global_pivots, num_pivots, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
    }
    if (rank < num_processes - 1) {
        MPI_Recv(global_pivots + num_pivots, num_pivots, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
    }

    // Step 3: Sort global pivots at each process
    qsort(global_pivots, num_pivots * 2, sizeof(int), compare); // Merge pivots from neighbors
}

// Function to redistribute data based on pivots
void redistribute_data(int *local_data, int local_size, int *global_pivots, int num_pivots, int rank, int num_processes) {
    int *new_local_data = (int*)malloc(local_size * sizeof(int));
    int new_local_size = 0;

    // Redistribute the data based on global pivots
    for (int i = 0; i < local_size; i++) {
        int position =#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper function to compare two integers
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

// Function to perform local sort on each process
void local_sort(int *local_data, int local_size) {
    qsort(local_data, local_size, sizeof(int), compare);
}

// Function to send and receive pivots between processes
void exchange_pivots(int *local_pivots, int *global_pivots, int num_pivots, int rank, int num_processes) {
    MPI_Status status;

    // Step 1: Send pivots to the right process
    if (rank > 0) {
        MPI_Send(local_pivots, num_pivots, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
    }
    if (rank < num_processes - 1) {
        MPI_Send(local_pivots, num_pivots, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    // Step 2: Receive pivots from neighbors
    if (rank > 0) {
        MPI_Recv(global_pivots, num_pivots, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
    }
    if (rank < num_processes - 1) {
        MPI_Recv(global_pivots + num_pivots, num_pivots, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
    }

    // Step 3: Sort global pivots at each process
    qsort(global_pivots, num_pivots * 2, sizeof(int), compare); // Merge pivots from neighbors
}

// Function to redistribute data based on pivots
void redistribute_data(int *local_data, int local_size, int *global_pivots, int num_pivots, int rank, int num_processes) {
    int *new_local_data = (int*)malloc(local_size * sizeof(int));
    int new_local_size = 0;

    // Redistribute the data based on global pivots
    for (int i = 0; i < local_size; i++) {
        int position = 0;
        while (position < num_pivots * 2 && local_data[i] > global_pivots[position]) {
            position++;
        }
        new_local_data[position] = local_data[i];
        new_local_size++;
    }

    // Sort the new local data (again local sort within each process)
    qsort(new_local_data, new_local_size, sizeof(int), compare);

    // Copy the result back to the original local data array
    memcpy(local_data, new_local_data, new_local_size * sizeof(int));
    free(new_local_data);
}

int main(int argc, char *argv[]) {
    int rank, num_processes;
    int *data = NULL, *local_data = NULL;
    int size = 100000; // Example array size
    int *global_pivots;  // Declare global_pivots here (no need for initialization)
    int num_pivots = 3; // Number of pivots to use
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    
    int local_size = size / num_processes; // Divide the size by number of processes
    local_data = (int*)malloc(local_size * sizeof(int));
    
    // Rank 0 initializes the data
    if (rank == 0) {
        data = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            data[i] = rand() % 1000; // Random data
        }
    }
    
    // Distribute the data to all processes
    if (rank == 0) {
        for (int i = 1; i < num_processes; i++) {
            MPI_Send(data + i * local_size, local_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        memcpy(local_data, data, local_size * sizeof(int));
    } else {
        MPI_Recv(local_data, local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Step 1: Sort locally
    local_sort(local_data, local_size);
    
    // Step 2: Find and exchange pivots
    int *local_pivots = (int*)malloc(num_pivots * sizeof(int));
    global_pivots = (int*)malloc(num_pivots * 2 * sizeof(int));  // Allocation for global pivots

    // Select local pivots
    local_pivots[0] = local_data[0];
    local_pivots[1] = local_data[local_size / 2];
    local_pivots[2] = local_data[local_size - 1];
    
    // Exchange pivots with neighbors
    exchange_pivots(local_pivots, global_pivots, num_pivots, rank, num_processes);
    
    // Step 3: Redistribute data based on pivots
    redistribute_data(local_data, local_size, global_pivots, num_pivots, rank, num_processes);
    
    // Step 4: Final local sort to ensure global order within each partition
    local_sort(local_data, local_size);
    
    // Gather the final sorted data back to rank 0
    if (rank == 0) {
        memcpy(data, local_data, local_size * sizeof(int));
        for (int i = 1; i < num_processes; i++) {
            MPI_Recv(data + i * local_size, local_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Print the final sorted array (optional)
        for (int i = 0; i < size; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
        free(data);
    } else {
        MPI_Send(local_data, local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Clean up
    free(local_data);
    free(local_pivots);
    free(global_pivots);
    
    MPI_Finalize();
    
    return 0;
}
 0;
        while (position < num_pivots * 2 && local_data[i] > global_pivots[position]) {
            position++;
        }
        new_local_data[position] = local_data[i];
        new_local_size++;
    }

    // Sort the new local data (again local sort within each process)
    qsort(new_local_data, new_local_size, sizeof(int), compare);

    // Copy the result back to the original local data array
    memcpy(local_data, new_local_data, new_local_size * sizeof(int));
    free(new_local_data);
}

int main(int argc, char *argv[]) {
    int rank, num_processes;
    int *data = NULL, *local_data = NULL;
    int size = 100000; // Example array size
    int *global_pivots;  // Declare global_pivots here (no need for initialization)
    int num_pivots = 3; // Number of pivots to use
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    
    int local_size = size / num_processes; // Divide the size by number of processes
    local_data = (int*)malloc(local_size * sizeof(int));
    
    // Rank 0 initializes the data
    if (rank == 0) {
        data = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            data[i] = rand() % 1000; // Random data
        }
    }
    
    // Distribute the data to all processes
    if (rank == 0) {
        for (int i = 1; i < num_processes; i++) {
            MPI_Send(data + i * local_size, local_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        memcpy(local_data, data, local_size * sizeof(int));
    } else {
        MPI_Recv(local_data, local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Step 1: Sort locally
    local_sort(local_data, local_size);
    
    // Step 2: Find and exchange pivots
    int *local_pivots = (int*)malloc(num_pivots * sizeof(int));
    global_pivots = (int*)malloc(num_pivots * 2 * sizeof(int));  // Allocation for global pivots

    // Select local pivots
    local_pivots[0] = local_data[0];
    local_pivots[1] = local_data[local_size / 2];
    local_pivots[2] = local_data[local_size - 1];
    
    // Exchange pivots with neighbors
    exchange_pivots(local_pivots, global_pivots, num_pivots, rank, num_processes);
    
    // Step 3: Redistribute data based on pivots
    redistribute_data(local_data, local_size, global_pivots, num_pivots, rank, num_processes);
    
    // Step 4: Final local sort to ensure global order within each partition
    local_sort(local_data, local_size);
    
    // Gather the final sorted data back to rank 0
    if (rank == 0) {
        memcpy(data, local_data, local_size * sizeof(int));
        for (int i = 1; i < num_processes; i++) {
            MPI_Recv(data + i * local_size, local_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Print the final sorted array (optional)
        for (int i = 0; i < size; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
        free(data);
    } else {
        MPI_Send(local_data, local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Clean up
    free(local_data);
    free(local_pivots);
    free(global_pivots);
    
    MPI_Finalize();
    
    return 0;
}

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
// Even-ranked process
if (rank % 2 == 0 && rank > 0) {
// Send to left neighbor (rank-1) and receive from it
std::vector<int> received_data(local_n);
MPI_Sendrecv(local_data.data(), local_n, MPI_INT, rank - 1, 0,
received_data.data(), local_n, MPI_INT, rank - 1, 0,
MPI_COMM_WORLD, MPI_STATUS_IGNORE);
local_data = compare_split(local_data, received_data, false); // Keep larger half
}
// Odd-ranked process
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
// Gather all local sorted chunks back to the master process
MPI_Gather(local_data.data(), local_n, MPI_INT, full_array.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
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
