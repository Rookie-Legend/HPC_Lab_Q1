#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring> // for memcpy

using clk = std::chrono::high_resolution_clock;
using ns  = std::chrono::duration<double, std::milli>; // milliseconds

// Helper: scatter manually using point-to-point
void manual_scatter(const std::vector<int>& full_data, std::vector<int>& local_data,
                    int rank, int size, int n_per_proc) {
    if (rank == 0) {
        // Send chunks to others
        for (int p = 1; p < size; p++) {
            MPI_Send(full_data.data() + p * n_per_proc, n_per_proc,
                     MPI_INT, p, 0, MPI_COMM_WORLD);
        }
        // Copy own chunk
        std::memcpy(local_data.data(), full_data.data(),
                    n_per_proc * sizeof(int));
    } else {
        MPI_Recv(local_data.data(), n_per_proc, MPI_INT,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Manual gather back to rank 0
void manual_gather(std::vector<int>& full_data, const std::vector<int>& local_data,
                   int rank, int size, int n_per_proc) {
    if (rank == 0) {
        std::memcpy(full_data.data(), local_data.data(),
                    n_per_proc * sizeof(int));
        for (int p = 1; p < size; p++) {
            MPI_Recv(full_data.data() + p * n_per_proc, n_per_proc, MPI_INT,
                     p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_data.data(), n_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

// Each process selects s samples uniformly from its local sorted array
std::vector<int> pick_samples(const std::vector<int>& local_data, int s) {
    std::vector<int> samples;
    int stride = local_data.size() / (s + 1);
    for (int i = 1; i <= s; i++) {
        samples.push_back(local_data[i * stride]);
    }
    return samples;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // -------------------- Parameters --------------------
    long long N = 1000000; // total number of elements (adjust e.g. 1e9 if memory allows)
    int samples_per_proc = size - 1; // rule of thumb
    int n_per_proc = N / size;

    std::vector<int> full_array;
    std::vector<int> local_data(n_per_proc);

    // -------------------- Generate & Scatter --------------------
    if (rank == 0) {
        full_array.resize(N);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dis(0, 1000000);
        for (long long i = 0; i < N; i++) {
            full_array[i] = dis(gen);
        }
    }

    auto t_scatter_start = clk::now();
    manual_scatter(full_array, local_data, rank, size, n_per_proc);
    auto t_scatter_end = clk::now();
    double scatter_ms = ns(t_scatter_end - t_scatter_start).count();

    // -------------------- Local sort --------------------
    auto t_local_start = clk::now();
    std::sort(local_data.begin(), local_data.end());
    auto t_local_end = clk::now();
    double local_sort_ms = ns(t_local_end - t_local_start).count();

    // -------------------- Pick samples --------------------
    std::vector<int> local_samples = pick_samples(local_data, samples_per_proc);

    // Gather samples to root (manual)
    std::vector<int> all_samples;
    if (rank == 0) {
        all_samples.resize(samples_per_proc * size);
        std::memcpy(all_samples.data(), local_samples.data(),
                    samples_per_proc * sizeof(int));
        for (int p = 1; p < size; p++) {
            MPI_Recv(all_samples.data() + p * samples_per_proc, samples_per_proc,
                     MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_samples.data(), samples_per_proc, MPI_INT,
                 0, 1, MPI_COMM_WORLD);
    }

    // -------------------- Choose pivots & broadcast --------------------
    std::vector<int> pivots(size - 1);
    if (rank == 0) {
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 1; i < size; i++) {
            pivots[i - 1] = all_samples[i * samples_per_proc];
        }
        // send pivots to all others
        for (int p = 1; p < size; p++) {
            MPI_Send(pivots.data(), size - 1, MPI_INT, p, 2, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(pivots.data(), size - 1, MPI_INT, 0, 2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    auto t_pivot_end = clk::now();
    double pivot_ms = ns(t_pivot_end - t_local_end).count();

    // -------------------- Partition local data into buckets --------------------
    std::vector<std::vector<int>> buckets(size);
    int idx = 0;
    for (int val : local_data) {
        while (idx < size - 1 && val > pivots[idx]) idx++;
        buckets[idx].push_back(val);
    }

    // -------------------- Exchange buckets --------------------
    auto t_exchange_start = clk::now();

    // Send sizes first
    std::vector<int> send_sizes(size), recv_sizes(size);
    for (int p = 0; p < size; p++) send_sizes[p] = buckets[p].size();

    // Exchange sizes with all processes
    for (int p = 0; p < size; p++) {
        if (p == rank) {
            recv_sizes[p] = send_sizes[p];
        } else {
            MPI_Send(&send_sizes[p], 1, MPI_INT, p, 3, MPI_COMM_WORLD);
            MPI_Recv(&recv_sizes[p], 1, MPI_INT, p, 3, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }

    // Post nonblocking receives
    std::vector<MPI_Request> reqs;
    std::vector<std::vector<int>> recv_bufs(size);
    for (int p = 0; p < size; p++) {
        if (recv_sizes[p] > 0) {
            recv_bufs[p].resize(recv_sizes[p]);
            if (p != rank) {
                MPI_Request r;
                MPI_Irecv(recv_bufs[p].data(), recv_sizes[p],
                          MPI_INT, p, 4, MPI_COMM_WORLD, &r);
                reqs.push_back(r);
            }
        }
    }

    // Send buckets
    for (int p = 0; p < size; p++) {
        if (send_sizes[p] > 0 && p != rank) {
            MPI_Request r;
            MPI_Isend(buckets[p].data(), send_sizes[p],
                      MPI_INT, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }

    // Wait all
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Merge received + own bucket
    std::vector<int> new_local;
    new_local.reserve(n_per_proc);
    for (int v : buckets[rank]) new_local.push_back(v);
    for (int p = 0; p < size; p++) {
        if (p != rank) {
            new_local.insert(new_local.end(),
                             recv_bufs[p].begin(), recv_bufs[p].end());
        }
    }

    auto t_exchange_end = clk::now();
    double exchange_ms = ns(t_exchange_end - t_exchange_start).count();

    // -------------------- Final local sort --------------------
    auto t_merge_start = clk::now();
    std::sort(new_local.begin(), new_local.end());
    auto t_merge_end = clk::now();
    double merge_ms = ns(t_merge_end - t_merge_start).count();

    // -------------------- Gather result --------------------
    auto t_gather_start = clk::now();
    full_array.resize(N);
    manual_gather(full_array, new_local, rank, size, n_per_proc);
    auto t_gather_end = clk::now();
    double gather_ms = ns(t_gather_end - t_gather_start).count();

    // -------------------- Print timings --------------------
    if (rank == 0) {
        std::cout << "Scatter: " << scatter_ms << " ms\n";
        std::cout << "Local sort: " << local_sort_ms << " ms\n";
        std::cout << "Pivoting: " << pivot_ms << " ms\n";
        std::cout << "Data exchange: " << exchange_ms << " ms\n";
        std::cout << "Final merge sort: " << merge_ms << " ms\n";
        std::cout << "Gather: " << gather_ms << " ms\n";

        // Optional: print first 20 sorted values
        std::cout << "First 20 sorted: ";
        for (int i = 0; i < std::min<long long>(20, N); i++) {
            std::cout << full_array[i] << " ";
        }
        std::cout << "...\n";
    }

    MPI_Finalize();
    return 0;
}
