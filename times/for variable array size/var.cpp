#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring> // for memcpy
#include <iomanip> // for table formatting

using clk = std::chrono::high_resolution_clock;
using ns  = std::chrono::duration<double, std::milli>; // ms

// Manual scatter
void manual_scatter(const std::vector<int>& full_data, std::vector<int>& local_data,
                    int rank, int size, int n_per_proc) {
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            MPI_Send(full_data.data() + p * n_per_proc, n_per_proc,
                     MPI_INT, p, 0, MPI_COMM_WORLD);
        }
        std::memcpy(local_data.data(), full_data.data(),
                    n_per_proc * sizeof(int));
    } else {
        MPI_Recv(local_data.data(), n_per_proc, MPI_INT,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Manual gather
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

// Pick samples
std::vector<int> pick_samples(const std::vector<int>& local_data, int s) {
    std::vector<int> samples;
    int stride = local_data.size() / (s + 1);
    for (int i = 1; i <= s; i++) {
        samples.push_back(local_data[i * stride]);
    }
    return samples;
}

// Run one experiment for a given N
std::vector<double> run_experiment(long long N, int rank, int size) {
    int samples_per_proc = size - 1;
    int n_per_proc = N / size;

    std::vector<int> full_array;
    std::vector<int> local_data(n_per_proc);

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

    auto t_local_start = clk::now();
    std::sort(local_data.begin(), local_data.end());
    auto t_local_end = clk::now();
    double local_sort_ms = ns(t_local_end - t_local_start).count();

    std::vector<int> local_samples = pick_samples(local_data, samples_per_proc);
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

    std::vector<int> pivots(size - 1);
    if (rank == 0) {
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 1; i < size; i++) {
            pivots[i - 1] = all_samples[i * samples_per_proc];
        }
        for (int p = 1; p < size; p++) {
            MPI_Send(pivots.data(), size - 1, MPI_INT, p, 2, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(pivots.data(), size - 1, MPI_INT, 0, 2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    auto t_pivot_end = clk::now();
    double pivot_ms = ns(t_pivot_end - t_local_end).count();

    std::vector<std::vector<int>> buckets(size);
    int idx = 0;
    for (int val : local_data) {
        while (idx < size - 1 && val > pivots[idx]) idx++;
        buckets[idx].push_back(val);
    }

    auto t_exchange_start = clk::now();
    std::vector<int> send_sizes(size), recv_sizes(size);
    for (int p = 0; p < size; p++) send_sizes[p] = buckets[p].size();

    for (int p = 0; p < size; p++) {
        if (p == rank) {
            recv_sizes[p] = send_sizes[p];
        } else {
            MPI_Send(&send_sizes[p], 1, MPI_INT, p, 3, MPI_COMM_WORLD);
            MPI_Recv(&recv_sizes[p], 1, MPI_INT, p, 3, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }

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
    for (int p = 0; p < size; p++) {
        if (send_sizes[p] > 0 && p != rank) {
            MPI_Request r;
            MPI_Isend(buckets[p].data(), send_sizes[p],
                      MPI_INT, p, 4, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    }
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    std::vector<int> new_local;
    for (int v : buckets[rank]) new_local.push_back(v);
    for (int p = 0; p < size; p++) {
        if (p != rank) {
            new_local.insert(new_local.end(),
                             recv_bufs[p].begin(), recv_bufs[p].end());
        }
    }

    auto t_exchange_end = clk::now();
    double exchange_ms = ns(t_exchange_end - t_exchange_start).count();

    auto t_merge_start = clk::now();
    std::sort(new_local.begin(), new_local.end());
    auto t_merge_end = clk::now();
    double merge_ms = ns(t_merge_end - t_merge_start).count();

    auto t_gather_start = clk::now();
    full_array.resize(N);
    manual_gather(full_array, new_local, rank, size, n_per_proc);
    auto t_gather_end = clk::now();
    double gather_ms = ns(t_gather_end - t_gather_start).count();

    return {scatter_ms, local_sort_ms, pivot_ms, exchange_ms, merge_ms, gather_ms};
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Problem sizes to test
    std::vector<long long> test_sizes = {100000, 1000000, 5000000, 10000000};

    if (rank == 0) {
        std::cout << std::setw(12) << "N"
                  << std::setw(12) << "Scatter"
                  << std::setw(12) << "LocalSort"
                  << std::setw(12) << "Pivot"
                  << std::setw(12) << "Exchange"
                  << std::setw(12) << "Merge"
                  << std::setw(12) << "Gather" << "\n";
    }

    for (long long N : test_sizes) {
        MPI_Barrier(MPI_COMM_WORLD); // sync before each run
        std::vector<double> times = run_experiment(N, rank, size);

        if (rank == 0) {
            std::cout << std::setw(12) << N;
            for (double t : times) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(2) << t;
            }
            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
