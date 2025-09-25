#include <mpi.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>

using namespace std;

// Map C++ types to MPI types
template<typename T>
MPI_Datatype MPITypeFrom();
template<> MPI_Datatype MPITypeFrom<int>() { return MPI_INT; }

// Check global sortedness using neighbor-to-neighbor communication
template <typename T>
bool checkGlobalSortedness(MPI_Comm comm, const vector<T>& localData, int myRank, int p) {
    if (localData.empty()) return true;

    T localFirst = localData.front();
    T localLast  = localData.back();
    T prevLast = T();
    bool isSorted = true;

    MPI_Status status;
    if (myRank < p - 1) {
        MPI_Send(&localLast, 1, MPITypeFrom<T>(), myRank + 1, 0, comm);
    }
    if (myRank > 0) {
        MPI_Recv(&prevLast, 1, MPITypeFrom<T>(), myRank - 1, 0, comm, &status);
        if (prevLast > localFirst) isSorted = false;
    }

    // Propagate result to rank 0 using a chain of sends
    int sortedInt = isSorted ? 1 : 0;
    if (myRank > 0) {
        MPI_Send(&sortedInt, 1, MPI_INT, 0, 1, comm);
    } else {
        // rank 0 collects results from all other ranks
        int val;
        for (int r = 1; r < p; r++) {
            MPI_Recv(&val, 1, MPI_INT, r, 1, comm, MPI_STATUS_IGNORE);
            if (val == 0) sortedInt = 0;
        }
    }

    return sortedInt == 1;
}

// Deterministic generator: each rank generates its local chunk
vector<int> generateLocalChunk(uint64_t globalSize, int maxValue, int myRank, int p) {
    uint64_t chunkSize = globalSize / p;
    vector<int> arr(chunkSize);
    mt19937_64 gen(myRank + 123456789ULL);
    uniform_int_distribution<int> dist(0, maxValue);
    for (uint64_t i = 0; i < chunkSize; i++)
        arr[i] = dist(gen);
    return arr;
}

// Point-to-point parallel sort
void parallelSortP2P(MPI_Comm comm, vector<int>& data, int p, int myRank) {
    double startTime = MPI_Wtime();

    // -------------------- STEP 1: SAMPLE SELECTION --------------------
    const int a = (int)(16 * log(p) / log(2.0));
    mt19937 rndEngine(myRank + 987654321ULL);
    uniform_int_distribution<size_t> idxGen(0, data.size() - 1);
    vector<int> localSamples;
    for (int i = 0; i <= a; i++)
        localSamples.push_back(data[idxGen(rndEngine)]);

    // -------------------- STEP 2: SEND SAMPLES TO ROOT --------------------
    vector<int> allSamples;
    if (myRank == 0) {
        allSamples.resize(localSamples.size() * p);
        copy(localSamples.begin(), localSamples.end(), allSamples.begin());
        for (int r = 1; r < p; r++) {
            MPI_Recv(allSamples.data() + r * localSamples.size(), localSamples.size(),
                     MPI_INT, r, 0, comm, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(localSamples.data(), localSamples.size(), MPI_INT, 0, 0, comm);
    }

    // -------------------- STEP 3: ROOT SELECTS SPLITTERS --------------------
    vector<int> splitters(p - 1);
    if (myRank == 0) {
        sort(allSamples.begin(), allSamples.end());
        for (int i = 0; i < p - 1; i++)
            splitters[i] = allSamples[(a + 1) * (i + 1)];
        // send splitters to all ranks
        for (int r = 1; r < p; r++)
            MPI_Send(splitters.data(), p - 1, MPI_INT, r, 2, comm);
    } else {
        MPI_Recv(splitters.data(), p - 1, MPI_INT, 0, 2, comm, MPI_STATUS_IGNORE);
    }

    // -------------------- STEP 4: BUCKET LOCAL DATA --------------------
    vector<vector<int>> buckets(p);
    for (auto& b : buckets) b.reserve(data.size() / p * 2);
    for (int val : data) {
        int idx = upper_bound(splitters.begin(), splitters.end(), val) - splitters.begin();
        buckets[idx].push_back(val);
    }

    // -------------------- STEP 5: SEND BUCKETS TO RESPECTIVE RANKS --------------------
    vector<vector<int>> recvBuckets(p);
    for (int r = 0; r < p; r++) {
        if (r == myRank) {
            recvBuckets[r] = buckets[r]; // keep own bucket
        } else {
            // send bucket to rank r
            int sendSize = buckets[r].size();
            MPI_Send(&sendSize, 1, MPI_INT, r, 3, comm);
            if (sendSize > 0)
                MPI_Send(buckets[r].data(), sendSize, MPI_INT, r, 4, comm);

            // receive bucket from rank r
            int recvSize;
            MPI_Recv(&recvSize, 1, MPI_INT, r, 3, comm, MPI_STATUS_IGNORE);
            recvBuckets[r].resize(recvSize);
            if (recvSize > 0)
                MPI_Recv(recvBuckets[r].data(), recvSize, MPI_INT, r, 4, comm, MPI_STATUS_IGNORE);
        }
    }

    // -------------------- STEP 6: MERGE RECEIVED BUCKETS --------------------
    vector<int> sortedLocal;
    for (auto& b : recvBuckets)
        sortedLocal.insert(sortedLocal.end(), b.begin(), b.end());

    sort(sortedLocal.begin(), sortedLocal.end());
    data.swap(sortedLocal);

    if (myRank == 0) {
        double endTime = MPI_Wtime();
        cout << "Parallel sort completed in " << (endTime - startTime) << " s\n";
    }
}

// -------------------- MAIN --------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int p, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    uint64_t globalSize = 1000000;
    int maxValue = 100;

    if (myRank == 0) {
        cout << "Enter total number of elements: ";
        cin >> globalSize;
        if (globalSize % p != 0) {
            cout << "Warning: not divisible by " << p
                 << ". Truncating to " << (globalSize / p) * p << endl;
            globalSize = (globalSize / p) * p;
        }
    }

    MPI_Bcast(&globalSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxValue, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> localData = generateLocalChunk(globalSize, maxValue, myRank, p);

    parallelSortP2P(MPI_COMM_WORLD, localData, p, myRank);

    bool globalSorted = checkGlobalSortedness(MPI_COMM_WORLD, localData, myRank, p);

    if (myRank == 0) {
        cout << "Is globally sorted: " << (globalSorted ? "YES" : "NO") << endl;
    }

    MPI_Finalize();
    return 0;
}
