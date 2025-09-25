#include <mpi.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>

using namespace std;

// --- Map C++ types to MPI types ---
template<typename T>
MPI_Datatype MPITypeFrom();
template<> MPI_Datatype MPITypeFrom<int>() { return MPI_INT; }

// --- Check if the global data across all ranks is sorted ---
// Sends the last element of each rank to the next rank and
// compares to ensure non-decreasing order across all ranks
template <typename T>
bool checkGlobalSortedness(MPI_Comm comm, const vector<T>& localData, int myRank, int p) {
    if (localData.empty()) return true;

    T localFirst = localData.front();
    T localLast  = localData.back();

    MPI_Status status;
    T prevLast;

    bool isSorted = true;
    if (myRank < p - 1) {
        // Send last element to next rank
        MPI_Send(&localLast, 1, MPITypeFrom<T>(), myRank + 1, 0, comm);
    }
    if (myRank > 0) {
        // Receive last element from previous rank
        MPI_Recv(&prevLast, 1, MPITypeFrom<T>(), myRank - 1, 0, comm, &status);
        if (prevLast > localFirst) isSorted = false; // check order
    }

    int localSorted = isSorted ? 1 : 0;
    int globalSorted = 0;
    // Reduce across all ranks to check if any rank is unsorted
    MPI_Allreduce(&localSorted, &globalSorted, 1, MPI_INT, MPI_MIN, comm);

    return globalSorted == 1;
}

// --- Parallel sample sort using MPI_Allgather and MPI_Alltoallv ---
template<class Element>
void parallelSort(MPI_Comm comm, vector<Element>& data,
                  MPI_Datatype mpiType, int p, int myRank)
{
    double sortStart = MPI_Wtime();  // start total timing
    double commTime = 0.0;           // track communication time

    // --- Step 1: Choose random local samples for splitter selection ---
    random_device rd;
    mt19937 rndEngine(rd());
    uniform_int_distribution<size_t> dataGen(0, data.size() - 1);

    vector<Element> locS;           // local samples
    const int a = (int)(16 * log(p) / log(2.0)); // heuristic for number of samples
    for (size_t i = 0; i < (size_t)(a + 1); ++i)
        locS.push_back(data[dataGen(rndEngine)]);

    // --- Step 2: Gather all local samples from all ranks ---
    vector<Element> s(locS.size() * p);

    double c1 = MPI_Wtime();
    MPI_Allgather(locS.data(), locS.size(), mpiType,
                  s.data(), locS.size(), mpiType, comm);
    double c2 = MPI_Wtime();
    commTime += c2 - c1;

    // --- Step 3: Sort all gathered samples ---
    sort(s.begin(), s.end());

    // --- Step 4: Choose splitters ---
    for (size_t i = 0; i < p - 1; ++i)
        s[i] = s[(a + 1) * (i + 1)];  // select appropriate splitters
    s.resize(p - 1);

    // --- Step 5: Partition local data into buckets according to splitters ---
    vector<vector<Element>> buckets(p);
    for (auto& bucket : buckets)
        bucket.reserve((data.size() / p) * 2); // preallocate some space

    for (auto& el : data) {
        const auto bound = upper_bound(s.begin(), s.end(), el);
        buckets[bound - s.begin()].push_back(el); // assign to correct bucket
    }

    // --- Step 6: Flatten buckets for MPI_Alltoallv ---
    data.clear();

    vector<int> sCounts, sDispls, rCounts(p), rDispls(p + 1);
    sDispls.push_back(0);
    for (auto& bucket : buckets) {
        data.insert(data.end(), bucket.begin(), bucket.end()); // flatten
        sCounts.push_back((int)bucket.size());                // send counts
        sDispls.push_back((int)bucket.size() + sDispls.back()); // send displacements
    }

    // --- Step 7: Exchange bucket sizes with all ranks ---
    c1 = MPI_Wtime();
    MPI_Alltoall(sCounts.data(), 1, MPI_INT,
                 rCounts.data(), 1, MPI_INT, comm);
    c2 = MPI_Wtime();
    commTime += c2 - c1;

    // --- Step 8: Compute receive displacements and allocate receive buffer ---
    rDispls[0] = 0;
    for (int i = 1; i <= p; i++)
        rDispls[i] = rCounts[i - 1] + rDispls[i - 1];

    vector<Element> rData(rDispls.back());

    // --- Step 9: Exchange actual bucket data ---
    c1 = MPI_Wtime();
    MPI_Alltoallv(data.data(), sCounts.data(), sDispls.data(), mpiType,
                  rData.data(), rCounts.data(), rDispls.data(), mpiType, comm);
    c2 = MPI_Wtime();
    commTime += c2 - c1;

    // --- Step 10: Final local sort of received data ---
    sort(rData.begin(), rData.end());
    rData.swap(data);

    double sortEnd = MPI_Wtime();
    if (myRank == 0) {
        cout << "Sorting phase (excl. comm): " << (sortEnd - sortStart - commTime) << " s" << endl;
        cout << "Communication time: " << commTime << " s" << endl;
    }
}

// --- Generate deterministic local data chunk for each rank ---
vector<int> generateLocalChunk(uint64_t globalSize, int maxValue, int myRank, int p) {
    uint64_t chunkSize = globalSize / p;
    vector<int> arr(chunkSize);

    mt19937_64 gen(myRank + 123456789ULL);      // seed per rank
    uniform_int_distribution<int> dist(0, maxValue);

    for (uint64_t i = 0; i < chunkSize; i++) {
        arr[i] = dist(gen);
    }
    return arr;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int p, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    uint64_t globalSize = 1000000; // default 1e6 elements
    int maxValue = 100;

    // --- Input total elements and maximum value ---
    if (myRank == 0) {
        cout << "Enter total number of elements: ";
        cin >> globalSize;
        cout << "Enter maximum value: ";
        cin >> maxValue;

        if (globalSize % p != 0) {
            cout << "Warning: globalSize not divisible by " << p
                 << ". Truncating to " << (globalSize / p) * p << endl;
            globalSize = (globalSize / p) * p;
        }
    }

    // --- Broadcast parameters to all ranks ---
    MPI_Bcast(&globalSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxValue, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double totalStart = MPI_Wtime();

    // --- Generate only local portion of data ---
    vector<int> localData = generateLocalChunk(globalSize, maxValue, myRank, p);

    // --- Perform parallel sample sort ---
    parallelSort(MPI_COMM_WORLD, localData, MPI_INT, p, myRank);

    // --- Verify global sortedness ---
    bool globalSorted = checkGlobalSortedness(MPI_COMM_WORLD, localData, myRank, p);

    double totalEnd = MPI_Wtime();

    if (myRank == 0) {
        cout << "Total execution time: " << (totalEnd - totalStart) << " s" << endl;
        cout << "Is globally sorted: " << (globalSorted ? "YES" : "NO") << endl;
    }

    MPI_Finalize();
    return 0;
}
