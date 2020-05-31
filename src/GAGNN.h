#pragma once

#include "GNeuralNetwork.h"
#include <MyHeaders/XRandom.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
using std::sort;
using std::thread;
using std::mutex;
using std::condition_variable;
using std::deque;

constexpr float FAST_RAND_TO_PROB = 1.0f / 32767.0f;

static unsigned int GSeed = 0;
unsigned int fast_rand();

// Extended information about the network
struct ExNetInfo
{
	float TotalMutationValue;
	float MaxMutationValue;
	float MinMutationValue;

	unsigned int Generation;
	unsigned int NumberOfMutations;

	ExNetInfo() : TotalMutationValue(0.0f), MaxMutationValue(-INFINITY), MinMutationValue(INFINITY),
		Generation(0), NumberOfMutations(0) { };
};

struct NetworkWithInfo
{
	// The network
	GNeuralNetwork Network;
	// Info about the network
	ExNetInfo ExInfo;

	NetworkWithInfo() : Network(GNeuralNetwork()), ExInfo(ExNetInfo()) { }
};

struct GAGNNParams
{
	// A network will be alive for 'DieGen' number of generations
	unsigned int MaxGen;
	// number of parents
	unsigned int ParentCount;
	// The mutation probability
	float MutationProb;
	// alpha values
	float a1;
	float a2;
	// max norm
	float MaxNorm;

	GAGNNParams()
	{
		MaxGen = (unsigned int)-1;
		MutationProb = a1 = a2 = 0.0f;
		ParentCount = 0;
		MaxNorm = INFINITY;
	}
};

// Each thread is going to use this information
struct ThreadInfo
{
	// The thread that will use all this info
	thread Thread;

	// a mutex and a condition variable to synchronize everything
	mutex m;
	condition_variable cv;

	// ID of the thread, starts at 0
	unsigned int ID;
	// First offspring position
	unsigned int First;
	// Last offspring position
	unsigned int Last;

	GAGNNParams Parameters;

	// Pointers to input and output
	const vector<vector<float>>* input;
	const vector<vector<float>>* output;

	// true if thread has to terminate
	bool Quit;
	// true if thread has finished its work
	bool Done;
	// true if thread has to wakeup
	bool WakeUp;

	// a default constructor
	ThreadInfo()
	{
		Quit = Done = WakeUp = false;
		ID = First = Last = 0;
		input = output = 0;
		Parameters = GAGNNParams();
	}
};

class GAGNN
{
private:
	// Population size
	unsigned int PopCount;
	// Number of "elites" in the population
	// Only the elites produde offsprings
	unsigned int EliteCount;
	// Total number of offsprings
	unsigned int OffspringCount;

	// The following property must hold:
	//		PopCount > EliteCount

	// Number of random floats stored in vector NormalDistributedFloats
	unsigned int RandomFloatsCount;
	// Number of worker threads
	unsigned int ThreadCount;
	// The population vector
	vector<NetworkWithInfo> Population;
	// A vector to store random numbers, this acts like a lookup table
	vector<float> NormalDistributedFloats;
	// A container to hold all threads and information for each thread
	deque<ThreadInfo> ThreadInformation;
public:
	GAGNN() : PopCount(0), EliteCount(0), OffspringCount(0), RandomFloatsCount(0), ThreadCount(0),
		Population(vector<NetworkWithInfo>()) { }

	GAGNN(const GNeuralNetwork& reference, const unsigned int populationCount, const unsigned int eliteCount,
		const float mutationSD, const unsigned int randomNumberCount, const unsigned int threadCount = thread::hardware_concurrency())
	{
		Initialize(reference, populationCount, eliteCount, mutationSD, randomNumberCount, threadCount);
	}

	void Initialize(const GNeuralNetwork& reference, const unsigned int populationCount, const unsigned int eliteCount,
		const float mutationSD, const unsigned int randomNumberCount, const unsigned int threadCount = thread::hardware_concurrency());

	void GenerateRandomFloats(const unsigned int count, const float SD);

	void CalcNextGeneration(const GAGNNParams Params, const vector<vector<float>>& input, const vector<vector<float>>& output);

	unsigned int KillEliteAtRandom(const float p);
	void MaxNorm(const float max);

	NetworkWithInfo GetBest(const unsigned int i = 0);

	void TerminateThreads();

	~GAGNN() { TerminateThreads(); }
};
