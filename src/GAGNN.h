#pragma once
#include "GNeuralNetwork.h"
#include <MyHeaders/XRandom.h>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
using std::sort;
using std::thread;
using std::mutex;
using std::condition_variable;
using std::deque;
using std::random_shuffle;

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

// Each thread is going to use this information
struct ThreadInfo
{
	// true if thread has to terminate
	bool Quit;
	// true if thread has finished its work
	bool Done;
	// true if thread has to wakeup
	bool WakeUp;

	// a mutex and a condition variable to synchronize everything
	mutex m;
	condition_variable cv;

	// ID of the thread, starts at 0
	unsigned int ID;
	// First offspring position
	unsigned int First;
	// Last offspring position
	unsigned int Last;
	// How many "elites" are in the Population vector
	unsigned int EliteCount;
	// How many parents for each offspring
	unsigned int ParentCount;
	// A network will be alive for 'DieGen' number of generations
	unsigned int MaxGen;
	// The mutation probability
	float MutationProb;

	// Pointers to input and output
	const vector<vector<float>>* input;
	const vector<vector<float>>* output;

	// The thread that will use all this info
	thread Thread;

	// a default constructor
	ThreadInfo()
	{
		Quit = Done = WakeUp = false;
		ID = First = Last = EliteCount = ParentCount = MaxGen = 0;
		MutationProb = 0.0f;
		input = output = 0;
	}
};

class GAGNN
{
private:
	// variables that are used only in xorshf96() function
	unsigned int x = 123456789;
	unsigned int y = 362436069;
	unsigned int z = 521288629;
	// the xorshf96() function to produce random numbers fast
	unsigned int xorshf96();

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
	// Initialization SD
	float InitSD;
	// The population vector
	vector<NetworkWithInfo> Population;
	// A vector to store random numbers, this acts like a lookup table
	vector<float> NormalDistributedFloats;
	// A container to hold all threads and information for each thread
	deque<ThreadInfo> ThreadInformation;
public:
	GAGNN() : PopCount(0), EliteCount(0), OffspringCount(0), RandomFloatsCount(0), ThreadCount(0), InitSD(0.0f), Population(vector<NetworkWithInfo>()) { }

	GAGNN(const GNeuralNetwork& reference, const unsigned int populationCount, const unsigned int eliteCount,
		const float initSD, const float mutationSD, const unsigned int randomNumberCount, const unsigned int threadCount = thread::hardware_concurrency())
	{
		Initialize(reference, populationCount, eliteCount, initSD, mutationSD, randomNumberCount, threadCount);
	}

	void Initialize(const GNeuralNetwork& reference, const unsigned int populationCount, const unsigned int eliteCount,
		const float initSD, const float mutationSD, const unsigned int randomNumberCount, const unsigned int threadCount = thread::hardware_concurrency());

	NetworkWithInfo CalcNextGeneration(const int parentCount, const float mutationProb,
		const vector<vector<float>>& input, const vector<vector<float>>& output, const unsigned int maxgen = (unsigned int)-1);

	unsigned int KillEliteAtRandom(const float p, const float newSD);

	void TerminateThreads();

	~GAGNN() { TerminateThreads(); }
};
