#include "GAGNN.h"

void GAGNN::Initialize(const GNeuralNetwork& reference, const unsigned int populationCount, const unsigned int eliteCount,
	const unsigned int randomNumberCount, const unsigned int threadCount)
{
	srand((unsigned int)time(0));

	TerminateThreads();

	if (eliteCount >= populationCount)
		return;

	PopCount = populationCount;
	EliteCount = eliteCount;
	OffspringCount = populationCount - eliteCount;
	RandomFloatsCount = randomNumberCount;
	ThreadCount = threadCount;

	Population.resize(PopCount);
	for (unsigned int i = 0; i < PopCount; i++)
	{
		Population[i].Network.Layers = reference.Layers;
		Population[i].Network.RandomizeWeights();
		Population[i].Network.Info.Error = INFINITY;
		Population[i].Network.Info.NumberOfWeights = Population[i].Network.NumberOfWeights();
	}

	GenerateRandomFloats(randomNumberCount);

	const auto Worker = [&](int ID)
	{
		unsigned int GSeed = rand();

		const auto fast_rand = [&GSeed]()
		{
			GSeed = (214013 * GSeed + 2531011);
			return (GSeed >> 16) & 0x7FFF;
		};

		while (true)
		{
			// wait till wakeup or quit is set to true
			std::unique_lock<std::mutex> lk(ThreadInformation[ID].m);
			ThreadInformation[ID].cv.wait(lk, [&]() {return ThreadInformation[ID].WakeUp || ThreadInformation[ID].Quit; });

			if (ThreadInformation[ID].Quit)
				return;

			for (unsigned int o = ThreadInformation[ID].First; o < ThreadInformation[ID].Last; o++)
			{
				// early stop
				if (ThreadInformation[ID].input->size() == 0 || ThreadInformation[ID].output->size() == 0)
					break;

				// die if must
				if (Population[o].ExInfo.Generation >= ThreadInformation[ID].Parameters.MaxGen)
				{
					Population[o].Network.RandomizeWeights();
					Population[o].Network.Info.Accuracy = NetInfo().Accuracy;
					Population[o].Network.Info.Error = NetInfo().Error;
					Population[o].ExInfo = ExNetInfo();
					break;
				}

				Population[o].Network.RunThroughEachWeight([&](vector<Layer>& layers, const Point& p, const unsigned int index, float& weight)
					{
						// a random parent
						const int prnt = fast_rand() % ThreadInformation[ID].Parameters.ParentCount;
						if (index == (unsigned int)-1) // is bias
						{
							weight = Population[prnt].Network.Layers[p.L].Neurons[p.N].Bias;
						}
						else // is not bias
						{
							weight = Population[prnt].Network.Layers[p.L].Neurons[p.N].InputVector[index].Weight;
						}

						// check if it is time to introduce a mutation
						if ((float)fast_rand() * FAST_RAND_TO_PROB >= ThreadInformation[ID].Parameters.MutationProb)
							return; // return if not

						// introduce a mutation and store some statistics
						const float val = NormalDistributedFloats[fast_rand() % RandomFloatsCount] * ThreadInformation[ID].Parameters.MutationCoeff;
						weight += val;
						Population[o].ExInfo.NumberOfMutations++;
						Population[o].ExInfo.TotalMutationValue += val;

						if (Population[o].ExInfo.MaxMutationValue < val)
							Population[o].ExInfo.MaxMutationValue = val;
						else if (Population[o].ExInfo.MinMutationValue > val)
							Population[o].ExInfo.MinMutationValue = val;
					});

				// apply maxnorm
				if (ThreadInformation[ID].Parameters.MaxNorm > 0.0f)
					Population[o].Network.MaxNorm(ThreadInformation[ID].Parameters.MaxNorm);

				// update error, accuracy and some statistics
				const NetInfo tmpInfo = Population[o].Network.TotalErrorAccuracy(*ThreadInformation[ID].input, *ThreadInformation[ID].output,
					Population[EliteCount - 1].Network.Info.Error);

				// store error and accuracy
				Population[o].Network.Info.Error = tmpInfo.Error;
				Population[o].Network.Info.Accuracy = tmpInfo.Accuracy;

				// update with w1 and/or w2 if must
				if (ThreadInformation[ID].Parameters.a1 > 0.0f)
					Population[o].Network.Info.Error += ThreadInformation[ID].Parameters.a1 * Population[o].Network.W1();
				if (ThreadInformation[ID].Parameters.a2 > 0.0f)
					Population[o].Network.Info.Error += ThreadInformation[ID].Parameters.a2 * Population[o].Network.W2();

				// increase generation counter
				Population[o].ExInfo.Generation++;
			}

			// notify that work is done
			ThreadInformation[ID].Done = true;
			ThreadInformation[ID].WakeUp = false;

			lk.unlock();
			ThreadInformation[ID].cv.notify_one();
		}
	};

	// calculate basic info for each thread and start thread
	const unsigned int numPerThread = OffspringCount / ThreadCount;
	ThreadInformation.resize(ThreadCount);
	for (unsigned int i = 0; i < ThreadCount; i++)
	{
		ThreadInformation[i].ID = i;
		ThreadInformation[i].First = EliteCount + i * numPerThread;
		ThreadInformation[i].Last = EliteCount + (i + 1) * numPerThread;
		ThreadInformation[i].Thread = thread(Worker, i);
	}
}

void GAGNN::GenerateRandomFloats(const unsigned int count)
{
	NormalRealRandom rnd(0, 1.0);
	NormalDistributedFloats.resize(count);
	for (unsigned int i = 0; i < count; i++)
	{
		NormalDistributedFloats[i] = (float)rnd();
	}
	RandomFloatsCount = count;
}

void GAGNN::CalcNextGeneration(const GAGNNParams Params, const vector<vector<float>>& input, const vector<vector<float>>& output)
{
	// wakeup threads
	for (unsigned int i = 0; i < ThreadCount; i++)
	{
		{
			std::lock_guard<std::mutex> lk(ThreadInformation[i].m);
			ThreadInformation[i].WakeUp = true;
			ThreadInformation[i].Done = false;
			ThreadInformation[i].Parameters = Params;
			ThreadInformation[i].input = &input;
			ThreadInformation[i].output = &output;
		}
		ThreadInformation[i].cv.notify_one();
	}

	// wait to finish
	for (unsigned int i = 0; i < ThreadCount; i++)
	{
		std::unique_lock<std::mutex> lk(ThreadInformation[i].m);
		ThreadInformation[i].cv.wait(lk, [&]() {return ThreadInformation[i].Done; });
	}

	// sort all
	std::sort(Population.begin(), Population.end(), [](const NetworkWithInfo& a, const NetworkWithInfo& b) {return a.Network.Info.Error < b.Network.Info.Error; });
}

unsigned int GAGNN::KillEliteAtRandom(const float p)
{
	unsigned int Count = 0;
	for (unsigned int i = 0; i < EliteCount; i++)
	{
		if ((float)rand() * (1.0f / RAND_MAX) >= p)
			continue;

		Population[i].Network.RandomizeWeights();
		Population[i].Network.Info.Accuracy = NetInfo().Accuracy;
		Population[i].Network.Info.Error = NetInfo().Error;
		Population[i].ExInfo = ExNetInfo();
		Count++;
	}
	return Count;
}

void GAGNN::MaxNorm(const float max)
{
	for (unsigned int i = 0; i < PopCount; i++)
	{
		Population[i].Network.MaxNorm(max);
	}
}

NetworkWithInfo& GAGNN::GetBest(const unsigned int i)
{
	return Population[i];
}

void GAGNN::TerminateThreads()
{
	// wakeup threads
	for (unsigned int i = 0; i < ThreadCount; i++)
	{
		{
			std::lock_guard<std::mutex> lk(ThreadInformation[i].m);
			ThreadInformation[i].WakeUp = true;
			ThreadInformation[i].Quit = true;
		}
		// notify about termination and wait
		ThreadInformation[i].cv.notify_one();
		ThreadInformation[i].Thread.join();
	}

	// reset all
	ThreadInformation.clear();
	Population.clear();
	PopCount = EliteCount = OffspringCount = 0;

	NormalDistributedFloats.clear();
	RandomFloatsCount = 0;

	ThreadInformation.clear();
	ThreadCount = 0;
}
