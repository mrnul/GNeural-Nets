#include "GAGNN.h"

unsigned int GAGNN::xorshf96()
{
	x ^= x << 16;
	x ^= x >> 5;
	x ^= x << 1;

	const unsigned int t = x;
	x = y;
	y = z;
	z = t ^ x ^ y;

	return z;
}

void GAGNN::Initialize(const GNeuralNetwork& reference, const unsigned int populationCount, const unsigned int eliteCount,
	const float initSD, const float mutationSD, const unsigned int randomNumberCount, const unsigned int threadCount)
{
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
		Population[i].Network.RandomizeWeights(0.0f, initSD);
		Population[i].Network.Info.Error = INFINITY;
		Population[i].Network.Info.NumberOfWeights = Population[i].Network.NumberOfWeights();
	}

	NormalRealRandom rnd(0.0, mutationSD);
	NormalDistributedFloats.resize(RandomFloatsCount);
	for (unsigned int i = 0; i < RandomFloatsCount; i++)
	{
		NormalDistributedFloats[i] = (float)rnd();
	}

	const auto Worker = [&](int ID)
	{
		while (true)
		{
			// wait till wakeup or quit is set to true
			std::unique_lock<std::mutex> lk(ThreadInformation[ID].m);
			ThreadInformation[ID].cv.wait(lk, [&]() {return ThreadInformation[ID].WakeUp || ThreadInformation[ID].Quit; });

			if (ThreadInformation[ID].Quit)
				return;

			vector<int> parents(ThreadInformation[ID].ParentCount);
			for (unsigned int o = ThreadInformation[ID].First; o < ThreadInformation[ID].Last; o++)
			{
				// early stop
				if (ThreadInformation[ID].input->size() == 0 || ThreadInformation[ID].output->size() == 0)
					break;

				// figure out who the parents are
				for (unsigned int p = 0; p < ThreadInformation[ID].ParentCount; p++)
					parents[p] = xorshf96() % EliteCount;

				// for each layer
				const unsigned int LayerCount = (unsigned int)Population[o].Network.Layers.size();
				for (unsigned int l = 0; l < LayerCount; l++)
				{
					// for each neuron
					const unsigned int NeuronCount = (unsigned int)Population[o].Network.Layers[l].Neurons.size();
					for (unsigned int n = 0; n < NeuronCount; n++)
					{
						// calculate bias
						const int p = xorshf96() % ThreadInformation[ID].ParentCount;
						Population[o].Network.Layers[l].Neurons[n].Bias = Population[parents[p]].Network.Layers[l].Neurons[n].Bias;

						// introduce a mutation if must and store some statistics
						if ((float)xorshf96() / (float)(ULONG_MAX) < ThreadInformation[ID].MutationProb)
						{
							const float val = NormalDistributedFloats[xorshf96() % RandomFloatsCount];
							Population[o].Network.Layers[l].Neurons[n].Bias += val;
							Population[o].ExInfo.NumberOfMutations++;
							Population[o].ExInfo.TotalMutationValue += val;

							if (Population[o].ExInfo.MaxMutationValue < val)
								Population[o].ExInfo.MaxMutationValue = val;
							else if (Population[o].ExInfo.MinMutationValue > val)
								Population[o].ExInfo.MinMutationValue = val;
						}

						// calculate weights
						// for each InputInfo
						for (auto& x : Population[o].Network.Layers[l].Neurons[n].InputMap)
						{
							// use parents to produce an offspring
							// offspring inherits values at random
							const int p = xorshf96() % ThreadInformation[ID].ParentCount;
							x.second.Weight = Population[parents[p]].Network.Layers[l].Neurons[n].InputMap[x.second.InNeuron].Weight;

							// introduce a mutation if must and store some statistics
							if ((float)xorshf96() / (float)(ULONG_MAX) < ThreadInformation[ID].MutationProb)
							{
								const float val = NormalDistributedFloats[xorshf96() % RandomFloatsCount];
								x.second.Weight += val;
								Population[o].ExInfo.NumberOfMutations++;
								Population[o].ExInfo.TotalMutationValue += val;

								if (Population[o].ExInfo.MaxMutationValue < val)
									Population[o].ExInfo.MaxMutationValue = val;
								else if (Population[o].ExInfo.MinMutationValue > val)
									Population[o].ExInfo.MinMutationValue = val;
							}
						}
					}
				}
				// update error, accuracy and some statistics
				const NetInfo tmpInfo = Population[o].Network.TotalErrorAccuracy(*ThreadInformation[ID].input, *ThreadInformation[ID].output,
					Population[EliteCount - 1].Network.Info.Error);

				Population[o].Network.Info.Error = tmpInfo.Error;
				Population[o].Network.Info.Accuracy = tmpInfo.Accuracy;
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
		ThreadInformation[i].EliteCount = EliteCount;
		ThreadInformation[i].Thread = thread(Worker, i);
	}
}

NetworkWithInfo GAGNN::CalcNextGeneration(const int parentCount, const float mutationProb,
	const vector<vector<float>>& input, const vector<vector<float>>& output)
{
	// wakeup threads
	for (unsigned int i = 0; i < ThreadCount; i++)
	{
		{
			std::lock_guard<std::mutex> lk(ThreadInformation[i].m);
			ThreadInformation[i].WakeUp = true;
			ThreadInformation[i].Done = false;
			ThreadInformation[i].ParentCount = parentCount;
			ThreadInformation[i].MutationProb = mutationProb;
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

	// return the best
	return Population[0];
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