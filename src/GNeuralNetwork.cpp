#include "GNeuralNetwork.h"
#include <immintrin.h>

float DefaultActivation(const float x)
{
	return x >= 0.0f ? x : std::expm1f(x);
}

void DoFeedWork(vector<Layer>& layers, const Point& p)
{
	if (p.L == 0)
		return;

	if (layers[p.L].Neurons[p.N].Zeroize)
		layers[p.L].Neurons[p.N].WeightedSum = 0.0f;

	const unsigned int InSize = (unsigned int)layers[p.L].Neurons[p.N].InputVector.size();
	const unsigned int remReps = InSize % 16;
	const unsigned int mainReps = InSize - remReps;

	// for every 16 inputs use intrinsics
	for (size_t i = 0; i < mainReps; i += 16)
	{
		// a constant reference only to make life easier
		const vector<InputInfo>& Input = layers[p.L].Neurons[p.N].InputVector;

		// vectorize the weights
		const __m512 Weights = _mm512_set_ps
		(
			Input[i + 0].Weight,
			Input[i + 1].Weight,
			Input[i + 2].Weight,
			Input[i + 3].Weight,
			Input[i + 4].Weight,
			Input[i + 5].Weight,
			Input[i + 6].Weight,
			Input[i + 7].Weight,
			Input[i + 8].Weight,
			Input[i + 9].Weight,
			Input[i + 10].Weight,
			Input[i + 11].Weight,
			Input[i + 12].Weight,
			Input[i + 13].Weight,
			Input[i + 14].Weight,
			Input[i + 15].Weight
		);

		// vectorize the activations
		const __m512 Activations = _mm512_set_ps
		(
			layers[Input[i + 0].InNeuron.L].Neurons[Input[i + 0].InNeuron.N].Activation,
			layers[Input[i + 1].InNeuron.L].Neurons[Input[i + 1].InNeuron.N].Activation,
			layers[Input[i + 2].InNeuron.L].Neurons[Input[i + 2].InNeuron.N].Activation,
			layers[Input[i + 3].InNeuron.L].Neurons[Input[i + 3].InNeuron.N].Activation,
			layers[Input[i + 4].InNeuron.L].Neurons[Input[i + 4].InNeuron.N].Activation,
			layers[Input[i + 5].InNeuron.L].Neurons[Input[i + 5].InNeuron.N].Activation,
			layers[Input[i + 6].InNeuron.L].Neurons[Input[i + 6].InNeuron.N].Activation,
			layers[Input[i + 7].InNeuron.L].Neurons[Input[i + 7].InNeuron.N].Activation,
			layers[Input[i + 8].InNeuron.L].Neurons[Input[i + 8].InNeuron.N].Activation,
			layers[Input[i + 9].InNeuron.L].Neurons[Input[i + 9].InNeuron.N].Activation,
			layers[Input[i + 10].InNeuron.L].Neurons[Input[i + 10].InNeuron.N].Activation,
			layers[Input[i + 11].InNeuron.L].Neurons[Input[i + 11].InNeuron.N].Activation,
			layers[Input[i + 12].InNeuron.L].Neurons[Input[i + 12].InNeuron.N].Activation,
			layers[Input[i + 13].InNeuron.L].Neurons[Input[i + 13].InNeuron.N].Activation,
			layers[Input[i + 14].InNeuron.L].Neurons[Input[i + 14].InNeuron.N].Activation,
			layers[Input[i + 15].InNeuron.L].Neurons[Input[i + 15].InNeuron.N].Activation
		);

		// multiply and add to the weighted sum
		layers[p.L].Neurons[p.N].WeightedSum += _mm512_reduce_add_ps(_mm512_mul_ps(Activations, Weights));
	}

	// for the rest use a simple for loop
	for (size_t i = mainReps; i < InSize; i++)
	{
		const vector<InputInfo>& Input = layers[p.L].Neurons[p.N].InputVector;

		layers[p.L].Neurons[p.N].WeightedSum += layers[Input[i].InNeuron.L].Neurons[Input[i].InNeuron.N].Activation * Input[i].Weight;
	}

	//activate
	layers[p.L].Neurons[p.N].Activation = layers[p.L].Neurons[p.N].ActivationFunction(layers[p.L].Neurons[p.N].WeightedSum + layers[p.L].Neurons[p.N].Bias);
}

bool DefaultClassifierCheck(const vector<Neuron>& out, const vector<float>& correct)
{
	const unsigned int size = (unsigned int)out.size();
	unsigned int maxPosOut = (unsigned int)-1;
	unsigned int maxPosCor = (unsigned int)-2;
	float maxOut = -INFINITY;
	float maxCor = -INFINITY;
	for (unsigned int i = 0; i < size; i++)
	{
		if (out[i].Activation > maxOut)
		{
			maxOut = out[i].Activation;
			maxPosOut = i;
		}

		if (correct[i] > maxCor)
		{
			maxCor = correct[i];
			maxPosCor = i;
		}
	}
	return maxPosOut == maxPosCor;
}

string GetOutputString(const vector<float>& output, const int decimal)
{
	const unsigned int size = (unsigned int)output.size();
	string tmp;
	string res = "{ ";
	for (unsigned int i = 0; i < size - 1; i++)
	{
		tmp = std::to_string(output[i]);
		tmp = tmp.substr(0, tmp.find('.') + decimal + 1);
		res += tmp;
		res += " , ";
	}

	tmp = std::to_string(output.back());
	tmp = tmp.substr(0, tmp.find('.') + decimal + 1);
	res += tmp;
	res += " }";
	return res;
}

unsigned GNeuralNetwork::Zeroize(const bool forced)
{
	unsigned int ZeroizeCount = 0;

	RunThroughEachNeuron([&ZeroizeCount, &forced](vector<Layer>& layer, const Point& p)
		{
			if (layer[p.L].Neurons[p.N].Zeroize || forced)
			{
				layer[p.L].Neurons[p.N].WeightedSum = 0.0f;
				layer[p.L].Neurons[p.N].Activation = 0.0f;
				ZeroizeCount++;
			}
		});

	return ZeroizeCount;
}

void GNeuralNetwork::RunThroughEachNeuron(function<void(vector<Layer>& layers, const Point& p)> Perform)
{
	//for each layer
	const unsigned int LayerCount = (unsigned int)Layers.size();
	for (unsigned int l = 0; l < LayerCount; l++)
	{
		//for each neuron
		const unsigned int NeuronCount = (unsigned int)Layers[l].Neurons.size();
		for (unsigned int n = 0; n < NeuronCount; n++)
		{
			//call perform for each neuron
			Perform(Layers, Point(l, n));
		}
	}
}

void GNeuralNetwork::RunThroughEachNeuronInLayer(function<void(vector<Layer>& layers, const Point& p)> Perform, const unsigned int layerNum)
{
	const unsigned int NeuronCount = (unsigned int)Layers[layerNum].Neurons.size();
	for (unsigned int n = 0; n < NeuronCount; n++)
	{
		Perform(Layers, Point(layerNum, n));
	}
}

void GNeuralNetwork::RunThroughEachWeight(function<void(vector<Layer>& layers, const Point& p, const unsigned int index, float& weight)> Perform)
{
	//for each layer
	const unsigned int LayerCount = (unsigned int)Layers.size();
	for (unsigned int l = 1; l < LayerCount; l++)
	{
		//for each neuron
		const unsigned int NeuronCount = (unsigned int)Layers[l].Neurons.size();
		for (unsigned int n = 0; n < NeuronCount; n++)
		{
			//for each weight
			const unsigned int InputCount = (unsigned int)Layers[l].Neurons[n].InputVector.size();
			for (unsigned int i = 0; i < InputCount; i++)
			{
				Perform(Layers, Point(l, n), i, Layers[l].Neurons[n].InputVector[i].Weight);
			}
			//and bias
			Perform(Layers, Point(l, n), (unsigned int)-1, Layers[l].Neurons[n].Bias);
		}
	}
}

void GNeuralNetwork::RunThroughEachLayer(function<void(vector<Layer>& layers, const unsigned int l)> Perform)
{
	//for each layer
	const unsigned int LayerCount = (unsigned int)Layers.size();
	for (unsigned int l = 0; l < LayerCount; l++)
	{
		Perform(Layers, l);
	}
}

const vector<Neuron>& GNeuralNetwork::Feed(const vector<float>& input, const bool forceZero)
{
	if (forceZero)
		Zeroize(true);

	//fill input layer
	const unsigned int InputLayerNeuronCount = (unsigned int)Layers[0].Neurons.size();
	for (unsigned int n = 0; n < InputLayerNeuronCount; n++)
	{
		Layers[0].Neurons[n].Activation = input[n];
		Layers[0].Neurons[n].WeightedSum = input[n];
	}

	RunThroughEachNeuron(DoFeedWork);

	//softmax
	const unsigned int OutLayerNeuronCount = (unsigned int)Layers.back().Neurons.size();
	float max = -INFINITY;
	for (unsigned int n = 0; n < OutLayerNeuronCount; n++)
	{

		if (max < Layers.back().Neurons[n].WeightedSum)
			max = Layers.back().Neurons[n].WeightedSum;
	}

	float denominator = 0.0f;
	for (unsigned int n = 0; n < OutLayerNeuronCount; n++)
	{
		denominator += std::exp(Layers.back().Neurons[n].WeightedSum - max);
	}

	for (unsigned int n = 0; n < OutLayerNeuronCount; n++)
	{
		Layers.back().Neurons[n].Activation = std::exp(Layers.back().Neurons[n].WeightedSum - max) / denominator;
	}
	return Layers.back().Neurons;
}

float GNeuralNetwork::TotalError(const vector<vector<float>>& input, const vector<vector<float>>& output)
{
	const unsigned int InputCount = (unsigned int)input.size();
	const unsigned int OutputSize = (unsigned int)output[0].size();
	Info.Error = 0.0f;
	for (unsigned int i = 0; i < InputCount; i++)
	{
		const auto& res = Feed(input[i]);
		for (unsigned int c = 0; c < OutputSize; c++)
		{
			//Info.Error += abs(output[i][c] - res[c].Activation);
			if (output[i][c] != 0.0f)
			{
				Info.Error += -output[i][c] * std::log(res[c].Activation);
				break;
			}
		}
	}
	return Info.Error;
}

NetInfo GNeuralNetwork::TotalErrorAccuracy(const vector<vector<float>>& input, const vector<vector<float>>& output, const float cutoff,
	bool(ClassifierCheck(const vector<Neuron>& in, const vector<float>& out)))
{
	const unsigned int InputCount = (unsigned int)input.size();
	const unsigned int OutputSize = (unsigned int)output[0].size();
	unsigned int CorrectCount = 0;
	Info.Error = 0.0f;
	for (unsigned int i = 0; i < InputCount; i++)
	{
		const auto& res = Feed(input[i]);
		for (unsigned int c = 0; c < OutputSize; c++)
		{
			//Info.Error += abs(output[i][c] - res[c].Activation);
			if (output[i][c] != 0.0f)
			{
				Info.Error += -output[i][c] * std::log(res[c].Activation);
				break;
			}
		}
		if (ClassifierCheck(res, output[i]))
			CorrectCount++;

		if (Info.Error >= cutoff)
			break;
	}
	Info.Accuracy = (float)CorrectCount * 100.0f / (float)InputCount;
	return Info;
}

float GNeuralNetwork::Accuracy(const vector<vector<float>>& input, const vector<vector<float>>& output,
	bool(ClassifierCheck(const vector<Neuron>& in, const vector<float>& out)))
{
	const unsigned int InputCount = (unsigned int)input.size();
	const unsigned int OutputSize = (unsigned int)output[0].size();
	unsigned int CorrectCount = 0;
	for (unsigned int i = 0; i < InputCount; i++)
	{
		const auto& res = Feed(input[i]);
		if (ClassifierCheck(res, output[i]))
			CorrectCount++;
	}
	Info.Accuracy = (float)(CorrectCount) * 100.0f / (float)(InputCount);
	return Info.Accuracy;
}

void GNeuralNetwork::RandomizeWeights(const float norm)
{
	UniformRealRandom rnd(-1.0, 1.0);

	RunThroughEachWeight([&](vector<Layer>& layers, const Point& p, const unsigned int index, float& weight)
		{
			const float c = sqrt(12.0f / (Layers[p.L - 1].Neurons.size() + 1.0f + Layers[p.L].Neurons[p.N].InputVector.size()));
			weight = (float)rnd() * c;
		}
	);

	if (norm > 0.0f)
		ForceNorm(norm);
}

void GNeuralNetwork::BuildFFNetwork(const vector<int> topology, const float p)
{
	const vector<float> probs(topology.size(), p);
	BuildFFNetwork(topology, probs);
}

void GNeuralNetwork::BuildFFNetwork(const vector<int> topology, const vector<float> p)
{
	UniformRealRandom rnd(0.0, 1.0);

	const int layerCount = (int)topology.size();
	Layers.resize(layerCount);
	Layers[0].Neurons.resize(topology[0]);
	for (int l = 1; l < layerCount; l++)
	{
		Layers[l].Neurons.resize(topology[l]);
		for (int n = 0; n < topology[l]; n++)
		{
			Layers[l].Neurons[n].Zeroize = true;
			for (int iv = 0; iv < topology[size_t(l) - 1]; iv++) // size_t to supress a warning
			{
				if (rnd() < p[size_t(l) - 1])
				{
					InsertConnection(Point(l, n), InputInfo(Point(l - 1, iv), 0.0f));
				}
			}
		}
	}

	Info.NumberOfWeights = NumberOfWeights();
}

float GNeuralNetwork::MaxWeightValue()
{
	float max = -INFINITY;
	RunThroughEachWeight([&max](vector<Layer>& layers, const Point& p, const unsigned int index, float& weight)
		{
			if (max < weight)
				max = weight;
		});

	return max;
}

float GNeuralNetwork::MinWeightValue()
{
	float min = INFINITY;
	RunThroughEachWeight([&min](vector<Layer>& layers, const Point& p, const unsigned int index, float& weight)
		{
			if (min > weight)
				min = weight;
		});

	return min;
}

void GNeuralNetwork::MaxNorm(const float max)
{

	RunThroughEachNeuron([&](vector<Layer>& layers, const Point& p)
		{
			if (p.L == 0)
				return;

			float Norm = 0.0f;
			const unsigned int InCount = (unsigned int)layers[p.L].Neurons[p.N].InputVector.size();
			for (unsigned int i = 0; i < InCount; i++)
			{
				Norm += layers[p.L].Neurons[p.N].InputVector[i].Weight * layers[p.L].Neurons[p.N].InputVector[i].Weight;
			}
			Norm = std::sqrt(Norm);

			if (Norm <= max)
				return;

			for (unsigned int i = 0; i < InCount; i++)
			{
				layers[p.L].Neurons[p.N].InputVector[i].Weight = layers[p.L].Neurons[p.N].InputVector[i].Weight * max / Norm;
			}
		});
}

void GNeuralNetwork::ForceNorm(const float norm)
{
	RunThroughEachNeuron([&](vector<Layer>& layers, const Point& p)
		{
			if (p.L == 0)
				return;

			float Norm = 0.0f;
			const unsigned int InCount = (unsigned int)layers[p.L].Neurons[p.N].InputVector.size();
			for (unsigned int i = 0; i < InCount; i++)
			{
				Norm += layers[p.L].Neurons[p.N].InputVector[i].Weight * layers[p.L].Neurons[p.N].InputVector[i].Weight;
			}

			Norm = std::sqrt(Norm);

			for (unsigned int i = 0; i < InCount; i++)
			{
				layers[p.L].Neurons[p.N].InputVector[i].Weight = layers[p.L].Neurons[p.N].InputVector[i].Weight * norm / Norm;
			}
		});
}

float GNeuralNetwork::W1()
{
	float w1 = 0.0f;
	RunThroughEachWeight([&w1](vector<Layer>& layers, const Point& p, const unsigned int index, float& weight)
		{
			w1 += std::abs(weight);
		});

	return w1;
}

float GNeuralNetwork::W2()
{
	float w2 = 0.0f;
	RunThroughEachWeight([&w2](vector<Layer>& layers, const Point& p, const unsigned int index, float& weight)
		{
			w2 += weight * weight;
		});

	return w2;
}

unsigned int GNeuralNetwork::NumberOfWeights()
{
	unsigned int count = 0;
	RunThroughEachNeuron([&count](vector<Layer>& layers, const Point& p)
		{
			if (p.L == 0)
				return;

			count += (unsigned int)layers[p.L].Neurons[p.N].InputVector.size() + 1;
		});
	Info.NumberOfWeights = count;
	return count;
}

bool GNeuralNetwork::InsertConnection(const Point& Where, const InputInfo& Connection, const bool atEnd)
{
	const unsigned int PrevSize = (unsigned int)Layers[Where.L].Neurons[Where.N].InputVector.size();
	if (atEnd)
	{
		Layers[Where.L].Neurons[Where.N].InputVector.push_back(Connection);
	}
	else
	{
		Layers[Where.L].Neurons[Where.N].InputVector.insert(Layers[Where.L].Neurons[Where.N].InputVector.begin(), Connection);
	}

	const unsigned int CurrSize = (unsigned int)Layers[Where.L].Neurons[Where.N].InputVector.size();

	return PrevSize != CurrSize;
}

string GNeuralNetwork::GetOutputString(const int decimal)
{
	const unsigned int size = (unsigned int)Layers.back().Neurons.size();
	vector<float> tmp(size);

	for (unsigned int i = 0; i < size; i++)
	{
		tmp[i] = Layers.back().Neurons[i].Activation;
	}

	return ::GetOutputString(tmp, decimal);
}
