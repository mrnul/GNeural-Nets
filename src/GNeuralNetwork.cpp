#include "GNeuralNetwork.h"

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

	//for each input
	for (const auto& x : layers[p.L].Neurons[p.N].InputVector)
	{
		layers[p.L].Neurons[p.N].WeightedSum += layers[x.InNeuron.L].Neurons[x.InNeuron.N].Activation * x.Weight;
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
	for (unsigned int l = 0; l < LayerCount; l++)
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

void GNeuralNetwork::RandomizeWeights()
{
	UniformRealRandom rnd(-1.0, 1.0);

	RunThroughEachNeuron([&](vector<Layer>& layer, const Point& p)
		{
			if (p.L == 0)
				return;

			const float c = sqrt(12.0f / (Layers[p.L - 1].Neurons.size() + 1.0f + Layers[p.L].Neurons.size()));
			//random bias
			layer[p.L].Neurons[p.N].Bias = (float)rnd() * c;

			//for each input
			for (auto& x : layer[p.L].Neurons[p.N].InputVector)
			{
				x.Weight = (float)rnd() * c;
			}
		});
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

bool GNeuralNetwork::InsertConnection(const Point& Where, const InputInfo& Connection)
{
	const unsigned int PrevSize = (unsigned int)Layers[Where.L].Neurons[Where.N].InputVector.size();
	Layers[Where.L].Neurons[Where.N].InputVector.push_back(Connection);

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
