#pragma once

#include <vector>
#include <functional>
#include <MyHeaders/XRandom.h>
#include <map>
#include <string>
using std::map;
using std::vector;
using std::function;
using std::string;

// Information about the network
struct NetInfo
{
	float Error;
	float Accuracy;

	unsigned int NumberOfWeights;

	NetInfo() : Error(INFINITY), Accuracy(0.0f), NumberOfWeights(0) { }
};

// Reference to a (L)ayer and (N)euron
struct Point
{
	unsigned int L;
	unsigned int N;
	Point() : L(-1), N(-1) { }
	Point(const unsigned int l, const unsigned int n) : L(l), N(n) { }

	bool operator==(const Point& p)
	{
		return (p.L == L) && (p.N == N);
	}
};

inline bool operator<(const Point& left, const Point& right)
{
	if (left.L < right.L)
	{
		return true;
	}
	else if (left.L == right.L)
	{
		if (left.N < right.N)
		{
			return true;
		}
	}

	return false;
}

// Information about the input of each neuron
struct InputInfo
{
	Point InNeuron;
	float Weight;
	InputInfo() : InNeuron(Point()), Weight(0.0f) {}
	InputInfo(const Point& p, const float weight) : InNeuron(p), Weight(weight) { }
};

float DefaultActivation(const float x);

// Information about each neuron
struct Neuron
{
	float Activation;
	float WeightedSum;
	float Bias;
	float (*ActivationFunction)(const float x);
	bool Zeroize;

	// each neuron's input
	vector<InputInfo> InputVector;

	Neuron() : Activation(0.0f), WeightedSum(0.0f), Bias(0.0f), ActivationFunction(DefaultActivation), Zeroize(true),
		InputVector(vector<InputInfo>()) { }
};

struct Layer
{
	vector<Neuron> Neurons;
	Layer() : Neurons(vector<Neuron>()) { }
};

bool DefaultClassifierCheck(const vector<Neuron>& out, const vector<float>& correct);
string GetOutputString(const vector<float>& output, const int decimal = 4);

class GNeuralNetwork
{
private:

public:
	NetInfo Info;
	vector<Layer> Layers;

	GNeuralNetwork() : Layers(vector<Layer>()) { }
	GNeuralNetwork(const vector<int>& topology, const float p = 1.0f)
	{
		BuildFFNetwork(topology, p);
	}

	GNeuralNetwork(const vector<int>& topology, const vector<float> p)
	{
		BuildFFNetwork(topology, p);
	}

	unsigned Zeroize(const bool forced);

	void RunThroughEachNeuron(function<void(vector<Layer>& layers, const Point& p)> Perform);

	void RunThroughEachNeuronInLayer(function<void(vector<Layer>& layers, const Point& p)> Perform, const unsigned int layerNum);

	void RunThroughEachWeight(function<void(vector<Layer>& layers, const Point& p, const unsigned int index, float& weight)> Perform);

	void RunThroughEachLayer(function<void(vector<Layer>& layers, const unsigned int l)> Perform);

	const vector<Neuron>& Feed(const vector<float>& input, const bool forceZero = false);

	float TotalError(const vector<vector<float>>& input, const vector<vector<float>>& output);

	NetInfo TotalErrorAccuracy(const vector<vector<float>>& input, const vector<vector<float>>& output, const float cutoff = INFINITY,
		bool(ClassifierCheck(const vector<Neuron>& in, const vector<float>& out)) = DefaultClassifierCheck);

	float Accuracy(const vector<vector<float>>& input, const vector<vector<float>>& output,
		bool(ClassifierCheck(const vector<Neuron>& in, const vector<float>& out)) = DefaultClassifierCheck);

	void RandomizeWeights(const float norm = 0.0f);

	void BuildFFNetwork(const vector<int> topology, const float p = 1.0f);
	void BuildFFNetwork(const vector<int> topology, const vector<float> p);

	float MaxWeightValue();
	float MinWeightValue();

	float W1();
	float W2();

	void MaxNorm(const float max);
	void ForceNorm(const float norm);

	unsigned int NumberOfWeights();

	// inserts Connection in the InputVector of neuron (Where.L, Where.L) ONLY if it does not already exist
	bool InsertConnection(const Point& Where, const InputInfo& Connection, const bool atEnd = true);
	// searches for Connection in the InputVector of neuron (Where.L, Where.L)
	unsigned int ConnectionExists(const Point& Where, const InputInfo& Connection);

	string GetOutputString(const int decimal = 4) const;

	bool EqualTopology(const GNeuralNetwork& other) const;
};

// result = (networks[0] + networks[1] + ... networks[N]) / (N + 1)
unsigned int Average(const vector<GNeuralNetwork>& networks, GNeuralNetwork& result);
