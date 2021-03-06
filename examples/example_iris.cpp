#include <iostream>
#include <fstream>
#include <sstream>
#include "GNeuralNetwork.h"
#include "GAGNN.h"

using std::cout;
using std::endl;
using std::cin;

struct Data
{
	vector<float> input;
	vector<float> target;
};

bool LoadIRIS(const char* path, vector<Data>& data)
{
	using namespace std;
	ifstream file(path);

	if (!file.is_open())
		return false;

	string line;
	string val;
	while (!file.eof())
	{
		getline(file, line);
		if (line.empty())
			return true;

		stringstream ss(line);

		data.push_back(Data());
		data.back().input.resize(4);
		data.back().target.resize(3);

		for (unsigned int i = 0; i < 4; i++)
		{
			getline(ss, val, ',');
			data.back().input[i] = stof(val);
		}

		getline(ss, val, ',');
		data.back().target[stoi(val)] = 1;
	}

	return true;
}

int main()
{
	vector<Data> data;

	// load data
	LoadIRIS("iris.data", data);

	// shuffle them
	std::random_shuffle(data.begin(), data.end());

	vector<vector<float>> input;
	vector<vector<float>> output;

	// split into training and test sets
	// training 
	for (unsigned int i = 0; i < 120; i++)
	{
		input.push_back(data[i].input);
		output.push_back(data[i].target);
	}

	vector<vector<float>> testinput;
	vector<vector<float>> testoutput;

	// test set
	for (auto i = input.size(); i < data.size(); i++)
	{
		testinput.push_back(data[i].input);
		testoutput.push_back(data[i].target);
	}

	GAGNNParams Params;
	Params.MutationCoeff = 0.3f;
	Params.MutationProb = 0.1f;
	Params.ParentCount = 2;

	// 4 input nodes, one hidden layer with 8 nodes, 3 output nodes
	// Create a population of 20 networks of which 10 are the elites.
	// Create 1000 random floats ~N(0, 1.0)
	// Run in one worker thread
	GAGNN Test(GNeuralNetwork({ 4 , 8, 3 }), 20, 10, 1000, 1);

	//A variable to store the best network
	NetworkWithInfo res;

	cout.precision(3);
	while (Test.GetBest().Network.Info.Error > 1.0f)
	{
		Test.CalcNextGeneration(Params, input, output);
		res = Test.GetBest();
		cout << res.Network.Info.Error << "\t|->\t"\
			<< res.Network.Info.Accuracy << "\t" << res.ExInfo.Generation << "\t" << res.Network.Info.NumberOfWeights << "\t"\
			<< endl;
	}

	res.Network.TotalErrorAccuracy(input, output);
	cout << "Results Training set:\n"\
		<< res.Network.Info.Error << "\t|->\t"\
		<< res.Network.Info.Accuracy << "\t" << res.ExInfo.Generation << "\t" << res.Network.Info.NumberOfWeights << "\n"\
		<< endl;

	res.Network.TotalErrorAccuracy(testinput, testoutput);
	cout << "Results Test set:\n"\
		<< res.Network.Info.Error << "\t|->\t"\
		<< res.Network.Info.Accuracy << "\t" << res.ExInfo.Generation << "\t" << res.Network.Info.NumberOfWeights << "\n"\
		<< endl;

	cout << "Max weight:" << res.Network.MaxWeightValue() << "\n"\
		<< "Min weight:" << res.Network.MinWeightValue() << "\n"\
		<< endl;

	cout << "Done" << endl;
	cin.get();
}
