#include <iostream>
#include "GNeuralNetwork.h"
#include "GAGNN.h"

using std::cout;
using std::endl;

int main()
{
	//XOR inputs
	const vector<vector<float>> input = { {1,0}, {1,1}, {0,1}, {0,0} };

	//{1,0| = True
	//{0,1} = False
	const vector<vector<float>> output = { {1,0}, {0,1}, {1,0}, {0,1} };

	// 2 input nodes, one hidden layer with 2 nodes, 2 output nodes
	const vector<int> topology = { 2 , 2 , 2 };

	GAGNNParams Params;
	Params.MutationCoeff = 0.1f;
	Params.MutationProb = 0.1f;
	Params.ParentCount = 2;

	//Creaete a population of 50 networks of which 10 are the elites.
	//Create 1000 random floats ~N(0, 1.0)
	//Run in one worker thread
	GAGNN Test(GNeuralNetwork(topology), 50, 10, 1000, 1);

	//A variable to store the best network
	NetworkWithInfo res;

	cout.precision(3);
	while (res.Network.Info.Error > 0.01f)
	{
		Test.CalcNextGeneration(Params, input, output);
		res = Test.GetBest();
		cout << res.Network.Info.Error << "\t|->\t"\
			<< res.Network.Info.Accuracy << "\t" << res.ExInfo.Generation << "\t" << res.Network.Info.NumberOfWeights << "\t"\
			<< endl;

	}

	cout << "Max weight:" << res.Network.MaxWeightValue() << "\n"\
	<< "Min weight:" << res.Network.MinWeightValue() << "\n"\
	<< endl;

	//For each input print the calculated output
	for (int i = 0; i < input.size(); i++)
	{
		res.Network.Feed(input[i]);
		cout << GetOutputString(input[i]) << " --> " << res.Network.GetOutputString() << endl;
	}
}
