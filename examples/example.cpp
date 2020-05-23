#include <iostream>
#include "GNeuralNetwork.h"
#include "GAGNN.h"

using std::cout;
using std::endl;

int main()
{
	const vector<vector<float>> input =  {{1,0}, {1,1}, {0,1}, {0,0}};
	const vector<vector<float>> output = {{1,0}, {0,1}, {1,0}, {0,1}};

	const vector<int> topology = { (int)input[0].size() , 2 , (int)output[0].size() };

	GNeuralNetwork Net;
	Net.BuildFFNetwork(topology, 1.0f);

	NetworkWithInfo res;
	GAGNN Test(Net, 10, 2, 1.0f, 1.0f, 1000, 1);

	while (res.Network.Info.Error > 0.01f)
	{
		res = Test.CalcNextGeneration(2, 0.1f, input, output);
		cout << res.Network.Info.Error << " |-> ";
		cout << res.Network.Info.Accuracy << " " << res.ExInfo.Generation << " " << res.Network.Info.NumberOfWeights << " ";
		cout << endl;
	}
	cout << endl;
	for (int i = 0; i < input.size(); i++)
	{
		res.Network.Feed(input[i]);
		cout << GetOutputString(input[i]) << " --> " << res.Network.GetOutputString() << endl;
	}
}
