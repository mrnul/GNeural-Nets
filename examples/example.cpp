#include <iostream>
#include "GNeuralNetwork.h"
#include "GAGNN.h"

using std::cout;
using std::endl;

int main()
{
	const vector<vector<float>> input = { {1,0}, {1,1}, {0,1}, {0,0} };
	const vector<vector<float>> output = { {1,0}, {0,1}, {1,0}, {0,1} };

	const vector<int> topology = { (int)input[0].size() , 2 , (int)output[0].size() };


	GAGNNParams Params;
	Params.MaxGen = 30;
	Params.MutationProb = 0.2f;
	Params.ParentCount = 2;

	GAGNN Test(GNeuralNetwork(topology), 10, 4, 1.0f, 100, 1);
	NetworkWithInfo res;

	cout.precision(3);
	while (res.Network.Info.Error > 0.01f)
	{
		Test.CalcNextGeneration(Params, input, output);
		res = Test.GetBest();
		cout << res.Network.Info.Error << "\t|->\t";
		cout << res.Network.Info.Accuracy << "\t" << res.ExInfo.Generation << "\t" << res.Network.Info.NumberOfWeights << "\t";
		cout << Test.KillEliteAtRandom(0.01f);
		cout << endl;

	}
	cout << endl;
	for (int i = 0; i < input.size(); i++)
	{
		res.Network.Feed(input[i]);
		cout << GetOutputString(input[i]) << " --> " << res.Network.GetOutputString() << endl;
	}
}
