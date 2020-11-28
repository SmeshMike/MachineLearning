using System;
using System.Collections.Generic;
using ANNLib;

namespace ANNSample
{
    class AnnSample
    {
        static void Main(string[] args)
        {
			Console.WriteLine( "hello ANN!" );
            Ann.ANeuralNetwork tmp = new Ann.ANeuralNetwork();
            tmp.GetTestString();
            var configurate = new List<uint>();
            var nr = tmp.CreateNeuralNetwork(configurate, RootANN.ActivationType.BipolarSygmoid, 1);
            List<List<double>> inputs, outputs;
            nr.Load("..\\Debug\\xor.nn");
            Console.WriteLine(nr.GetType());

		}
    }
}
