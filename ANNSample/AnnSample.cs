using System;
using System.Collections.Generic;
using System.Linq;
using AnnLibrary;

namespace ANNSample
{
    class AnnSample
    {
        static void Main(string[] args)
        {
            ANeuralNetwork tmp = new ANeuralNetwork();
            tmp.Load("..\\..\\..\\..\\PerceptronSavedData.txt");
            var line = Console.ReadLine();
            while (line!="q")
            {

                var tmpInput = line.Split().Select(double.Parse).ToList();
                var tmpOutput = tmp.Predict(tmpInput);
                foreach (var element in tmpOutput)
                {
                    Console.WriteLine(element);
                }

                line = Console.ReadLine();
            }
        }
    }
}
