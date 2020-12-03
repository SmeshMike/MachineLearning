﻿using System;
using System.Collections.Generic;
using System.Linq;
using ANNLib;

namespace ANNSample
{
    class AnnSample
    {
        static void Main(string[] args)
        {
            Ann.ANeuralNetwork tmp = new Ann.ANeuralNetwork();
            tmp.Load("..\\..\\..\\..\\savedData.txt");
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
