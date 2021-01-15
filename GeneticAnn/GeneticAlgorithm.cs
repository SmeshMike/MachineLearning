using AnnLibrary;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Transactions;
using static AnnLibrary.ANeuralNetwork;

namespace GeneticAnn
{
    public class GeneticAlgorithm
    {
        List<List<double>> inputs;
        private List<List<double>> outputs;
        struct NeuralIndividual
        {
            public ANeuralNetwork network;
            public double error;
            public NeuralIndividual(ANeuralNetwork tmpNr, double tmpError) : this()
            {
                network = tmpNr;
                error = tmpError;
            }
        }
        private List<NeuralIndividual> Nr;
        public ANeuralNetwork result;
        void CreatePopulation(int count)
        {
            List<uint> Configurate = new List<uint>
            {
                2,
                2,
                1
            };
            Nr = new List<NeuralIndividual>();
            inputs = new List<List<double>>();
            outputs = new List<List<double>>(); 
            for (int i = 0; i < count; i++)
            {
                inputs.Clear();
                outputs.Clear();
                var tmpNr = new ANeuralNetwork(Configurate, ActivationType.PositiveSygmoid, 1);
                tmpNr.LoadData("..\\..\\..\\..\\PerceptronData.txt", inputs, outputs);
                tmpNr.RandomInit();
                //var tmpError = tmpNr.BackPropTraining(inputs, outputs, 1, 0.0001, 1, false);//Обучение сети метдом оьбратного распространения ошибки
                tmpNr.IsTrained = true;
                var tm = CheckAccuracy(tmpNr);
                NeuralIndividual tmpNi = new NeuralIndividual(tmpNr, tm);
                Nr.Add(tmpNi);

            }
        }

        void CrossPopulation()
        {
            List<Tuple<NeuralIndividual, NeuralIndividual>> neuralPair = new List<Tuple<NeuralIndividual,NeuralIndividual>>();
            List<uint> Configurate = new List<uint>
            {
                2,
                2,
                1
            };
            ANeuralNetwork ann = new ANeuralNetwork(Configurate, ActivationType.PositiveSygmoid, 1);
            ann.RandomInit();
            int tmpCount = Nr.Count;
            List<int> usedIndex = new List<int>();
            for (int i = 0; i < tmpCount; i++)
            {
                int rand = new Random().Next(tmpCount);
                while (usedIndex.Contains(rand))
                {
                    rand = new Random().Next(tmpCount);
                }
                usedIndex.Add(rand);

                if (rand == tmpCount - 1)
                {
                    for (var layerIdx = 0; layerIdx < Nr[i].network.Weights.Count; layerIdx++)
                    {
                        for (var fromIdx = 0; fromIdx < Nr[i].network.Weights[layerIdx].Count; fromIdx++)
                        {
                            for (var toIdx = 0; toIdx < Nr[i].network.Weights[layerIdx][fromIdx].Count; toIdx++)
                            {
                                var tmpRand = new Random().Next(1000);
                                if(tmpRand >= 50)
                                    ann.Weights[layerIdx][fromIdx][toIdx] = Nr[rand].network.Weights[layerIdx][fromIdx][toIdx];
                                else
                                    ann.Weights[layerIdx][fromIdx][toIdx] = Nr[0].network.Weights[layerIdx][fromIdx][toIdx];
                            }
                        }
                    }
                }
                else
                {
                    for (var layerIdx = 0; layerIdx < Nr[i].network.Weights.Count; layerIdx++)
                    {
                        for (var fromIdx = 0; fromIdx < Nr[i].network.Weights[layerIdx].Count; fromIdx++)
                        {
                            for (var toIdx = 0; toIdx < Nr[i].network.Weights[layerIdx][fromIdx].Count; toIdx++)
                            {
                                var tmpRand = new Random().Next(1000);
                                if (tmpRand >= 50)
                                    ann.Weights[layerIdx][fromIdx][toIdx] = Nr[rand].network.Weights[layerIdx][fromIdx][toIdx];
                                else
                                    ann.Weights[layerIdx][fromIdx][toIdx] = Nr[rand+1].network.Weights[layerIdx][fromIdx][toIdx];
                            }
                        }
                    }
                }
                
                ann.IsTrained = true;
                var err = CheckAccuracy(ann);
                Nr.Add(new NeuralIndividual(ann, err));
            }

            for (int i = 0; i < Nr.Count - 1; i++)
            {
                for (int j = 0; j < Nr.Count - i - 1; j++)
                {
                    if (Nr[j].error > Nr[j + 1].error)
                    {
                        var temp = Nr[j];
                        Nr[j] = Nr[j + 1];
                        Nr[j + 1] = temp;
                    }
                }
            }
            Nr.RemoveRange(tmpCount, tmpCount );
        }

        double CheckAccuracy()
        {
            double error = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                for (int j = 0; j < outputs[i].Count; j++)
                {
                    error += (outputs[i][j] -Nr[0].network.Predict(inputs[i])[j])* (outputs[i][j] - Nr[0].network.Predict(inputs[i])[j]);
                }
            }

            return Math.Sqrt(error / inputs.Count);
        }


        double CheckAccuracy(ANeuralNetwork ann)
        {
            double error = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                for (int j = 0; j < outputs[i].Count; j++)
                {
                    var tmp = ann.Predict(inputs[i])[j];
                    error += (outputs[i][j] - ann.Predict(inputs[i])[j]) * (outputs[i][j] - ann.Predict(inputs[i])[j]);
                }
                
            }

            return Math.Sqrt(error/inputs.Count);
        }

        public void Train(double accuracy, int populationCount)
        {
            Stopwatch stopwatch = new Stopwatch();
            double curError = 1;
            CreatePopulation(populationCount);
            int iter = 0;
            while (curError > accuracy)
            {
                stopwatch.Start();
                CrossPopulation();
                curError = CheckAccuracy();
                if (iter%100==0)
                {
                    stopwatch.Stop();
                    var ts = stopwatch.Elapsed;
                    stopwatch.Reset();
                    Console.WriteLine("Iteration: " + iter + "\tError: " + curError + "\tTime: " + ts.TotalSeconds);
                }

                iter++;
            }

            result = Nr[0].network;
        }
    }
}
