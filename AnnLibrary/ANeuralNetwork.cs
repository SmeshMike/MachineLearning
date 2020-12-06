using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AnnLibrary
{
    public class Ann
    {
        public class ANeuralNetwork : AnnRoot
        {
            public override bool Load(string filepath)
            {
                var file = new StreamReader(filepath);
                var line = file.ReadLine();
                if (line != null && !line.Contains("activation type:"))
                    throw new Exception("incorrect file format");
                line = file.ReadLine();
                if (line != null)
                {
                    Enum.TryParse(line.Trim(), out AnnRoot.ActivationType tmp);
                    FunctionType = tmp;
                }

                IsTrained = true;

                line = file.ReadLine();
                if (line != null && !line.Contains("activation scale:"))
                    throw new Exception("incorrect file format");
                line = file.ReadLine();
                if (line != null) Scale = Convert.ToDouble(line.Trim());

                line = file.ReadLine();
                if (line != null && !line.Contains("configuration:"))
                    throw new Exception("incorrect file format");
                line = file.ReadLine();
                if (line != null)
                {
                    line = line.Trim();
                    var tmp = line.Split('\t', ' ');
                    Configuration = tmp.Select(uint.Parse).ToList();
                }

                line = file.ReadLine();
                if (line != null && !line.Contains("weights:"))
                    throw new Exception("incorrect file format");
                
                Weights = new List<List<List<double>>>(Configuration.Count - 1);
                for (var layerIdx = 0; layerIdx < Configuration.Count - 1; layerIdx++)
                {
                    Weights.Add( new List<List<double>>());
                    for (var fromIdx = 0; fromIdx < Configuration[layerIdx] ; fromIdx++)
                    {
                        line = file.ReadLine();
                        Weights[layerIdx].Add(new List<double>());
                        var tmp = line.TrimEnd().Split('\t', ' ');
                        Weights[layerIdx][fromIdx].AddRange( tmp.Select(double.Parse).ToList());
                    }
                }

                return true;
            }

            public override bool Save(string filepath)
            {
                if (!IsTrained)
                    return false;
                StreamWriter file = new StreamWriter(filepath);
                file.WriteLine("activation type:");
                file.WriteLine(FunctionType);
                file.WriteLine("activation scale:");
                file.WriteLine(Scale);
                file.WriteLine("configuration:");
                foreach (var u in Configuration)
                {
                    var neuronCount = (int)u;
                    file.Write(neuronCount + "\t");
                }
                file.WriteLine();
                file.WriteLine("weights:");
                foreach (var weightLine in Weights.SelectMany(weightMatrix => weightMatrix))
                {
                    foreach (var weight in weightLine)
                    {
                        file.Write(weight + " ");
                    }

                    file.WriteLine();
                }

                file.Close();
                return true;

            }

            public override void RandomInit()
            {
                var rand = new Random();

                Weights = new List<List<List<double>>>(Configuration.Count - 1);

                for (var i = 0; i < Configuration.Count - 1; i++)
                {
                    Weights.Add(new List<List<double>>((int)Configuration[i]));

                    for (var j = 0; j < Configuration[i] && i < Configuration.Count - 1; j++)
                    {
                        Weights[i].Add(new List<double>((int)Configuration[i + 1]));
                    }

                    for (var j = 0; j < Configuration[i] && i < Configuration.Count - 1; j++)
                    {
                        for (var k = 0; k < Configuration[i + 1]; k++)
                        {
                            Weights[i][j].Add(rand.NextDouble());
                        }

                    }
                }
            }

            public override string GetType()
            {
                return FunctionType == AnnRoot.ActivationType.BipolarSygmoid ? "Биполярная сигмоида" : "Позитивная сигмоида";
            }

            public override List<double> Predict(List<double> input)
            {
                if (!IsTrained || Configuration.Count == 0 || Configuration[0] != input.Count)
                {
                    Console.WriteLine("Problems!");
                }

                var prevOut = input;
                

                for (var layerIdx = 1; layerIdx < Configuration.Count; layerIdx++)
                {
                    var curOut = new List<double>();
                    for (var toIdx = 0; toIdx < Configuration[layerIdx]; toIdx++)
                    {
                        double tmp = 0;
                        for (var fromIdx = 0; fromIdx < Configuration[layerIdx-1]; fromIdx++)
                        {
                             tmp += Weights[layerIdx-1][fromIdx][toIdx] * prevOut[fromIdx];
                        }
                        curOut.Add( Activation(tmp));
                    }

                    prevOut = curOut;
                }

                return prevOut;
            }

            public ANeuralNetwork()
            {
                Configuration = new List<uint>();
                FunctionType = 0;
                Scale = 0;
            }
            public ANeuralNetwork(List<uint> configuration, AnnRoot.ActivationType activationType,
                           double scale)
            {
                Configuration = configuration;
                FunctionType = activationType;
                Scale = scale;
            }



            public override double BackPropTraining(List<List<double>> inputs, List<List<double>> outputs, int maxIteration = 10000, double eps = 0.1, double speed = 0.1,
                                                    bool stdDump = false)
            {
                RandomInit();
                if (inputs.Count != outputs.Count)
                    throw new Exception();

                double currentError;
                var currentIteration = 0;

                do
                {
                    currentError = inputs.Select((t, countIdx) => BackPropTrainingIteration(t, outputs[countIdx], speed)).Sum();

                    currentIteration++;
                    currentError = Math.Sqrt(currentError);

                    if (stdDump && currentIteration % 100 == 0)
                        Console.WriteLine("Iteration: " + currentIteration + "\tError: " + currentError);

                    if (currentError < eps)
                        IsTrained = true;

                } while (currentError > eps && currentIteration <= maxIteration);

                return currentError;
            }

            public override double BackPropTrainingIteration(List<double> input, List<double> output, double speed)

            {
                double currentError = 0;

                var tmpIn = new List<List<double>>();
                var tmpOut = new List<List<double>>();

                tmpIn.Add(input);
                tmpOut.Add(input);


                for (var layerIdx = 1; layerIdx < Configuration.Count; layerIdx++)
                {
                    tmpIn.Add(new List<double>(Convert.ToInt32(Configuration[layerIdx])));
                    tmpOut.Add(new List<double>(Convert.ToInt32(Configuration[layerIdx])));
                    for (var toIdx = 0; toIdx < Configuration[layerIdx]; toIdx++)
                    {
                        tmpIn[layerIdx].Add(0);
                        tmpOut[layerIdx].Add(0);
                        for (var fromIdx = 0; fromIdx < Configuration[layerIdx - 1]; fromIdx++)
                        {
                            tmpIn[layerIdx][toIdx] += tmpOut[layerIdx - 1][fromIdx] * Weights[layerIdx - 1][fromIdx][toIdx];
                        }

                        tmpOut[layerIdx][toIdx] = Activation(tmpIn[layerIdx][toIdx]);
                    }
                }

                var sigma = new List<List<double>>(Configuration.Count - 1);
                for (int i = 1; i < Configuration.Count; i++)
                {
                    sigma.Add(new List<double>());
                    for (int j = 0; j < Configuration[i]; j++)
                    {
                        sigma[i - 1].Add(0);
                    }
                }
                var dw = new List<List<List<double>>>(Weights.Count);
                for (int i = 0; i < Weights.Count; i++)
                {
                    dw.Add(new List<List<double>>((Weights[i].Count)));
                    for (int j = 0; j < Weights[i].Count; j++)
                    {
                        dw[i].Add(new List<double>());
                        for (int k = 0; k < Weights[i][j].Count; k++)
                        {
                            dw[i][j].Add(0);
                        }
                    }
                }

                for (var layerIdx = 0; layerIdx < output.Count; layerIdx++)
                {
                    sigma[^1][layerIdx] = (output[layerIdx] - tmpOut[^1][layerIdx]) * ActivationDerivative(tmpIn[^1][layerIdx]);
                    currentError += (output[layerIdx] - tmpOut[^1][layerIdx]) * (output[layerIdx] - tmpOut[^1][layerIdx]);
                }


                for (var layerIdx = Configuration.Count - 2; layerIdx > -1; --layerIdx)
                {
                    if (layerIdx < Configuration.Count - 2)
                    {
                        for (var fromIdx = 0; fromIdx < Configuration[layerIdx + 1]; fromIdx++)
                        {
                            for (var toIdx = 0; toIdx < Configuration[layerIdx + 2]; toIdx++)
                            {
                                sigma[layerIdx][fromIdx] += sigma[layerIdx + 1][toIdx] * Weights[layerIdx + 1][fromIdx][toIdx];
                            }

                            sigma[layerIdx][fromIdx] *= ActivationDerivative(tmpIn[layerIdx + 1][fromIdx]);
                        }
                    }

                    for (var fromIdx = 0; fromIdx < Configuration[layerIdx]; fromIdx++)
                    {
                        for (var toIdx = 0; toIdx < sigma[layerIdx].Count; toIdx++)
                        {

                            var tmpSigma = sigma[layerIdx][toIdx];
                            var tmpO = tmpOut[layerIdx][fromIdx];
                            dw[layerIdx][fromIdx][toIdx] = speed * tmpSigma * tmpO;
                        }
                    }
                }


                for (var layerIdx = 0; layerIdx < Weights.Count; layerIdx++)
                {
                    for (var fromIdx = 0; fromIdx < Weights[layerIdx].Count; fromIdx++)
                    {
                        for (var toIdx = 0; toIdx < Weights[layerIdx][fromIdx].Count; toIdx++)
                        {
                            Weights[layerIdx][fromIdx][toIdx] += dw[layerIdx][fromIdx][toIdx];
                        }
                    }
                }

                return currentError;
            }

            public override double Activation(double inputNeuron)
            {
                if (FunctionType == AnnRoot.ActivationType.PositiveSygmoid)
                {
                    return (1 / (1 + Math.Exp(-Scale * inputNeuron)));
                }
                else if (FunctionType == AnnRoot.ActivationType.BipolarSygmoid)
                {
                    return (2 / (1 + Math.Exp(-Scale * inputNeuron)) - 1);
                }

                return -1;
            }

            public override double ActivationDerivative(double inputNeuron)
            {
                if (FunctionType == AnnRoot.ActivationType.PositiveSygmoid)
                {
                    return Scale * (1 / (1 + Math.Exp(-Scale * inputNeuron)) * (1 - (1 / (1 + Math.Exp(-Scale * inputNeuron)))));
                }
                else if (FunctionType == AnnRoot.ActivationType.BipolarSygmoid)
                {
                    return Scale * 0.5 * (1 + (1 / (1 + Math.Exp(-Scale * inputNeuron))) * (1 - (1 / (1 + Math.Exp(-Scale * inputNeuron)))));
                }

                return -1;
            }
            public override string GetTestString()
            {
                return "Сеть обучена";
            }
        }

        public bool LoadData(string filepath, List<List<double>> inputs, List<List<double>> outputs)
        {
            StreamReader file = new StreamReader(filepath);
            string line = file.ReadLine();
            if (line != "input_count:")
                throw new Exception("incorrect file format");
            line = file.ReadLine();
            int inputCount = Convert.ToInt32(line);

            line = file.ReadLine();
            if (line != "output_count:")
                throw new Exception("incorrect file format");
            line = file.ReadLine();
            int outputCount = Convert.ToInt32(line);

            line = file.ReadLine();
            if (line != "example_count:")
                throw new Exception("incorrect file format");
            line = file.ReadLine();
            var exampleCount = Convert.ToInt32(line);

            line = file.ReadLine();
            if (line != "data:")
                throw new Exception("incorrect file format");

            for (var i = 0; i < exampleCount; ++i)
            {
                inputs.Add(new List<double>(inputCount));
                var tmp = file.ReadLine()?.Replace('.', ',').Split(' ').Select(double.Parse).ToList();
                if (tmp != null)
                {
                    foreach (var element in tmp)
                    {
                        inputs[i].Add(element);
                    }

                    outputs.Add(new List<double>(outputCount));
                }

                tmp = file.ReadLine()?.Replace('.', ',').Split(' ').Select(double.Parse).ToList();
                if (tmp != null)
                    foreach (var element in tmp)
                    {
                        outputs[i].Add(element);
                    }

                file.ReadLine();
            }

            return true;
        }

        public bool SaveData(string filepath, List<List<double>> inputs, List<List<double>> outputs)
        {
            {
                //if (inputs.size() != outputs.size())
                //    throw "input size and output size must be the same";
                //if (inputs.size() * outputs.size() == 0)
                //    throw "empty data";
                //size_t input_count = inputs[0].size();
                //size_t output_count = outputs[0].size();
                //for (size_t i = 0; i < inputs.size(); i++)
                //{
                //    if (inputs[i].size() != input_count)
                //        throw "incorrect input size";
                //    if (outputs[i].size() != output_count)
                //        throw "incorrect output size";
                //}
                //std::ofstream file(filepath);
                //if (!file.is_open()) return false;
                //file << std::setprecision(9);
                //file << "input_count:" << std::endl;
                //file << inputs[0].size() << std::endl;
                //file << "output_count:" << std::endl;
                //file << outputs[0].size() << std::endl;
                //file << "primer_count:" << std::endl;
                //file << inputs.size() << std::endl;
                //file << "data:" << std::endl;
                //for (size_t i = 0; i < inputs.size(); i++)
                //{
                //    for (size_t j = 0; j < input_count; j++)
                //    {
                //        file << inputs[i][j] << "\t";
                //    }
                //    file << std::endl;
                //    for (size_t j = 0; j < output_count; j++)
                //    {
                //        file << outputs[i][j] << "\t";
                //    }
                //    file << std::endl;
                //    file << std::endl;
                //}
                //file.close();
                //return true;

                return true;
            }
        }
    }
    
}