using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;

namespace ANNLib
{
    public class ANN
    {

        public class ANeuralNetwork : RootANN
        {
            /**
			* Прочитать нейронную сеть из файла. Сеть сохраняется вызовом метода Save.
			* @param filepath - имя и путь до файла с сетью.
			* @return - успешность считывания.
			*/
            public override bool Load(string filepath)
            {
                StreamReader file = new StreamReader(filepath);
                string line;
                int buffer;
                line = file.ReadLine();
                if (!line.Contains("activation type:"))
                    throw new Exception("incorrect file format");
                buffer = Convert.ToInt32(line.Substring(26).Trim());
                FunctionType = (ActivationType) buffer;
                IsTrained = true;

                line = file.ReadLine();
                if (!line.Contains("activation scale:"))
                    throw new Exception("incorrect file format");
                Scale = Convert.ToDouble(line.Substring(46).Trim());

                line = file.ReadLine();
                if (!line.Contains("configuration:"))
                    throw new Exception("incorrect file format");
                line = line.Substring(14).Trim();
                Configuration = line.Split().Cast<uint>().ToList();

                line = file.ReadLine();
                line = file.ReadLine();
                if (!line.Contains("weigths:"))
                    throw new Exception("incorrect file format");
                Weights = new List<List<List<double>>>(Configuration.Count - 1);
                for (var layer_idx = 0; layer_idx < Configuration.Count - 1; layer_idx++)
                {
                    Weights[layer_idx] = new List<List<double>>((int) Configuration[layer_idx]);
                    for (var from_idx = 0; from_idx < Weights[layer_idx].Count; from_idx++)
                    {
                        Weights[layer_idx][from_idx] = new List<double>((int) Configuration[layer_idx + 1]);
                        for (var to_idx = 0; to_idx < Weights[layer_idx][from_idx].Count; to_idx++)
                        {
                            Weights[layer_idx][from_idx][to_idx] = Convert.ToDouble(line);
                        }
                    }
                }

                return true;
            }

            /**
			* Сохранить нейронную сеть в файл. Сеть загружается вызовом метода Load.
			* @param filepath - имя и путь до файла с сетью.
			* @return - успешность сохранения.
			*/
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
                file.WriteLine(Configuration.Count);
                foreach (int neuronCount in Configuration)
                {
                    file.Write(neuronCount + "\t");
                }

                file.WriteLine("weights:");
                foreach (var weightMatrix in Weights)
                {
                    foreach (var weightLine in weightMatrix)
                    {
                        foreach (var weight in weightLine)
                        {
                            file.Write(weight + " ");
                        }

                        file.WriteLine();
                    }
                }

                return true;

            }


            /**
			* Проинициализирвать веса сети случайным образом.
			*/
            public override void RandomInit()
            {
                Random rand = new Random();

                Weights = new List<List<List<double>>>(Configuration.Count());

                for (int i = 0; i < Configuration.Count; i++)
                {
                    Weights.Add(new List<List<double>>((int) Configuration[i]));

                    for (int j = 0; j < Configuration[i] && i < Configuration.Count - 1; j++)
                    {
                        Weights[i].Add(new List<double>((int) Configuration[i + 1]));
                    }

                    for (int j = 0; j < Configuration[i] && i < Configuration.Count - 1; j++)
                    {
                        for (int k = 0; k < Configuration[i + 1]; k++)
                        {
                            Weights[i][j].Add(rand.NextDouble());
                        }

                    }
                }


                //for (var layer_index = 0; layer_index < Configuration.Count; layer_index++)
                //{

                //    for (var from_index = 0; from_index < Weights[layer_index].Count(); from_index++)
                //    {
                //        //Weights[layer_index][from_index] =
                //         //   new List<double>(Convert.ToInt32(Configuration[layer_index + 1]));
                //        for (var to_index = 0; to_index < Weights[layer_index][from_index].Count(); to_index++)
                //        {
                //            Weights[layer_index][from_index][to_index] = rand.NextDouble();
                //        }
                //    }
                //}
            }

            /**************************************************************************/
            /**********************ЭТО ВАМ НАДО РЕАЛИЗОВАТЬ САМИМ**********************/
            /**************************************************************************/

            /**
			* Получить строку с типом сети.
			* @return описание сети, содержит запись о типе нейронной сети и авторе библиотеки.
			*/
            public override string GetType()
            {
                return FunctionType == ActivationType.BipolarSygmoid ? "Биполярная сигмоида" : "Позитивная сигмоида";
            }

            /**
		    * Спрогнозировать выход по заданному входу.
		    * @param input - вход, длина должна соответствовать количеству нейронов во входном слое.
		    * @return выход сети, длина соответствует количеству нейронов в выходном слое.
		    */
            public override List<double> Predict(List<double> input)
            {
                if (!IsTrained || Configuration.Count == 0 || Configuration[0] != input.Count
                )//если сеть не обучена или конфигурция пуста 
                {
                    Console.WriteLine("Problems!");//то у нас проблемы
                }

                List<double> prevOut = input;            //вектор входов 
                List<double> curOut = new List<double>();//вектор выходов 

                for (var layerIdx = 0; layerIdx < Configuration.Count - 1; layerIdx++)//цикл по количеству слоев
                {
                    for (var toIdx = 0; toIdx < Configuration[layerIdx + 1]; toIdx++)
                    {
                        for (var fromIdx = 0; fromIdx < Configuration[layerIdx]; fromIdx++)
                        {
                            curOut[toIdx] += Weights[layerIdx][fromIdx][toIdx] * prevOut[fromIdx];
                        }

                        curOut[toIdx] = Activation(curOut[toIdx]);
                    }

                    prevOut = curOut;
                }

                return prevOut;
            }

            /**
		    * Создать нейронную сеть
		    * @param configuration - конфигурация нейронной сети.
		    *   Каждый элемент представляет собой количество нейронов в очередном слое.
		    * @param activation_type - тип активационной функции (униполярная, биполярная).
		    * @param scale - масштаб активационной функции.
		    */


            public override ANeuralNetwork CreateNeuralNetwork(List<uint> configuration, ActivationType activationType,
                                                               double scale)
            {
                return new ANeuralNetwork() {Configuration = configuration, FunctionType = activationType, Scale = scale};
            }

            /**
			* Обучить сеть методом обратного распространения ошибки.
			* В ходе работы метода, после выполнения обучения флаг is_trained должен устанавливаться в true.
			* @param ann - нейронная сеть, которую необходимо обучить.
			* @param inputs - входы для обучения.
			* @param outputs - выходы для обучения.
			* @param max_iters - максимальное количество итераций при обучении.
			* @param eps - средняя ошибка по всем примерам при которой происходит остановка обучения.
			* @param speed - скорость обучения.
			* @param std_dump - сбрасывать ли информацию о процессе обучения в стандартный поток вывода?
			*/

            public override double BackPropTraining(List<List<double>> inputs, List<List<double>> outputs,
                                                    int maxIters = 10000, double eps = 0.1, double speed = 0.1, bool std_dump = false)
            {
                RandomInit();                     //рандомим
                if (inputs.Count != outputs.Count)//Если количество входов не равно количеству выходов, то вылетает исключение
                    throw new Exception();

                double currentError;//создаем 
                int currentIter = 0;

                do
                {
                    currentError = 0;
                    for (var countIdx = 0; countIdx < inputs.Count; countIdx++)
                        currentError += BackPropTrainingIteration(inputs[countIdx], outputs[countIdx], speed);

                    currentIter++;
                    currentError = Math.Sqrt(currentError);

                    if (std_dump && currentIter % 100 == 0)
                        Console.WriteLine("Iteration: " + currentIter + "\tError: " + currentError);

                    if (currentError < eps)
                        IsTrained = true;

                } while (currentError > eps && currentIter <= maxIters);

                return currentError;
            }

            /**
			* Провести одну итерацию обучения методом обратного распространения ошибки.
			* @param ann - нейронная сеть, которую необходимо обучить.
			* @param input - вход для обучения.
			* @param outputs - выход для обучения.
			* @param speed - скорость обучения.
			*/
            public override double BackPropTrainingIteration(List<double> input, List<double> output, double speed)
            {
                double currentError = 0;//счетчик ошибок

                List<List<double>> tmpOut = new List<List<double>>(); 
                //первый выход равен входу
                tmpOut.Add(input);

                //прямой ход
                for (var layerIdx = 0; layerIdx < Configuration.Count - 1; layerIdx++)//цикл по слоям
                {
                    tmpOut.Add(new List<double>(Convert.ToInt32(Configuration[layerIdx + 1])));
                    tmpOut[layerIdx + 1].Capacity = Convert.ToInt32(Configuration[layerIdx + 1]);
                    for (var toIdx = 0; toIdx < Configuration[layerIdx + 1]; toIdx++)// цикл 
                    {
                        tmpOut[layerIdx + 1].Add(new double());
                        tmpOut[layerIdx + 1][toIdx] = 0;
                        for (var fromIdx = 0; fromIdx < Configuration[layerIdx]; fromIdx++)
                        {
                            tmpOut[layerIdx + 1][toIdx] += tmpOut[layerIdx][fromIdx] * Weights[layerIdx][fromIdx][toIdx];
                        }

                        tmpOut[layerIdx + 1][toIdx] = Activation(tmpOut[layerIdx + 1][toIdx]);
                    }
                }

                var sigma = new List<List<double>>(Configuration.Count);
                var dw = new List<List<List<double>>>(Configuration.Count - 1);
                sigma.Last().Add(tmpOut.Last().Count);

                for (var layerIdx = 0; layerIdx < output.Count; layerIdx++)
                {
                    sigma.Last()[layerIdx] = (output[layerIdx] - tmpOut.Last()[layerIdx]) * ActivationDerivative(tmpOut.Last()[layerIdx]);
                    currentError += (output[layerIdx] - tmpOut.Last()[layerIdx]) * (output[layerIdx] - tmpOut.Last()[layerIdx]);
                }

                //обратный ход
                for (var layerIdx = Configuration.Count - 2; layerIdx + 1 != 0; layerIdx--)
                {
                    dw[layerIdx].Capacity = (Weights[layerIdx].Count);
                    sigma[layerIdx].Capacity = Convert.ToInt32(Configuration[layerIdx]);

                    for (var fromIdx = 0; fromIdx < Configuration[layerIdx]; fromIdx++)
                    {
                        for (var toIdx = 0; toIdx < Configuration[layerIdx + 1]; toIdx++)
                        {
                            sigma[layerIdx][fromIdx] += sigma[layerIdx + 1][toIdx] * Weights[layerIdx][fromIdx][toIdx];
                        }

                        sigma[layerIdx][fromIdx] *= ActivationDerivative(tmpOut[layerIdx][fromIdx]);
                        dw[layerIdx][fromIdx].Capacity = (Weights[layerIdx][fromIdx].Count);

                        for (var toIdx = 0; toIdx < Configuration[layerIdx + 1]; toIdx++)
                        {
                            dw[layerIdx][fromIdx][toIdx] = speed * sigma[layerIdx + 1][toIdx] * tmpOut[layerIdx][fromIdx];
                        }
                    }
                }

                //модификация весов
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

            /***************************************************************************/
            /***************************************************************************/




            /**
			* Вычислить значение активационной функции.
			* @param neuron_input - входное значение нейрона.
			* @return - значение активационной фунции.
			*/
            public override double Activation(double inputNeuron)
            {
                if (FunctionType == ActivationType.PositiveSygmoid)
                {
                    return (1 / (1 + Math.Exp(-Scale * inputNeuron)));
                }
                else if (FunctionType == ActivationType.BipolarSygmoid)
                {
                    return (2 / (1 + Math.Exp(-Scale * inputNeuron)) - 1);
                }

                return -1;
            }

            /**
            * Вычислить значение производной активационной функции.
            * @param activation - значение активационной фнункции, для которой хотим вычислить производную.
            * @return - значение производной активационной фунции.
            */
            public override double ActivationDerivative(double activation)
            {
                if (FunctionType == ActivationType.PositiveSygmoid)
                {
                    return Scale * activation * (1 - activation);
                }
                else if (FunctionType == ActivationType.BipolarSygmoid)
                {
                    return Scale * 0.5f * (1 + activation) * (1 - activation);
                }

                return -1;
            }

            /**
            * Тестовая функция для проверки подключения библиотеки.
            * @return строка с поздравлениями.
            */

            //protected
            public override string GetTestString()
            {
                return "Сеть обучена";
            }
        }

        /**
        * Считать данные из файла.
        * @param filepath - путь и имя к файлу с данными.
        * @param inputs - буфер для записи входов.
        * @param outputs - буфер для записи выходов.
        * @return - успешность чтения.
        */
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
            int exapmle_count = Convert.ToInt32(line);

            line = file.ReadLine();
            if (line != "data:")
                throw new Exception("incorrect file format");

            for (int i = 0; i < exapmle_count; i++)
            {
                inputs.Add(new List<double>(inputCount));
                var tmp = file.ReadLine()?.Replace('.', ',').Split(' ').Select(double.Parse).ToList();
                foreach (var element in tmp)
                {
                    inputs[i].Add(element);
                }

                outputs.Add(new List<double>(outputCount));
                tmp = file.ReadLine()?.Replace('.', ',').Split(' ').Select(double.Parse).ToList();
                foreach (var element in tmp)
                {
                    inputs[i].Add(element);
                }

                file.ReadLine();
            }

            return true;
        }

        /**
        * Записать данные в файл.
        * @param filepath - путь и имя к файлу с данными.
        * @param inputs - входы для записи.
        * @param outputs - выходы для записи.
        * @return - успешность записи.
        */
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