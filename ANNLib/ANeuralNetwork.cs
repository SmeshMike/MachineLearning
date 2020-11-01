using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;

namespace ANNLib
{
    class ANN
    {

        public class ANeuralNetwork : IANeuralNetwork
        {

            

            /**
			* Прочитать нейронную сеть из файла. Сеть сохраняется вызовом метода Save.
			* @param filepath - имя и путь до файла с сетью.
			* @return - успешность считывания.
			*/
            public bool Load(string filepath)
            {
                StreamReader file = new StreamReader(filepath);
                string line;
                int buffer;
                line = file.ReadLine();
                if(!line.Contains("Тип активационной функции:"))
                    throw new Exception("incorrect file format");
                buffer = Convert.ToInt32(line.Substring(26).Trim());
                Configuration = (IANeuralNetwork.ActivationType)buffer;
                IANeuralNetwork.isTrained = true;

                line = file.ReadLine();
                if (!line.Contains("Масштабирующий коэффициент аргумента сигмоиды:"))
                    throw new Exception("incorrect file format");
                IANeuralNetwork.scale = Convert.ToDouble(line.Substring(46).Trim());

                line = file.ReadLine();
                line = file.ReadLine();
                if (!line.Contains("Размер:"))
                    throw new Exception("incorrect file format");
                line = line.Substring(7).Trim();
                IANeuralNetwork.configuration = line.Split().Cast<uint>().ToList();

                return true;
			}

            /**
			* Сохранить нейронную сеть в файл. Сеть загружается вызовом метода Load.
			* @param filepath - имя и путь до файла с сетью.
			* @return - успешность сохранения.
			*/
            public bool Save(string filepath)
            {
                if (IANeuralNetwork.isTrained)
                {
                    Console.WriteLine("Нейросеть ещё не обучена");
                    return false;
                }
                else
                {
                    StreamWriter file = new StreamWriter(filepath);
                    file.WriteLine("Тип активационной функции:" + IANeuralNetwork.activationType + "\nМасштабирующий коэффициент аргумента сигмоиды:" + IANeuralNetwork.scale + "\nКонфигурация сети:\nРазмер:" + IANeuralNetwork.configuration.Count + "\nДанные:\n");
                    foreach (var element in IANeuralNetwork.configuration)
                    {
                        file.Write(element + "\t");
                    }
                    file.WriteLine("Веса:\n");
                    foreach (var weightMatrix in IANeuralNetwork.weights)
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
                    file.Close();
                    return true;
                }
            }

            /**
			* Проинициализирвать веса сети случайным образом.
			*/
            public void RandomInit()
            {
                Random rand = new Random();


                IANeuralNetwork.weights = new List<List<List<double>>>(IANeuralNetwork.configuration.Count());
                for (var layer_index = 0; layer_index < IANeuralNetwork.configuration.Count; layer_index++)
                {
                    IANeuralNetwork.weights[layer_index] = new List<List<double>>(Convert.ToInt32(IANeuralNetwork.configuration[layer_index]));
                    for (var from_index = 0; from_index < IANeuralNetwork.weights[layer_index].Count(); from_index++)
                    {
                        IANeuralNetwork.weights[layer_index][from_index]= new List<double>(Convert.ToInt32(IANeuralNetwork.configuration[layer_index + 1]));
                        for (var to_index = 0; to_index < IANeuralNetwork.weights[layer_index][from_index].Count(); to_index++)
                        {
                            IANeuralNetwork.weights[layer_index][from_index][to_index] = rand.NextDouble();
                        }
                    }
                }
            }

            /**************************************************************************/
            /**********************ЭТО ВАМ НАДО РЕАЛИЗОВАТЬ САМИМ**********************/
            /**************************************************************************/

            /**
			* Получить строку с типом сети.
			* @return описание сети, содержит запись о типе нейронной сети и авторе библиотеки.
			*/
            public virtual string GetType()
            {
                return IANeuralNetwork.activationType == IANeuralNetwork.ActivationType.BipolarSygmoid  ? "Биполярная сигмоида": "Позитивная сигмоида";
            }

            /**
		    * Спрогнозировать выход по заданному входу.
		    * @param input - вход, длина должна соответствовать количеству нейронов во входном слое.
		    * @return выход сети, длина соответствует количеству нейронов в выходном слое.
		    */
            public virtual List<double> Predict(List<double> input)
            {
                
            }

            /**
		    * Создать нейронную сеть
		    * @param configuration - конфигурация нейронной сети.
		    *   Каждый элемент представляет собой количество нейронов в очередном слое.
		    * @param activation_type - тип активационной функции (униполярная, биполярная).
		    * @param scale - масштаб активационной функции.
		    */


            public ANeuralNetwork[] CreateNeuralNetwork(List<uint> configuration, Enum activationType, double scale)
            {
                return null;
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

            public double BackPropTraining<T>(List<List<float>> inputs, List<List<float>> outputs, int maxIters = 10000, double eps = 0.1, double speed = 0.1,
                                              bool std_dump = false)
            {
                return 0;
            }

            public List<uint> GetConfiguration()
            {
                return IANeuralNetwork.configuration;
            }

            /**
			* Провести одну итерацию обучения методом обратного распространения ошибки.
			* @param ann - нейронная сеть, которую необходимо обучить.
			* @param input - вход для обучения.
			* @param outputs - выход для обучения.
			* @param speed - скорость обучения.
			*/
            public double BackPropTrainingIteration(List<double> input, List<double> output, float speed)
            {
                return 0;
            }

            /***************************************************************************/
            /***************************************************************************/


            

            /**
			* Вычислить значение активационной функции.
			* @param neuron_input - входное значение нейрона.
			* @return - значение активационной фунции.
			*/
            public double Activation(double inputNeuron)
            {
                if (IANeuralNetwork.activationType == IANeuralNetwork.ActivationType.PositiveSygmoid) {
                    return (1 / (1 + Math.Exp(-IANeuralNetwork.scale * inputNeuron)));
                }
                else if (IANeuralNetwork.activationType == IANeuralNetwork.ActivationType.BipolarSygmoid) {
                    return (2 / (1 + Math.Exp(-IANeuralNetwork.scale * inputNeuron)) - 1);
                }
                return -1;
            }

            /**
* Вычислить значение производной активационной функции.
* @param activation - значение активационной фнункции, для которой хотим вычислить производную.
* @return - значение производной активационной фунции.
*/
            public double ActivationDerivative(double activation)
            {
                if (IANeuralNetwork.activationType == IANeuralNetwork.ActivationType.PositiveSygmoid)
                {
                    return IANeuralNetwork.scale * activation * (1 - activation);
                }
                else if (IANeuralNetwork.activationType == IANeuralNetwork.ActivationType.BipolarSygmoid)
                {
                    return IANeuralNetwork.scale * 0.5f * (1 + activation) * (1 - activation);
                }
                return -1;
            }

            /**
    * Тестовая функция для проверки подключения библиотеки.
    * @return строка с поздравлениями.
    */

            //protected
            public string GetTestString()
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
            var text = file.ReadToEnd();
            string line;
            while ((line = file.ReadLine()) != null)
            {
                Console.WriteLine(line);
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
            return true;
        }
    }
}