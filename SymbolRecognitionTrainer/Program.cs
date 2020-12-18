using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AnnLibrary;
using FeatureExtractionLibrary;

namespace SymbolRecognitionTrainer
{
    class Program
    {
        //static void TrainNetwork()
        //{
        //    List<uint> Configurate = new List<uint>();//Создаем конфигурацию сети, состоящую из трех слоев
        //    Configurate.Add(2);//1-ой входной слой  сети с двумя нейронами
        //    Configurate.Add(10);//2-ой слой сети с двумя нейронами
        //    Configurate.Add(1);//4-й выходнрой слой сети с одним нейроном
        //    ANeuralNetwork NR = new ANeuralNetwork(Configurate, AnnRoot.ActivationType.BipolarSygmoid, 1);
        //    List<List<double>> inputs = new List<List<double>>();
        //    List<List<double>> outputs = new List<List<double>>();//создаем пустые вектора под  входные и выходные данные 
        //    NR.LoadData("..\\..\\..\\..\\data.txt", inputs, outputs);// выгружаем данные из файла
        //    // инцилизируем веса сети случайным образом
        //    NR.BackPropTraining(inputs, outputs, 300000, 0.1, 0.05, true);//Обучение сети метдом оьбратного распространения ошибки
        //    Console.WriteLine(NR.GetType());//в каждые 100 итераций выводит данные в строку
        //    NR.Save("..\\..\\..\\..\\savedData.txt");//сохраняем в файл
        //}

        static void Main(string[] args)
        {
			string key;
            do
            {
                Console.WriteLine("===Enter next values to do something:===");
                Console.WriteLine("  '1' - to generate data.");
                Console.WriteLine("  '2' - to train network.");
                Console.WriteLine("  '3' - to check recognizing precision.");
                Console.WriteLine("  '4' - to recognize single image.");
                Console.WriteLine("  'exit' - to close the application.");
                Console.WriteLine();
                key = Console.ReadLine();
                var pm = new ProcessMoments();
                if (key == "1")
                {
                    var poliManager = new PolynomialManager();
                    var pd = new ProcessData();
                    poliManager.InitBasis(10, 50);
                    pd.DistributeData("..\\..\\..\\..\\Data\\LabeledData\\", "..\\..\\..\\..\\Data\\GroundData\\", "..\\..\\..\\..\\Data\\TestData\\", 50);
                    var dictionary = pm.GenerateMoments("..\\..\\..\\..\\Data\\GroundData\\", 50,poliManager);
                    pd.SaveMoments("..\\..\\..\\..\\Moments.yaml", dictionary);

                }
                else if (key == "2")
                {
                    var pd = new ProcessData();
                    var tmpDict = new SortedDictionary<string, List<ComplexMoments>>();
                    pd.ReadMoments("..\\..\\..\\..\\Moments.yaml", tmpDict);
                    var layers = new List<uint> {100, 150, 60, 25, 10,4};
                    pm.Train(layers, tmpDict, 300000, 0.5, 0.00005);
                }
                else if (key == "3")
                {
                    pm.Check();
                }
                else if (key == "4")
                {
                    
                }
            } while (key != "exit");
		}
    }
}
