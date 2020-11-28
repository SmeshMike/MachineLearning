using System;
using System.Collections.Generic;
using ANNLib;

namespace ANNTrainer
{
    class Program
    {
        static void Main(string[] args)
        {
			Console.WriteLine("hello ANN!");
            ANN.ANeuralNetwork tmp = new ANN.ANeuralNetwork();
            ANN tmp1 = new ANN();
            tmp.GetTestString();
            List<uint> Configurate = new List<uint>();//Создаем конфигурацию сети, состоящую из трех слоев
            Configurate.Add(2);//1-ой входной слой  сети с двумя нейронами
            Configurate.Add(10);//2-ой слой сети с двумя нейронами
            Configurate.Add(1);//4-й выходнрой слой сети с одним нейроном
            ANN.ANeuralNetwork NR = tmp.CreateNeuralNetwork(Configurate, RootANN.ActivationType.BipolarSygmoid, 1);
            List<List<double>> inputs = new List<List<double>>();
            List<List<double>> outputs = new List<List<double>>();//создаем пустые вектора под  входные и выходные данные 
            tmp1.LoadData("..\\..\\..\\..\\data.txt", inputs, outputs);// выгружаем данные из файла
            // инцилизируем веса сети случайным образом
            NR.BackPropTraining( inputs, outputs, 300000, 0.1, 0.05, true);//Обучение сети метдом оьбратного распространения ошибки
            Console.WriteLine(NR.GetType());//в каждые 100 итераций выводит данные в строку
            NR.Save("..\\..\\..\\..\\savedData.txt");//сохраняем в файл
        }
    }
}
