using System;
using System.Collections.Generic;
using AnnLibrary;

namespace ANNTrainer
{
    class Trainer
    {
        static void Main(string[] args)
        {
            List<uint> Configurate = new List<uint>();//Создаем конфигурацию сети, состоящую из трех слоев
            Configurate.Add(2);//1-ой входной слой  сети с двумя нейронами
           //2-ой слой сети с двумя нейронами
           Configurate.Add(5);
           Configurate.Add(3);
            Configurate.Add(1);//4-й выходнрой слой сети с одним нейроном
            ANeuralNetwork NR = new ANeuralNetwork(Configurate, AnnRoot.ActivationType.PositiveSygmoid, 1);
            List<List<double>> inputs = new List<List<double>>();
            List<List<double>> outputs = new List<List<double>>();//создаем пустые вектора под  входные и выходные данные 
            NR.LoadData("..\\..\\..\\..\\PerceptronData.txt", inputs, outputs);// выгружаем данные из файла
            // инцилизируем веса сети случайным образом
            NR.BackPropTraining( inputs, outputs, 3000000, 0.01, 1, true);//Обучение сети метдом оьбратного распространения ошибки
            Console.WriteLine(NR.GetType());//в каждые 100 итераций выводит данные в строку
            NR.Save("..\\..\\..\\..\\PerceptronSavedData.txt");//сохраняем в файл
        }
    }
}
