using System;
using System.Collections.Generic;
using ANNLib;

namespace ANNSample
{
    class Program
    {
        static void Main(string[] args)
        {
			Console.WriteLine( "hello ANN!" );
            ANN.ANeuralNetwork tmp = new ANN.ANeuralNetwork();
            tmp.GetTestString();
            var Configurate = new List<uint>();//Создаем конфигурацию сети, состоящую из трех слоев

            var NR = tmp.CreateNeuralNetwork(Configurate, RootANN.ActivationType.BipolarSygmoid, 1);
            List<List<double>> inputs, outputs;//создаем пустые вектора под  входные и выходные данные 
            NR.Load("..\\Debug\\xor.nn");
            Console.WriteLine(NR.GetType());//в каждые 100 итераций выводит данные в строку
            //LoadData("..\\Debug\\xor.data", inputs, outputs);// выгружаем данные из файла
            //vector<float> inp;
            //for (int i = 0; i < inputs.size(); i++)
            //{
            //    cout << inputs[i][0] << "\t" << inputs[i][0] << "\t" << NR->Predict(inputs[i])[0] << endl; ;
            //}

            //return 0;
		}
    }
}
