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
                    poliManager.InitBasis(15, 100);
                    pd.DistributeData("..\\..\\..\\..\\Data\\LabeledData\\", "..\\..\\..\\..\\Data\\GroundData\\", "..\\..\\..\\..\\Data\\TestData\\", 50);
                    var dictionary = pm.GenerateMoments("..\\..\\..\\..\\Data\\GroundData\\", 100,poliManager);
                    pd.SaveMoments("..\\..\\..\\..\\Moments.yaml", dictionary);

                }
                else if (key == "2")
                {
                    var pd = new ProcessData();
                    var tmpDict = new SortedDictionary<string, List<ComplexMoments>>();
                    pd.ReadMoments("..\\..\\..\\..\\Moments.yaml", tmpDict);
                    var layers = new List<uint> {225,160, 60, 25, 9};
                    pm.Train(layers, tmpDict, 100000, 0.4, 0.01);
                }
                else if (key == "3")
                {
                    pm.Check();
                }
                else if (key == "4")
                {
                    pm.CheckOne();
                }
            } while (key != "exit");
		}
    }
}
