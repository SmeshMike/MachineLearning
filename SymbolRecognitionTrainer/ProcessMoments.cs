using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AnnLibrary;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.ML;
using FeatureExtractionLibrary;
using static FeatureExtractionLibrary.PolynomialManager;

namespace SymbolRecognitionTrainer
{
    class ProcessMoments
    {
        /**Массив распознанных значений. Ключ - номер активированного выходного нейрона. 
        Значение - значение цифры.*/
        List<string> values;

        string Recognize(ComplexMoments moments)
        {
            
            ANeuralNetwork networt = new ANeuralNetwork();
            var output = networt.Predict(moments.ToListOfDouble());
            return output.IndexOf(output.Max()).ToString();
        }

        public SortedDictionary<string, List<ComplexMoments>> GenerateMoments(string path, int diameter, PolynomialManager polyManager)
        {
            var sampleDirs = Directory.GetDirectories(path).ToList();
            SortedDictionary<string, List<ComplexMoments>> result = new SortedDictionary<string, List<ComplexMoments>>();
            //Перебираем все папки.
            for (var i = 0; i < sampleDirs.Count; i++)
            {
                // Ищем файл со значением буквы.
                var tmpFile = new StreamReader(sampleDirs[i] + "\\" + "value.txt");
                var key = tmpFile.ReadToEnd();
                tmpFile.Close();
                var files = Directory.GetFiles( sampleDirs[i] + "\\", "*.png");
                var tmp = new List<ComplexMoments>();
                foreach (var file in files)
                {
                    //Считываем картинку.
                    var moment = new ComplexMoments();
                    //Обрабатываем.
                    ProcessOneImage(file, polyManager,diameter, out moment);
                    tmp.Add(moment);
                }
                //Сохраняем
                result.Add(key, tmp);
            }

            return result;
        }

        public void ProcessOneImage(string imagePath,PolynomialManager polyManager, int diameter, out ComplexMoments res)
        {
            var image = CvInvoke.Imread(imagePath, ImreadModes.Grayscale);
            if (image.IsEmpty)
            {
                throw new Exception("Empty image");
            }
            CvInvoke.Threshold(image, image, 127, 255, ThresholdType.BinaryInv);
            var blobs = polyManager.DetectBlobs(image);
            if (blobs.Count != 1)
            {
                throw new Exception("Incorrect input data. More then one blob.");
            }
            var nblobs = polyManager.NormalizeBlobs(blobs, diameter);
            res = polyManager.Decompose(nblobs[0]);
        }
        double PrecisionTest(SortedDictionary<string, List<ComplexMoments>> moments)
        {
            int right_answers = 0;
            int tests = 0;
            foreach (var moment in moments)
            {
                for (int j = 0; j < moments.Count; j++)
                {
                    var recognized = Recognize(moment.Value[j]);
                    if (recognized == moment.Key)
                    {
                        right_answers++;
                    }
                    tests++;
                }
            }
            return (double)right_answers / (double)tests;
        }

        bool Save(string filename)
        {
            FileStorage fs = new FileStorage(filename, FileStorage.Mode.Write);
            if (!fs.IsOpened)
            {
                return false;
            }
            ANN_MLP network = new ANN_MLP();
            network.Write(fs);
            fs.Write("values" + values);
            fs.ReleaseAndGetString();
            return true;
        }

        bool Read(string filename)
        {
            FileStorage fs = new FileStorage(filename, FileStorage.Mode.Read);
            if (!fs.IsOpened)
            {
                return false;
            }
            ANN_MLP network = new ANN_MLP();
            network.Read(fs.GetRoot());
            values.Clear();
            //for (var iter = fs["values"]..begin(); iter != fs["values"].end(); iter++)
            //{
            //    values.push_back(*iter);
            //}

            fs.ReleaseAndGetString();
            return true;
        }

        public bool Train(List<uint> layers, SortedDictionary<string, List<ComplexMoments>> moments, int max_iters = 100000, double eps = 0.1, double speed = 0.1)
        {
            ANeuralNetwork network = new ANeuralNetwork(layers, AnnRoot.ActivationType.BipolarSygmoid, 1);
            var inputs = new List<List<double>>();
            var outputs = new List<List<double>>();
            for (var i = 0; i < 4; ++i)////9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
            {
                var output = new List<double>();
                for (int j = 0; j < 4; j++)////9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
                {
                    if (j == i)
                        output.Add(1);
                    else
                        output.Add(0);
                }

                for (int j = 0; j < moments[i.ToString()].Count; j++)
                {
                    var input = moments[i.ToString()][j].ToListOfDouble();
                    inputs.Add(input);
                    outputs.Add(output);
                }

                Console.WriteLine(i + " на вход пришла");
            }

            Console.WriteLine("Начинаем обучение");

            network.BackPropTraining(inputs, outputs, max_iters, eps, speed, true,250);
            //network.Save("..\\..\\..\\..\\savedData.txt");
            return true;

        }

        public void Check()
        {
            ANeuralNetwork network = new ANeuralNetwork();
            network.Load("..\\..\\..\\..\\savedData.txt");
            var sampleDirs = Directory.GetDirectories("..\\..\\..\\..\\Data\\TestData\\").ToList();
            int fCount = Directory.GetFiles("..\\..\\..\\..\\Data\\GroundData\\", "*.png", SearchOption.AllDirectories).Length;
            double precision = 0;
            SortedDictionary<string, List<ComplexMoments>> result = new SortedDictionary<string, List<ComplexMoments>>();
            var poliManager = new PolynomialManager();
            poliManager.InitBasis(10, 50);
            foreach (var dir in sampleDirs)
            {
                var files = Directory.GetFiles(dir, "*.png", SearchOption.TopDirectoryOnly).ToList();
                var value = Convert.ToInt32(new StreamReader(dir + "\\" + "value.txt").ReadToEnd());

                foreach (var file in files)
                {
                    ComplexMoments tmpMoments;
                    ProcessOneImage(file, poliManager, 50, out tmpMoments);
                    var tmpInput = tmpMoments.ToListOfDouble();

                    var output = network.Predict(tmpInput);
                    var predictedValue = Convert.ToInt32(output.Max());
                    if (predictedValue == value)
                        precision += 100 / fCount;
                }
                Console.WriteLine("Точность " + precision + "%");
            }
        }

	}
}
