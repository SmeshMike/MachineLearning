using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AnnLibrary;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.ML;
using FeatureExtractionLibrary;

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
                var files = Directory.GetFiles(sampleDirs[i] + "\\", "*.png");
                var tmp = new List<ComplexMoments>();
                foreach (var file in files)
                {
                    //Считываем картинку.
                    //Обрабатываем.
                    ProcessOneImage(file, polyManager, diameter, out var moment);
                    tmp.Add(moment);
                }

                //Сохраняем
                result.Add(key, tmp);
            }

            return result;
        }

        public void ProcessOneImage(string imagePath, PolynomialManager polyManager, int diameter, out ComplexMoments res)
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

            return (double) right_answers / (double) tests;
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

        public bool Train(List<uint> layers, SortedDictionary<string, List<ComplexMoments>> moments, int maxIters = 100000, double eps = 0.1, double speed = 0.1)
        {
            ANeuralNetwork network = new ANeuralNetwork(layers, AnnRoot.ActivationType.BipolarSygmoid, 1);
            var inputs = new List<List<double>>();
            var outputs = new List<List<double>>();

            var max = int.MinValue;
            foreach (var value in moments)
            {
                if (value.Value.Count > max)
                    max = value.Value.Count;
            }

            int iter = 0;
            int counter = 0;
            while (iter < max && counter < layers[^1])
            {
                for (var i = 0; i < moments.Keys.Count; ++i)
                {
                    var output = new List<double>();
                    for (var j = 0; j < moments.Keys.Count; j++)
                    {
                        output.Add(j == i ? 1 : 0);
                    }

                    counter++;
                    if (moments[i.ToString()].Count - 1 <= iter)
                        continue;
                    var input = moments[i.ToString()][iter].ToListOfDouble();
                    inputs.Add(input);
                    outputs.Add(output);
                    counter--;
                    iter++;

                }
            }

            Console.WriteLine("Данные на входе, начинаем обучение");

            network.BackPropTraining(inputs, outputs, maxIters, eps, speed, true, 1);
            network.Save("..\\..\\..\\..\\savedData.txt");
            return true;

        }

        public void Check()
        {
            ANeuralNetwork network = new ANeuralNetwork();
            network.Load("..\\..\\..\\..\\savedData.txt");
            var sampleDirs = Directory.GetDirectories("..\\..\\..\\..\\Data\\TestData\\").ToList();
            int fCount = Directory.GetFiles("..\\..\\..\\..\\Data\\GroundData\\", "*.png", SearchOption.AllDirectories).Length;
            double precision = 0;
            var poliManager = new PolynomialManager();
            poliManager.InitBasis(10, 100);
            double count = 0;
            double trueCount = 0;
            foreach (var dir in sampleDirs)
            {
                var files = Directory.GetFiles(dir, "*.png", SearchOption.TopDirectoryOnly).ToList();
                var value = Convert.ToInt32(new StreamReader(dir + "\\" + "value.txt").ReadToEnd());

                foreach (var file in files)
                {
                    ++count;
                    ComplexMoments tmpMoments;
                    ProcessOneImage(file, poliManager, 100, out tmpMoments);
                    var tmpInput = tmpMoments.ToListOfDouble();

                    var output = network.Predict(tmpInput);
                    var predictedValue = Convert.ToInt32(output.IndexOf(output.Max()));
                    if (predictedValue == value)
                        //++trueCount;
                        precision += Convert.ToDouble(100 / (double) fCount);
                }

                Console.WriteLine("Точность " + precision + "%");
                //Console.WriteLine("Точность " + trueCount/count + "%");
            }
        }

        public void CheckOne()
        {
            ANeuralNetwork network = new ANeuralNetwork();
            network.Load("..\\..\\..\\..\\savedData.txt");
            var sampleDirs = Directory.GetDirectories("..\\..\\..\\..\\Data\\TestData\\").ToList();
            int fCount = Directory.GetFiles("..\\..\\..\\..\\Data\\GroundData\\", "*.png", SearchOption.AllDirectories).Length;
            var poliManager = new PolynomialManager();
            poliManager.InitBasis(10, 100);
            double count = 0;
            double trueCount = 0;
            foreach (var dir in sampleDirs)
            {
                var files = Directory.GetFiles(dir, "*.png", SearchOption.TopDirectoryOnly).ToList();
                var index = new Random().Next(files.Count);
                var file = files[index];
                var value = Convert.ToInt32(new StreamReader(dir + "\\" + "value.txt").ReadToEnd());

                ++count;
                ComplexMoments tmpMoments;
                ProcessOneImage(file, poliManager, 100, out tmpMoments);
                CvInvoke.Imshow(dir, tmpMoments.Real);
                var tmpInput = tmpMoments.ToListOfDouble();

                var output = network.Predict(tmpInput);
                var predictedValue = Convert.ToInt32(output.IndexOf(output.Max()));

                Console.WriteLine(predictedValue);
                CvInvoke.WaitKey();
            }
        }
    }

}

