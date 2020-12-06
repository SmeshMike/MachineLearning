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

        SortedDictionary<string, List<ComplexMoments>> GenerateMoments(string path, PolynomialManager polyManager)
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
                
                foreach (var file in files)
                {
                    //Считываем картинку.
                    var moment = new ComplexMoments();
                    //Обрабатываем.
                    ProcessOneImage(file, polyManager, out moment);
                    //Сохраняем
                    result.Add(key, moment);
                }
            }

            return result;
        }

        void ProcessOneImage(string imagePath,PolynomialManager polyManager, out ComplexMoments res)
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
            var nblobs = polyManager.NormalizeBlobs(blobs, (Polynomials[0][0].Item1.Cols));
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

        public bool Train(List<int> layers, int max_iters = 100000, double eps = 0.1, double speed = 0.1)
        {

            ANeuralNetwork network = new ANeuralNetwork();
            var pm = new PolynomialManager();
            var pd = new ProcessData();
            pm.InitBasis(15, 100); 
            pd.DistributeData("..\\..\\..\\..\\Data\\LabeledData\\", "..\\..\\..\\..\\Data\\GroundData\\", "..\\..\\..\\..\\Data\\TestData\\", 50);
            var dictionary = GenerateMoments("..\\..\\..\\..\\Data\\GroundData\\", pm);
            pd.SaveMoments("..\\..\\..\\..\\Moment.txt", dictionary);
            var tmp = new SortedDictionary<string, List<ComplexMoments>>();
            pd.ReadMoments("..\\..\\..\\..\\Moment.txt", tmp);

            return true;

        }

	}
}
