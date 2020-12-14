using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using FeatureExtractionLibrary;
using System.Text;
using Emgu.CV;

namespace SymbolRecognitionTrainer
{
    public class ProcessData
    {
        public bool DistributeData(string labeledDataPath, string groundDataPath, string testDataPath, double percent)
        {
            List<string> sampleDirs = Directory.GetDirectories(labeledDataPath).ToList();
            List<string> dirNames = new List<string>();
            foreach (var dir in sampleDirs)
            {
                dirNames.Add( new DirectoryInfo(dir).Name);
            }
            // Для каждой папки создадим соответсвующую в каталоге для примеров и тестовых данных.
            for (var i = 0; i < sampleDirs.Count; i++)
            {

                Directory.CreateDirectory(groundDataPath + "\\" + dirNames[i]);
                Directory.CreateDirectory(testDataPath + "\\" + dirNames[i]);

            }

            //Заглянем в каждую директорию и раскидаем файлы
            for (var i = 0; i < sampleDirs.Count; i++)
            {

                string labeledVal = labeledDataPath + "\\" + dirNames[i] + "\\" + "value.txt";
                string groundVal = groundDataPath + "\\" + dirNames[i] + "\\" + "value.txt";
                string testVal = testDataPath + "\\" + dirNames[i] + "\\" + "value.txt";
                File.Copy(labeledVal, groundVal, true);
                File.Copy(labeledVal, testVal, true);
                
                var postfix = Directory.GetFiles( sampleDirs[i], "*.png");
                foreach (var fix in postfix)
                {
                    string copyFrom = fix;
                    string copyTo;
                    if (new Random().Next(100) < percent)
                    {
                        copyTo = fix.Replace("LabeledData", "GroundData");
                        File.Copy(copyFrom, copyTo, true);
                    }
                    else
                    {
                        copyTo = fix.Replace("LabeledData", "TestData");
                        File.Copy(copyFrom, copyTo, true);
                    }
                }
            }

            return true;
        }

        public bool SaveMoments(string filename,SortedDictionary<string, List<ComplexMoments>> moments)
        {


            FileStorage fs = new FileStorage(filename, FileStorage.Mode.Write);
            if (!fs.IsOpened)
            {
                return false;
            }

            var max = int.MaxValue;
            foreach (var value in moments)
            {
                if (value.Value.Count < max)
                    max = value.Value.Count;
            }

            fs.Write("train_data", "head");
            for (int i = 0; i < max; ++i)
            {
                //fs.Write(value.Key, "value");
                foreach (var value in moments)
                {
             
                    fs.Write(value.Value[i].Real, ("re" + value.Key + i));
                    fs.Write(value.Value[i].Image, "im" + value.Key + i);
                    fs.Write(value.Value[i].Abs, "abs" + value.Key + i);
                    fs.Write(value.Value[i].Phase, "phase" + value.Key + i);
                }
            }

            fs.Write("train_data", "head");
            foreach (var value in moments)
            {
                //fs.Write(value.Key , "value");

                for (int i = max; i < value.Value.Count; ++i)
                {
                    fs.Write(value.Value[i].Real, ("re"+value.Key+i));
                    fs.Write(value.Value[i].Image, "im"+value.Key+i);
                    fs.Write(value.Value[i].Abs, "abs"+value.Key+i);
                    fs.Write(value.Value[i].Phase, "phase"+value.Key+i);
                }
            }

            fs.ReleaseAndGetString();
            return true;
        }

        public bool ReadMoments(string filename, SortedDictionary<string, List<ComplexMoments>> moments)
        {
            FileStorage fs = new FileStorage(filename, FileStorage.Mode.Read);
            if (!fs.IsOpened)
            {
                return false;
            }
            var values = new[] {"0", "1", "2", "3", "4", "5", "6", "7", "8"};
            foreach (var value in values)
            {
                int fCount = Directory.GetFiles("..\\..\\..\\..\\Data\\GroundData\\Sample00"+value, "*.png", SearchOption.TopDirectoryOnly).Length;
                var tmpMoments = new List<ComplexMoments>();
                for (int i = 0; i < fCount; ++i)
                {
                    var tmpMoment = new ComplexMoments();
                    var tmpReMat = new Mat();
                    var tmpImMat = new Mat();
                    var tmpAbsMat = new Mat();
                    var tmpPhaseMat = new Mat();
                    var tmpRe = fs.GetNode("re"+ value + i);
                    var tmpIm = fs.GetNode("im"+ value + i);
                    var tmpAbs = fs.GetNode("abs"+ value + i);
                    var tmpPhase = fs.GetNode("phase"+ value + i);
                    tmpRe.ReadMat(tmpReMat);
                    tmpIm.ReadMat(tmpImMat);
                    tmpAbs.ReadMat(tmpAbsMat);
                    tmpPhase.ReadMat(tmpPhaseMat);
                    tmpMoment.Real = tmpReMat;
                    tmpMoment.Image = tmpImMat;
                    tmpMoment.Abs = tmpAbsMat;
                    tmpMoment.Phase = tmpPhaseMat;
                    tmpMoments.Add(tmpMoment);
                }
                moments.Add(value, tmpMoments);
            }

            return true;
        }

    }
}

