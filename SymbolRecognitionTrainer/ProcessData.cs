using System;
using System.Collections.Generic;
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
                string findPath = labeledDataPath + "\\" + dirNames[i];
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

            fs.Write("train_data [");
            foreach (var value in moments)
            {
                fs.Write("{ value " + value.Key + " moments [");
                foreach (var moment in value.Value)
                {
                    fs.Write("{ re " + moment.Real + " im " + moment.Image + " abs " + moment.Abs + " phase " + moment.Phase + " }");
                }
                fs.Write(" ]" + " }");
            }
            fs.Write("]");
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

            return true;
        }

    }
}

