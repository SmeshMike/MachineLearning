using System;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using FeatureExtractionLib;
using static FeatureExtractionLib.PolynomialManager;

namespace DecompositionSample
{
    class Program
    {
        //public static void Main(string[] args)
        //{
        //    Console.WriteLine(FeatureExtraction.GetTestString());
        //    var beg_im = CvInvoke.Imread("../../../../../myimage.png", ImreadModes.AnyColor);

        //    var bp = new Blob();
        //    var pm = new PolynomialManager();
        //    var side = 128;

        //    bp.GetType();
        //    var blobs = bp.DetectBlobs(beg_im);
        //    var counturs = bp.NormalizeBlobs(blobs, side);

        //    pm.GetType();
        //    pm.InitBasis(16, side);
        //    for (var i = 0; i < counturs.Count; i++)
        //    {
        //        var decomp = pm.Decompose(counturs[i]);
        //        var im = pm.Recovery(decomp);
        //        var t = im.ToImage<Bgr, Byte>();
        //        CvInvoke.Imshow("", t);
        //    }
        //}
        public static void Main()
        {

            Console.WriteLine("Hello world");

            int max;
            int diametr;

            Console.WriteLine( "Max polinom:");
            max = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();
            Console.WriteLine("Diametr:");
            diametr = Convert.ToInt32(Console.ReadLine()); ;
            Console.WriteLine();

            PolynomialManager pm = new PolynomialManager();

            pm.InitBasis(max, diametr);
            Visualisation.ShowPolynomials("Basic:", Polynomials);

            Mat image = CvInvoke.Imread("..\\..\\..\\..\\Picture.png", ImreadModes.Grayscale);
            CvInvoke.Imshow("Picture:", image);
            CvInvoke.WaitKey();

            List<Mat> blobs;
            //vector<cv::Mat> normalized_blobs;

            blobs = pm.DetectBlobs(image);
            //foreach (var blob in blobs)
            //{
            //    CvInvoke.Imshow("smth", blob);
            //    CvInvoke.WaitKey();
            //}
            blobs = pm.NormalizeBlobs(blobs, diametr);
            //foreach (var blob in blobs)
            //{
            //    CvInvoke.Imshow("smth", blob);
            //    CvInvoke.WaitKey();
            //}
            List<ComplexMoments> blobsDecompos = new List<ComplexMoments>(blobs.Count);
            List<Mat> recoveryBlobs = new List<Mat>(blobs.Count);

            for (int i = 0; i < recoveryBlobs.Count; i++)
            {
                recoveryBlobs[i] = Mat.Zeros(diametr, diametr, DepthType.Cv64F, 1);
            }

            for (int i = 0; i < blobs.Count; i++)
            {
                blobsDecompos.Add(pm.Decompose(blobs[i]));
                recoveryBlobs.Add(pm.Recovery(blobsDecompos[i]));
                Visualisation.ShowBlobDecomposition("Восстановленные цифры:", blobs[i], recoveryBlobs[i]);
                CvInvoke.WaitKey(0);
            }
        }
    }
}
