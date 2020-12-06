using System;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.CvEnum;
using FeatureExtractionLibrary;
using static FeatureExtractionLibrary.PolynomialManager;

namespace DecompositionSample
{
    class DecompositionSample
    {
        public static void Main()
        {
            Console.WriteLine("Hello world");

            Console.WriteLine("Max polinom:");
            var max = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();
            Console.WriteLine("Diametr:");
            var diameter = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine();

            var pm = new PolynomialManager();

            pm.InitBasis(max, diameter);
            Visualisation.ShowPolynomials("Basic:", Polynomials);

            var image = CvInvoke.Imread("..\\..\\..\\..\\Picture.png", ImreadModes.Grayscale);
            CvInvoke.Imshow("Picture:", image);
            CvInvoke.WaitKey();

            var blobs = pm.DetectBlobs(image);
            blobs = pm.NormalizeBlobs(blobs, diameter);

            var decomposedBlobs = new List<ComplexMoments>(blobs.Count);
            var recoveredBlobs = new List<Mat>(blobs.Count);

            for (var i = 0; i < blobs.Count; i++)
            {
                decomposedBlobs.Add(pm.Decompose(blobs[i]));
                recoveredBlobs.Add(pm.Recovery(decomposedBlobs[i]));
                Visualisation.ShowBlobDecomposition("Восстановленные цифры:", blobs[i], recoveredBlobs[i]);
                CvInvoke.WaitKey();
            }
        }
    }
}