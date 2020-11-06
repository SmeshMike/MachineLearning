using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using FeatureExtractionLib;

namespace DecompositionSample
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine(FeatureExtraction.GetTestString());
            var beg_im = CvInvoke.Imread("../../../../../myimage.png", ImreadModes.AnyColor);


            var bp = new Blob();
            var pm = new PolynomialManager();

            var side = 128;

            bp.GetType();
            var blobs = bp.DetectBlobs(beg_im);
            var counturs = bp.NormalizeBlobs(blobs, side);

            pm.GetType();
            pm.InitBasis(16, side);
            for (var i = 0; i < counturs.Count; i++)
            {
                var decomp = pm.Decompose(counturs[i]);
                var im = pm.Recovery(decomp);
                var t = im.ToImage<Bgr, Byte>();
                CvInvoke.Imshow("", t);
            }
        }
    }
}
