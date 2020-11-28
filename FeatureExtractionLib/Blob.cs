using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FeatureExtractionLib
{
    public class Blob
    {
        string GetType()
        {
            return "NV Iblob chek";
        }

        public List<Mat> DetectBlobs(Mat image)
        {
            List<Mat> result = new List<Mat>();
            VectorOfVectorOfPointF counturs = new VectorOfVectorOfPointF();
            
            CvInvoke.FindContours(image, counturs, null ,  RetrType.Tree, ChainApproxMethod.ChainApproxNone,  new Point(15, 15));
            result.Capacity = (counturs.Size);
            for (int i = 0; i < counturs.Size; i++)
            {
                var elipse= PointCollection.EllipseLeastSquareFitting(counturs[i].ToArray());
                var center = elipse.RotatedRect.Center;
                int h = Convert.ToInt32(elipse.RotatedRect.Size.Height);
                int w = Convert.ToInt32(elipse.RotatedRect.Size.Width);
                result[i] = new Mat(h, w, DepthType.Cv8U, 0);
                CvInvoke.DrawContours(result[i], counturs,i, new MCvScalar(256,256,256),1, LineType.EightConnected,new Mat(), 2147483647, new Point(Convert.ToInt32(center.X), Convert.ToInt32(center.Y)));
            }
            return result;
        }

       public List<Mat> NormalizeBlobs(List<Mat> blobs, int side)
        {
            List<Mat> result = new List<Mat>(blobs.Count);
            for (int i = 0; i < blobs.Count; i++)
                CvInvoke.Resize(blobs[i], result[i], new Size(side, side));
            return result;
        }
	}
}
