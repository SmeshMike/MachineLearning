using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;

namespace DecompositionSample
{
    public static class Visualisation
    {
        public static void ShowPolynomials(string wndName, List<List<Tuple<Matrix<double>, Matrix<double>>>> polynomials)
        {
            int jMax = 0;
            if (polynomials.Count == 0)
            {
                throw new Exception("Nothing to show");
            }

            for (var i = 0; i < polynomials.Count; i++)
            {
                if (polynomials[i].Count == 0)
                {
                    throw new Exception("Nothing to show");
                }

                if (polynomials[i].Count > jMax) jMax = polynomials[i].Count;
                for (var j = 0; j < polynomials[i].Count; j++)
                {
                    if (polynomials[i][j].Item1.Mat.IsEmpty)
                    {
                        throw new Exception("Empty polynomial");
                    }

                    if (polynomials[i][j].Item2.Mat.IsEmpty)
                    {
                        throw new Exception("Empty polynomial");
                    }
                }
            }

            int diameter = polynomials[0][0].Item1.Cols;
            Matrix<byte> showMat = new Matrix<byte>(diameter * polynomials.Count, diameter * jMax * 2);
            Matrix<byte> bufMat = new Matrix<byte>(polynomials.Count, polynomials.Count);
            showMat.Mat.SetTo(new MCvScalar(127));
            for (var i = 0; i < polynomials.Count; i++)
            {
                for (var j = 0; j < polynomials[i].Count; j++)
                {
                    double[] min = new double[1];
                    double[] max = new double[1];
                    Point[] points = new Point[1];
                    polynomials[i][j].Item1.Mat.MinMax(out min, out max, out points, out points);
                    var alpha = 255 / (max[0] - min[0]);
                    var beta = - 255 * min[0]/ ((max[0] - min[0]));
                    polynomials[i][j].Item1.Mat.ConvertTo(bufMat, DepthType.Cv8U, alpha, beta);
                    var roi = showMat.GetSubRect(new Rectangle(2 * j * diameter, i * diameter, diameter, diameter));
                    bufMat.Mat.CopyTo(roi.Mat);
                    polynomials[i][j].Item2.Mat.ConvertTo(bufMat, DepthType.Cv8U, alpha, beta);
                    roi = showMat.GetSubRect(new Rectangle((2 * j + 1) * diameter, i * diameter, diameter, diameter));
                    bufMat.Mat.CopyTo(roi.Mat);
                }
            }
            CvInvoke.Imshow(wndName, showMat);
        }

        public static void ShowBlobDecomposition(string wndName, Mat blob, Mat decomposition)
        {
            if (blob.IsEmpty) throw new Exception("Empty blob!");
            if (decomposition.IsEmpty) throw new Exception("Empty blob!");
            if (blob.Size != decomposition.Size) throw new Exception("Incorrect size!");
            if (blob.Depth != DepthType.Cv8U) throw new Exception("Incorrect blob mat type!");
            if (decomposition.Depth != DepthType.Cv64F) throw new Exception("Incorrect decomposition mat type!");
            Mat showDecomposition = new Mat();
            double[] min = new double[1];
            double[] max = new double[1];
            Point[] points = new Point[1];
            decomposition.MinMax(out min, out max, out points, out points);
            var alpha = 255 / (max[0] - min[0]);
            var beta = -255 * min[0] / ((max[0] - min[0]));
            decomposition.ConvertTo(showDecomposition, DepthType.Cv8U, alpha, beta);
            Matrix<byte> showMat = new Matrix<byte>(blob.Rows, blob.Cols * 2);
            Matrix<byte> roi = showMat.GetSubRect(new Rectangle(0, 0, blob.Cols, blob.Rows));
            blob.CopyTo(roi);
            roi = showMat.GetSubRect(new Rectangle(blob.Cols, 0, blob.Cols, blob.Rows));
            showDecomposition.CopyTo(roi);
            CvInvoke.Imshow(wndName, showMat);
        }
    }
}
