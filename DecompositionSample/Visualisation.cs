using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace DecompositionSample
{
    public static class Visualisation
    {
        public static void ShowPolynomials(string wndName, List<List<Tuple<Matrix<double>, Matrix<double>>>> polynomials)
        {
            if (polynomials.Count == 0)
            {
                throw new Exception("Nothing to show");
            }

            foreach (var polynomial in polynomials)
            {
                if (polynomial.Count == 0)
                {
                    throw new Exception("Nothing to show");
                }

                foreach (var polynomialElement in polynomial)
                {
                    if (polynomialElement.Item1.Mat.IsEmpty)
                    {
                        throw new Exception("Empty polynomial");
                    }

                    if (polynomialElement.Item2.Mat.IsEmpty)
                    {
                        throw new Exception("Empty polynomial");
                    }
                }
            }

            int diameter = polynomials[0][0].Item1.Cols;
            Matrix<byte> showMat = new Matrix<byte>(diameter * polynomials.Count,  diameter * polynomials.Count);
            Matrix<byte> bufMat = new Matrix<byte>(diameter, diameter);
            showMat.Mat.SetTo(new MCvScalar(127));
            for (var i = 0; i < polynomials.Count; i++)
            {
                var shift1 = 0;
                var shift2 = 0;
                
                for (var j = 0; j < polynomials[i].Count; j++)
                {
                    if (j % 2 != 0)
                    {
                        shift1++;
                        if (i % 2 == 0 && i > 0)
                            shift2 = 1;
                    }

                    if ((i % 2 != 0 && j % 2 == 0) || (i % 2 == 0 && j % 2 != 0))
                        continue;
                    
                    polynomials[i][j].Item1.Mat.MinMax(out var min, out var max, out _, out _);
                    var alpha = 255 / (max[0] - min[0]);
                    var beta = -255 * min[0] / ((max[0] - min[0]));

                    polynomials[i][j].Item1.Mat.ConvertTo(bufMat, DepthType.Cv8U, alpha, beta);
                    var roi = showMat.GetSubRect(new Rectangle((2*j -shift1*2 - shift2) * diameter, i * diameter, diameter, diameter));
                    bufMat.Mat.CopyTo(roi.Mat);

                    polynomials[i][j].Item2.Mat.ConvertTo(bufMat, DepthType.Cv8U, alpha, beta);
                    roi = showMat.GetSubRect(new Rectangle((2*j + 1 - shift1*2 - shift2) * diameter, i * diameter, diameter, diameter));
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
            decomposition.MinMax(out var min, out var max, out _, out _);
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
