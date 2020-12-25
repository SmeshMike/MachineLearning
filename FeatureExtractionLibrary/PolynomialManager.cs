using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using OrthoBasis = System.Collections.Generic.List<System.Collections.Generic.List<System.Tuple<Emgu.CV.Matrix<double>, Emgu.CV.Matrix<double>>>>;

namespace FeatureExtractionLibrary
{
    public class PolynomialManager
    {

        /** Полиномы. */
        public static OrthoBasis Polynomials { get; set; }

        public List<Mat> DetectBlobs(Mat image)
        {
            var contours = new VectorOfVectorOfPoint();
            var binary = new Mat(new Size(image.Rows, image.Cols), DepthType.Cv8U, 1);
            CvInvoke.Threshold(image, binary, 127, 255, ThresholdType.BinaryInv);
            var hierarchy = new Mat();
            CvInvoke.FindContours(binary, contours, hierarchy, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);

            var tmpBlob = new List<Mat>();
            for (int idx = 0; idx >= 0; idx = hierarchy.GetValue(0, idx))
            {
                var circle = CvInvoke.MinEnclosingCircle(contours[idx]);
                var center = circle.Center;
                var radius = circle.Radius;
                radius = (int) radius + 1;
                center.X = radius - (int) center.X;
                center.Y = radius - (int) center.Y;
                tmpBlob.Add(Mat.Zeros((int) (2 * radius), (int) (2 * radius), DepthType.Cv8U, 1));
                CvInvoke.DrawContours(tmpBlob.Last(), contours, idx, new MCvScalar(255, 255, 255), (int) LineType.Filled,
                        LineType.EightConnected, hierarchy, 4, new Point((int) center.X, (int) center.Y));
            }

            return tmpBlob;
        }

        public List<Mat> NormalizeBlobs(List<Mat> blobs, int side)
        {
            List<Mat> tmpBlobs = blobs;
            for (var i = 0; i < tmpBlobs.Count; i++)
            {
                CvInvoke.Resize(blobs[i], tmpBlobs[i], new Size(side, side));
            }
            return tmpBlobs;
        }

        public ComplexMoments Decompose(Mat blob)
        {
            var result = new ComplexMoments
            {
                    Real = Mat.Zeros(Polynomials.Count, Polynomials.Count, DepthType.Cv64F, 1),
                    Image = Mat.Zeros(Polynomials.Count, Polynomials.Count, DepthType.Cv64F, 1),
                    Abs = Mat.Zeros(Polynomials.Count, Polynomials.Count, DepthType.Cv64F, 1),
                    Phase = Mat.Zeros(Polynomials.Count, Polynomials.Count, DepthType.Cv64F, 1)
            };
            var tmpBlob = new Mat();
            _ = new Point[1];
            blob.MinMax(out var min, out var max, out _, out _);
            var alpha = 255 / (max[0] - min[0]);
            var beta = -255 * min[0] / ((max[0] - min[0]));
            blob.ConvertTo(tmpBlob, DepthType.Cv64F, alpha, beta);
            double tmpRealDenominator = 0;
            double tmpImageDenominator = 0;
            double tmpAbsDenominator = 0;
            for (var n = 0; n < Polynomials.Count; ++n)
            for (var m = 0; m < n+1; ++m)
            {
                if ((n - m) % 2 != 0) continue;
                var tmpMoment = tmpBlob.Dot(Polynomials[n][m].Item1);
                result.Real.SetValue(n, m, Math.Abs(tmpMoment) > 1e-20 ? tmpMoment : 0);
                tmpMoment = tmpBlob.Dot(Polynomials[n][m].Item2);
                result.Image.SetValue(n, m, Math.Abs(tmpMoment) > 1e-20 ? tmpMoment : 0);
                var tmpReal = result.Real.GetValue(n, m);
                var tmpImage = result.Image.GetValue(n, m);
                tmpRealDenominator += tmpReal* tmpReal;
                tmpImageDenominator += tmpImage * tmpImage;
                double tmpAbs = Math.Sqrt(tmpReal * tmpReal + tmpImage * tmpImage);
                tmpAbsDenominator += tmpAbs * tmpAbs;
                result.Abs.SetValue(m, n, tmpAbs);
                double tmpAngle = Math.Atan2(tmpImage, tmpReal);
                result.Phase.SetValue(m, n, tmpAngle);
            }
            //for (var n = 0; n < Polynomials.Count; ++n)
            //    for (var m = 0; m < n + 1; ++m)
            //    {
            //        if ((n - m) % 2 != 0) continue;
            //        double tmpMoment = result.Real.GetValue(n, m) / Math.Sqrt(tmpRealDenominator);
            //        result.Real.SetValue(n, m, tmpMoment);
            //        tmpMoment = result.Image.GetValue(n, m) / Math.Sqrt(tmpImageDenominator);
            //        result.Image.SetValue(n, m, tmpMoment);
            //        tmpMoment = result.Abs.GetValue(n, m) / Math.Sqrt(tmpAbsDenominator);
            //        result.Abs.SetValue(m, n, tmpMoment);
            //    }
            return result;
        }

        public Mat Recovery(ComplexMoments decomposition)
        {
            var result = Mat.Zeros(Polynomials[0][0].Item1.Mat.Rows, Polynomials[0][0].Item1.Mat.Cols, DepthType.Cv64F, 1);
            for (var n = 0; n < Polynomials.Count; ++n)
            for (var m = 0; m < n + 1; ++m)
            {
                if ((n - m) % 2 == 0)
                    result += Polynomials[n][m].Item1.Mat * decomposition.Real.GetValue(n, m) + Polynomials[n][m].Item2.Mat * decomposition.Image.GetValue(n, m);
            }

            return result;
        }


        public void InitBasis(int nMax, int diameter)
        {
            RadialFunctions radialFunctions = new RadialFunctions();
            Polynomials = new OrthoBasis(nMax);
            for (var n = 0; n < nMax; ++n)
            {
                Polynomials.Add(new List<Tuple<Matrix<double>, Matrix<double>>>(n+1));

                for (var m = 0; m < n + 1; ++m)
                {
                    Polynomials[n].Add(new Tuple<Matrix<double>, Matrix<double>>(new Matrix<double>(diameter, diameter), new Matrix<double>(diameter, diameter)));
                    if ((n - m) % 2 != 0)
                        continue;
                    for (var x = 0; x < diameter; ++x)
                    {
                        for (var y = 0; y < diameter; ++y)
                        {
                            var l = Math.Sqrt((x - diameter / 2) * (x - diameter / 2) + (y - diameter / 2) * (y - diameter / 2)) * 2 / diameter;
                            var angle = Math.Atan2(-(y - diameter / 2), x - diameter / 2);
                            if (l > 1.0 || (n - m) % 2 != 0)
                            {
                                Polynomials[n][m].Item1.Data[x, y] = 0;
                                Polynomials[n][m].Item2.Data[x, y] = 0;
                            }
                            else
                            {
                                var radialPart = radialFunctions.Zernike(l, n, m);
                                Polynomials[n][m].Item1.Data[x, y] = radialPart * Math.Cos(m * angle);
                                Polynomials[n][m].Item2.Data[x, y] = radialPart * Math.Sin(m * angle);
                            }
                        }
                    }

                    var tmp = Math.Sqrt(Polynomials[n][m].Item1.Mat.Dot(Polynomials[n][m].Item1.Mat) + Polynomials[n][m].Item2.Mat.Dot(Polynomials[n][m].Item2.Mat));

                    for (var x = 0; x < diameter; x++)
                    {
                        for (var y = 0; y < diameter; y++)
                        {
                            Polynomials[n][m].Item1.Data[x, y] /= tmp;
                            Polynomials[n][m].Item2.Data[x, y] /= tmp;
                        }
                    }
                }
            }
        }
    }
}

