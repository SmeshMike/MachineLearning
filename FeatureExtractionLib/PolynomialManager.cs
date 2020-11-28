using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using static FeatureExtractionLib.RadialFunctions;
using OrthoBasis = System.Collections.Generic.List<System.Collections.Generic.List<System.Tuple<Emgu.CV.Matrix<double>, Emgu.CV.Matrix<double>>>>;

namespace FeatureExtractionLib
{
    public class PolynomialManager
    {

        /** Полиномы. */
        public static OrthoBasis Polynomials { get; set; }

        delegate RadialFunctions Function();

        /**
		 * Разложить картинку в ряд по полиномам.
		 * @param blob - картинка (смежная область), должна быть типа CV_8UC1.
		 * @return decomposition разложение.
		 */

        string GetTestString()
        {
            return "You successfuly plug feature extraction library!";
        }

        ///**
        // * Создать обработчик смежных областей.
        // * @return обработчик смежных областей.
        // */
        //PolynomialManager CreateBlobProcessor()
        //{
        //    return new PolynomialManager();
        //}

        ///**
        // * Создать объект, ответственный за работу с полиномами.
        // * @return объект, ответсвенный за работу с полиномами.
        // */
        //PolynomialManager CreatePolynomialManager()
        //{
        //    return new PolynomialManager();
        //}

        /**
		 * Восстановить картинку из разложения.
		 * @param decomposition - разложение картинки в ряд.
		 * @return восстановленное изображение, имеет тип CV_64FC1.
		 */


        public List<Mat> DetectBlobs(Mat image)
        {
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat binary = new Mat(new Size(image.Rows, image.Cols), DepthType.Cv8U, 1);
            CvInvoke.Threshold(image, binary, 127, 255, ThresholdType.BinaryInv);
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(binary, contours, hierarchy, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);

            List<Mat> blums = new List<Mat>();
            for (int idx = 0; idx >= 0; idx = hierarchy.GetValue(0,idx))
            {
                var circle = CvInvoke.MinEnclosingCircle(contours[idx]);
                var center = circle.Center;
                var radius = circle.Radius;
                radius = (int)radius + 1;
                center.X = radius - (int)center.X;
                center.Y = radius - (int)center.Y;
                blums.Add(Mat.Zeros((int)(2 * radius), (int)(2 * radius), DepthType.Cv8U, 1));
                CvInvoke.DrawContours(blums.Last(), contours, idx, new MCvScalar(255, 255, 255), (int)LineType.Filled, LineType.EightConnected, hierarchy, 4, new Point((int)center.X, (int)center.Y));
            }

            return blums;
        }

        public List<Mat> NormalizeBlobs(List<Mat> blobs, int side)
        {
            List<Mat> nblums = blobs;
            for (int i = 0; i < nblums.Count; i++)
            {
                CvInvoke.Resize(blobs[i], nblums[i], new Size(side, side));
            }

            return nblums;
        }

        public ComplexMoments Decompose(Mat blob)
        {
            ComplexMoments result = new ComplexMoments
            {
                    real = Mat.Zeros(Polynomials.Count, Polynomials.Count, DepthType.Cv64F, 1),
                    image = Mat.Zeros(Polynomials.Count, Polynomials.Count, DepthType.Cv64F, 1),
                    abs = Mat.Zeros(Polynomials.Count, Polynomials.Count, DepthType.Cv64F, 1),
                    phase = Mat.Zeros(Polynomials.Count, Polynomials.Count, DepthType.Cv64F, 1)
            };
            Mat blum = new Mat();
            double[] min = new double[1];
            double[] max = new double[1];
            Point[] points = new Point[1];
            blob.MinMax(out min, out max, out points, out points);
            var alpha = 255 / (max[0] - min[0]);
            var beta = -255 * min[0] / ((max[0] - min[0]));
            blob.ConvertTo(blum, DepthType.Cv64F, alpha, beta);
            for (var n = 0; n < Polynomials.Count; n++)
                for (var m = 0; m <= n; m++)
                {
                    if ((n - m) % 2 == 0)
                    {
                        double temp = blum.Dot(Polynomials[n][m].Item1);
                        result.real.SetValue(n, m, Math.Abs(temp) > 1e-20 ? temp : 0);

                        temp = blum.Dot(Polynomials[n][m].Item2);
                        result.image.SetValue(n, m, Math.Abs(temp) > 1e-20 ? temp : 0);

                        double tmp = Math.Sqrt(result.real.GetValue(n, m) * result.real.GetValue(n, m) + result.image.GetValue(n, m) + result.image.GetValue(n, m));
                        result.abs.SetValue(n, m, tmp);
                        tmp = Math.Atan2(result.image.GetValue(n, m), result.real.GetValue(n, m));
                        result.phase.SetValue(n, m, tmp);
                    }
                }

            return result;
        }

        public Mat Recovery(ComplexMoments decomposition)
        {
            Mat Result = Mat.Zeros(Polynomials[0][0].Item1.Mat.SizeOfDimension[0], Polynomials[0][0].Item1.Mat.SizeOfDimension[1], DepthType.Cv64F, 1);
            for (var n = 0; n < Polynomials.Count; n++)
                for (var m = 0; m < n+1; m++)
                {
                    if((n-m)%2 ==0)
                    Result += Polynomials[n][m].Item1.Mat * decomposition.real.GetValue(n, m) - Polynomials[n][m].Item1.Mat * decomposition.image.GetValue(n, m);
                    //Result += Polynomials[n][m].Item1.Mat * decomposition.image.GetValue(n, m) + Polynomials[n][m].Item2.Mat * decomposition.real.GetValue(n, m);
                }

            return Result;
        }

        /**
         * Создать обработчик смежных областей.
         * @return обработчик смежных областей.
         */
        //BlobProcessor CreateBlobProcessor()
        //{
        //    return new BlobProcessor();
        //}

        /**
         * Создать объект, ответственный за работу с полиномами.
         * @return объект, ответсвенный за работу с полиномами.
         */
        PolynomialManager CreatePolynomialManager()
        {
            return new PolynomialManager();
        }

        /**
		 * Проинициализировать базис ортогональных полиномов ~ exp(jm*fi).
		 * @param n_max - максимальный радиальный порядок полиномов.
		 * @param diameter - диаметр окружности, на которой будут сгенерированы полиномы, пиксели.
		 */
        public void InitBasis(int nMax, int diameter)
        {
            RadialFunctions radialFunctions = new RadialFunctions();
            Polynomials = new OrthoBasis(nMax);
            for (var n = 0; n < nMax; ++n)
            {
                Polynomials.Add(new List<Tuple<Matrix<double>, Matrix<double>>>(nMax));

                for (var m = 0; m < n + 1; ++m)
                {
                    Polynomials[n].Add(new Tuple<Matrix<double>, Matrix<double>>(new Matrix<double>(diameter, diameter), new Matrix<double>(diameter, diameter)));

                    for (int x = 0; x < diameter; x++)
                    {
                        for (int y = 0; y < diameter; y++)
                        {
                            double l = Math.Sqrt((x - diameter / 2) * (x - diameter / 2) + (y - diameter / 2) * (y - diameter / 2)) * 2 / diameter;
                            if (l > 1.0)
                            {
                                Polynomials[n][m].Item1.Data[x, y] = 0;
                                Polynomials[n][m].Item2.Data[x, y] = 0;
                            }
                            else
                            {
                                double radialPart = radialFunctions.Zernike(l, n, m);
                                double angle = Math.Atan2(y- diameter/2, x- diameter / 2);
                                Polynomials[n][m].Item1.Data[x, y] =  (radialPart * Math.Cos(m*angle));
                                Polynomials[n][m].Item2.Data[x, y] = (radialPart * Math.Sin(m*angle));
                            }
                        }
                    }

                    var tmp = Math.Sqrt(Polynomials[n][m].Item1.Mat.Dot(Polynomials[n][m].Item1.Mat) + Polynomials[n][m].Item2.Mat.Dot(Polynomials[n][m].Item2.Mat));

                    for (int x = 0; x < diameter; x++)
                    {
                        for (int y = 0; y < diameter; y++)
                        {
                            Polynomials[n][m].Item1.Data[x, y] /= tmp;
                            Polynomials[n][m].Item2.Data[x, y] /= tmp;
                        }
                    }
                    
                }
                
            }
        }

    }

    /**
     * Получить базис ортогональных полиномов. 
     * @return базис ортогональных полиномов. Каждый полином представлен std::pair<cv::Mat, cv::Mat>.
     *		   в поле first хранится реальная часть полинома, в поле second мнимая. Каждая часть имеет тип CV_64FC1.
     */
    // OrthoBasis GetBasis();

}

