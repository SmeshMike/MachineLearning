using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using Emgu.CV;
using Emgu.CV.CvEnum;
using static FeatureExtractionLib.RadialFunctions;
using OrthoBasis = System.Collections.Generic.List<System.Collections.Generic.List<System.Tuple<Emgu.CV.Mat, Emgu.CV.Mat>>>;

namespace FeatureExtractionLib
{
    public class PolynomialManager
    {

        /** Полиномы. */
        protected OrthoBasis polynomials;


        /**
		 * Разложить картинку в ряд по полиномам.
		 * @param blob - картинка (смежная область), должна быть типа CV_8UC1.
		 * @return decomposition разложение.
		 */
        public ComplexMoments Decompose(Mat blob)
        {
            ComplexMoments result = new ComplexMoments();
            Mat tmp = blob;

            for (int i = 0; i < polynomials.Count; i++)
            for (int j = 0; j < polynomials[i].Count; j++)
            {
                CvInvoke.Multiply(polynomials[i][j].Item1, blob, tmp);
                result.image += tmp;
                CvInvoke.Multiply(polynomials[i][j].Item2, blob, tmp);
                result.real += tmp;
            }
            return result;
        }
        string GetTestString()
        {
            return "You successfuly plug feature extraction library!";
        }

        /**
         * Создать обработчик смежных областей.
         * @return обработчик смежных областей.
         */
        PolynomialManager CreateBlobProcessor()
        {
            return new PolynomialManager();
        }

        /**
         * Создать объект, ответственный за работу с полиномами.
         * @return объект, ответсвенный за работу с полиномами.
         */
        PolynomialManager CreatePolynomialManager()
        {
            return new PolynomialManager();
        }

        /**
		 * Восстановить картинку из разложения.
		 * @param decomposition - разложение картинки в ряд.
		 * @return восстановленное изображение, имеет тип CV_64FC1.
		 */
        public Mat Recovery(ComplexMoments decomposition)
        {
            Mat result = new Mat();
            Mat tmp = new Mat();
            for (int i = 0; i < polynomials.Count; i++)
            {
                for (int j = 0; j < polynomials[i].Count; j++)
                {
                    CvInvoke.Multiply(decomposition.real, polynomials[i][j].Item1, tmp);
                    result += tmp;

                    CvInvoke.Multiply(decomposition.image, polynomials[i][j].Item2, tmp);
                    result += tmp;
                }
            }
            return result;
        }

        /**
		 * Проинициализировать базис ортогональных полиномов ~ exp(jm*fi).
		 * @param n_max - максимальный радиальный порядок полиномов.
		 * @param diameter - диаметр окружности, на которой будут сгенерированы полиномы, пиксели.
		 */
        public void InitBasis(int n_max, int diameter)
        {
            polynomials.Clear();
            var psize = n_max;
            polynomials.Capacity = (n_max);
            RadialFunctions tmp;
            for (int n = 0; n < n_max; n++)
            {
                polynomials[n].Capacity = (psize);
                for (int i = 0; i < psize; i++)
                {
                    polynomials[n][i].Item1.SetTo(Mat.Zeros(diameter, diameter, DepthType.Cv64F, 0));
                    polynomials[n][i].Item2.SetTo(Mat.Zeros(diameter, diameter, DepthType.Cv64F, 0));
                    //std::vector<double> data_r,data_im;
                    for (int j = -diameter / 2; j < diameter / 2; j++)
                    {
                        for (int k = -diameter / 2; k < diameter / 2; k++)
                        {
                            var rad = Math.Sqrt(j * j + k * k);
                            if (rad > diameter / 2)
                                continue;
                            tmp = new RadialFunctions();
                            var r = tmp.Zernike(rad, n, n_max);
                            polynomials[n][i].Item1.SetValue(diameter / 2 + j, diameter / 2 + k, r * j / rad);
                            polynomials[n][i].Item2.SetValue(diameter / 2 + j, diameter / 2 + k, r * k / rad);
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
}
