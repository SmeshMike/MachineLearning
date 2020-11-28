using System;
using System.Drawing;
using System.Windows;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace FeatureExtractionLib
{   public class RadialFunctions
    {

		/* @brief взять факториал
        * @param n		- число фактрориал которго необходимо вычислить
        * @return		- значение факториала
        */
        double Factorial(int n)
        {
            static int ProdTree(int l, int r)
            {
                if (l > r)
                    return 1;
                if (l == r)
                    return l;
                if (r - l == 1)
                    return l * r;
                int m = (l + r) / 2;
                return ProdTree(l, m) * ProdTree(m + 1, r);
            }

            static int FactTree(int n)
            {
                if (n < 0)
                    return 0;
                if (n == 0)
                    return 1;
                if (n == 1 || n == 2)
                    return n;
                return ProdTree(2, n);
            }

            return FactTree(n);
        }

        public double Zernike(double rad, int n, int m)
        {
            int absM = Math.Abs(m);
            //проверим корректность принятых данных
            if (n < 0 || absM > n || rad > 1 || rad < 0) return 0;

            //не нулю равны только те полиномы, m-n у которых четное
            if ((n - absM) % 2 != 0) return 0;

            if (n == 0) return Math.Sqrt(Math.Sqrt(Math.PI));

            int sign = -1;
            double R = 0;
            //суммирование полинома
            for (int k = 0; k <= (n - absM) / 2; k++)
            {
                sign *= -1;
                R += sign * Factorial(n - k) * Math.Pow(rad, n - 2 * k) / (Factorial(k) * Factorial((n + m) / 2 - k) * Factorial((n - m) / 2 - k));
            }
            return R * Math.Sqrt(2 * n + 2);
        }

        //		Mat WalshGenerator(Mat walsh, int n)
        //		{

        //			Mat new_walsh = Mat.Zeros(walsh.Rows* 2, walsh.Cols* 2, DepthType.Cv8S, 0);
        //			Mat roi;

        //			roi = new_walsh((0, 0, walsh.Cols, walsh.Rows));
        //			walsh.CopyTo(roi);
        //			roi = new_walsh((walsh.Cols, 0, walsh.Cols, walsh.Rows));
        //			walsh.CopyTo(roi);
        //			roi = new_walsh((0, walsh.Rows, walsh.Cols, walsh.Rows));
        //			walsh.CopyTo(roi);
        //			roi = new_walsh((walsh.Cols, walsh.Rows, walsh.Cols, walsh.Rows));
        //			var data = walsh.GetDataPointer();
        //			for (int i = 0; i < walsh.Cols * walsh.Rows; i++)
        //			{
        //				data[i] *= -1;
        //			}
        //			walsh.copyTo(roi);
        //			return new_walsh.rows > n ? new_walsh : WalshGenerator(new_walsh, n);
        //		}

        //		double rf::RadialFunctions::ShiftedLegendre(double rad, int n)
        //		{
        //			if (rad < 0. || rad > 1.0) return 0;
        //			return sqrt((2. * n + 1.) / 2.) * Legendre((rad - 0.5) * 2, n);
        //		}

        //		double rf::RadialFunctions::ShiftedChebyshev(double rad, int n)
        //		{
        //			if (rad < 0. || rad > 1.0) return 0;
        //			double eps = 0.02;
        //			if (abs(rad - 1.0) < eps) rad = 1 - eps;
        //			if (rad < eps) rad = eps;
        //			double x = (rad - 0.5) * 2;
        //			double k = n == 1 ? SQRT_2 : 1.;
        //			return n == 0 ? pow(1 - x * x, 0.25) : cos(n * acos(x)) * pow(1 - x * x, 0.25) * SQRT_2 * k;
        //		}

        //		double rf::RadialFunctions::Walsh(double rad, int n, int n_max)
        //		{
        //			if (rad < 0. || rad > 1.0) return 0;
        //			cv::Mat w;
        //			if (walsh_n_max != n_max)
        //			{
        //				w = WalshGenerator(cv::Mat(1, 1, CV_8SC1, cv::Scalar::all(1)), n_max);
        //				walsh_matrix = w;
        //				walsh_n_max = n_max;
        //			}
        //			else
        //			{
        //				w = walsh_matrix;
        //			}
        //			char a = (w.ptr<char>(n)[(int)(rad * w.cols)]);
        //			return a > 0 ? 1. : -1.;
        //			return 0.;
        //		}

        //		double rf::RadialFunctions::Legendre(double x, int n)
        //		{
        //			if (n == 0) return 1;
        //			if (n == 1) return x;
        //			double l2 = 1., l1 = x, l0;
        //			for (int _n = 2; _n < n + 1; _n++)
        //			{
        //				l0 = (2. * _n + 1) / (_n + 1.) * x * l1 - _n / (_n + 1.) * l2;
        //				l2 = l1;
        //				l1 = l0;
        //			}
        //			return l0;
        //		}

        //		rf::RadialFunctions::~RadialFunctions() = default;

        //cv::Mat rf::RadialFunctions::walsh_matrix;
        //		int rf::RadialFunctions::walsh_n_max = 0;

    }
}
