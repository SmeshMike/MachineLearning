using System;
using System.Collections.Generic;
using System.Text;

namespace FeatureExtractionLib
{
    public class FeatureExtraction
    {
        public static string GetTestString()
        {
            return "You successfuly plug feature extraction library!";
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
    }
}
