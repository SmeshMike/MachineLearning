using System;
using System.Collections.Generic;
using System.Text;
using Emgu.CV;
using OrthoBasis = System.Collections.Generic.List<System.Collections.Generic.List<System.Tuple<Emgu.CV.Mat, Emgu.CV.Mat>>>;

namespace FeatureExtractionLib
{
    public class ComplexMoments
    {
        /** Реальные части. */
        public Mat real;

        /** Мнимые части. */
        public Mat image;

        /** Модули. */
        public Mat abs;

        /** Фазы. */
        public Mat phase;
    }
}
