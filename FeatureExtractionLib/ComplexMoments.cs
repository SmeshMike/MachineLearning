﻿using Emgu.CV;

namespace FeatureExtractionLib
{
    public class ComplexMoments
    {
        /** Реальные части. */
        public Mat Real;

        /** Мнимые части. */
        public Mat Image;

        /** Модули. */
        public Mat Abs;

        /** Фазы. */
        public Mat Phase;
    }
}
