using Emgu.CV;

namespace FeatureExtractionLibrary
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
