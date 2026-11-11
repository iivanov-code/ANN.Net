using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean of (ln(1 + predicted) - ln(1 + target))²
    /// </summary>
    internal sealed class MeanSquaredLogarithmicLossFunction : PointwiseLossFunction
    {
        public MeanSquaredLogarithmicLossFunction()
            : base(true)
        {
        }

        /// <summary>
        /// Keeps 1 + x above 0
        /// </summary>
        private static double Lift(double x) => Math.Max(x, Epsilon - 1);

        protected override double Value(double target, double predicted)
        {
            double difference = Math.Log(1 + Lift(predicted)) - Math.Log(1 + Lift(target));
            return difference * difference;
        }

        protected override double Derivative(double target, double predicted)
        {
            double lifted = Lift(predicted);
            return 2 * (Math.Log(1 + lifted) - Math.Log(1 + Lift(target))) / (1 + lifted);
        }
    }
}
