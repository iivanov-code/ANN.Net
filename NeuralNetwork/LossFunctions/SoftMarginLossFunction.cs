using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean of ln(1 + exp(-target · predicted)) for targets -1 and 1
    /// </summary>
    internal sealed class SoftMarginLossFunction : PointwiseLossFunction
    {
        public SoftMarginLossFunction()
            : base(true)
        {
        }

        /// <summary>
        /// ln(1 + exp(x)) without overflow
        /// </summary>
        private static double Softplus(double x) => x > 0 ? x + Math.Log(1 + Math.Exp(-x)) : Math.Log(1 + Math.Exp(x));

        protected override double Value(double target, double predicted)
        {
            return Softplus(-target * predicted);
        }

        protected override double Derivative(double target, double predicted)
        {
            double margin = target * predicted;
            return -target / (1 + Math.Exp(margin));
        }
    }
}
