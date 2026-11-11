using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean Poisson negative log-likelihood without its constant: predicted - target · ln(predicted)
    /// </summary>
    internal sealed class PoissonLossFunction : PointwiseLossFunction
    {
        public PoissonLossFunction()
            : base(true)
        {
        }

        protected override double Value(double target, double predicted)
        {
            return predicted - (target * Math.Log(Math.Max(predicted, Epsilon)));
        }

        protected override double Derivative(double target, double predicted)
        {
            return 1 - (target / Math.Max(predicted, Epsilon));
        }
    }
}
