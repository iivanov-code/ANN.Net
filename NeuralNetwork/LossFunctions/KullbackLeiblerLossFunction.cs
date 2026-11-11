using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Kullback-Leibler divergence of the predicted from the target distribution; targets of 0 add nothing
    /// </summary>
    internal sealed class KullbackLeiblerLossFunction : PointwiseLossFunction
    {
        public KullbackLeiblerLossFunction()
            : base(false)
        {
        }

        protected override double Value(double target, double predicted)
        {
            return target > 0 ? target * Math.Log(target / Math.Max(predicted, Epsilon)) : 0;
        }

        protected override double Derivative(double target, double predicted)
        {
            return target > 0 ? -target / Math.Max(predicted, Epsilon) : 0;
        }
    }
}
