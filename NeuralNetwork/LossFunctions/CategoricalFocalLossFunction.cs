using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Focal cross entropy of a probability distribution: the sum of -(1 - predicted)^γ · target · ln(predicted)
    /// </summary>
    internal sealed class CategoricalFocalLossFunction : PointwiseLossFunction
    {
        private readonly double gamma;

        public CategoricalFocalLossFunction(double gamma)
            : base(false)
        {
            this.gamma = gamma;
        }

        protected override double Value(double target, double predicted)
        {
            double p = Math.Clamp(predicted, Epsilon, 1 - Epsilon);
            return -Math.Pow(1 - p, gamma) * target * Math.Log(p);
        }

        protected override double Derivative(double target, double predicted)
        {
            double p = Math.Clamp(predicted, Epsilon, 1 - Epsilon);
            return -target * ((Math.Pow(1 - p, gamma) / p) - (gamma * Math.Pow(1 - p, gamma - 1) * Math.Log(p)));
        }
    }
}
