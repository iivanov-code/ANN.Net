using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean pseudo-Huber loss δ² (√(1 + (e / δ)²) - 1), a smooth Huber loss
    /// </summary>
    internal sealed class PseudoHuberLossFunction : PointwiseLossFunction
    {
        private readonly double delta;

        public PseudoHuberLossFunction(double delta)
            : base(true)
        {
            this.delta = delta;
        }

        protected override double Value(double target, double predicted)
        {
            double scaled = (predicted - target) / delta;
            return delta * delta * (Math.Sqrt(1 + (scaled * scaled)) - 1);
        }

        protected override double Derivative(double target, double predicted)
        {
            double error = predicted - target, scaled = error / delta;
            return error / Math.Sqrt(1 + (scaled * scaled));
        }
    }
}
