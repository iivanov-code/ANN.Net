using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean binary focal loss (Lin et al. 2017)
    /// </summary>
    internal sealed class FocalLossFunction : PointwiseLossFunction
    {
        private readonly double gamma;
        private readonly double alpha;

        public FocalLossFunction(double gamma, double alpha)
            : base(true)
        {
            this.gamma = gamma;
            this.alpha = alpha;
        }

        protected override double Value(double target, double predicted)
        {
            double p = Math.Clamp(predicted, Epsilon, 1 - Epsilon);
            return -((alpha * target * Math.Pow(1 - p, gamma) * Math.Log(p))
                     + ((1 - alpha) * (1 - target) * Math.Pow(p, gamma) * Math.Log(1 - p)));
        }

        protected override double Derivative(double target, double predicted)
        {
            double p = Math.Clamp(predicted, Epsilon, 1 - Epsilon);
            double positive = (-gamma * Math.Pow(1 - p, gamma - 1) * Math.Log(p)) + (Math.Pow(1 - p, gamma) / p);
            double negative = (gamma * Math.Pow(p, gamma - 1) * Math.Log(1 - p)) - (Math.Pow(p, gamma) / (1 - p));
            return -((alpha * target * positive) + ((1 - alpha) * (1 - target) * negative));
        }
    }
}
