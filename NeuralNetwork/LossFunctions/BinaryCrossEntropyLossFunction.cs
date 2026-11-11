using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean binary cross entropy of independent probabilities
    /// </summary>
    internal sealed class BinaryCrossEntropyLossFunction : PointwiseLossFunction
    {
        public BinaryCrossEntropyLossFunction()
            : base(true)
        {
        }

        private static double Clamp(double p) => Math.Clamp(p, Epsilon, 1 - Epsilon);

        protected override double Value(double target, double predicted)
        {
            double p = Clamp(predicted);
            return -((target * Math.Log(p)) + ((1 - target) * Math.Log(1 - p)));
        }

        protected override double Derivative(double target, double predicted)
        {
            double p = Clamp(predicted);
            return (p - target) / (p * (1 - p));
        }
    }
}
