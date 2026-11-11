using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Cross entropy of probability outputs: the sum of -target · ln(predicted)
    /// </summary>
    internal sealed class CrossEntropyLossFunction : PointwiseLossFunction
    {
        public CrossEntropyLossFunction()
            : base(false)
        {
        }

        protected override double Value(double target, double predicted)
        {
            return -target * Math.Log(Math.Max(predicted, Epsilon));
        }

        protected override double Derivative(double target, double predicted)
        {
            return -target / Math.Max(predicted, Epsilon);
        }
    }
}
