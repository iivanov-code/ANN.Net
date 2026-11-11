using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean absolute error
    /// </summary>
    internal sealed class MeanAbsoluteLossFunction : PointwiseLossFunction
    {
        public MeanAbsoluteLossFunction()
            : base(true)
        {
        }

        protected override double Value(double target, double predicted)
        {
            return Math.Abs(predicted - target);
        }

        protected override double Derivative(double target, double predicted)
        {
            return Math.Sign(predicted - target);
        }
    }
}
