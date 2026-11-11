using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean squared hinge loss for targets -1 and 1
    /// </summary>
    internal sealed class SquaredHingeLossFunction : PointwiseLossFunction
    {
        public SquaredHingeLossFunction()
            : base(true)
        {
        }

        protected override double Value(double target, double predicted)
        {
            double margin = Math.Max(0, 1 - (target * predicted));
            return margin * margin;
        }

        protected override double Derivative(double target, double predicted)
        {
            return -2 * target * Math.Max(0, 1 - (target * predicted));
        }
    }
}
