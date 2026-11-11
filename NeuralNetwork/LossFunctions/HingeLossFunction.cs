using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean hinge loss for targets -1 and 1
    /// </summary>
    internal sealed class HingeLossFunction : PointwiseLossFunction
    {
        public HingeLossFunction()
            : base(true)
        {
        }

        protected override double Value(double target, double predicted)
        {
            return Math.Max(0, 1 - (target * predicted));
        }

        protected override double Derivative(double target, double predicted)
        {
            return target * predicted < 1 ? -target : 0;
        }
    }
}
