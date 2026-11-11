using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean Huber loss: squared for errors up to δ, linear beyond
    /// </summary>
    internal sealed class HuberLossFunction : PointwiseLossFunction
    {
        private readonly double delta;

        public HuberLossFunction(double delta)
            : base(true)
        {
            this.delta = delta;
        }

        protected override double Value(double target, double predicted)
        {
            double error = Math.Abs(predicted - target);
            return error <= delta ? error * error / 2 : delta * (error - (delta / 2));
        }

        protected override double Derivative(double target, double predicted)
        {
            double error = predicted - target;
            return Math.Abs(error) <= delta ? error : delta * Math.Sign(error);
        }
    }
}
