using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean smooth L1 loss: the Huber loss divided by β
    /// </summary>
    internal sealed class SmoothL1LossFunction : PointwiseLossFunction
    {
        private readonly double beta;

        public SmoothL1LossFunction(double beta)
            : base(true)
        {
            this.beta = beta;
        }

        protected override double Value(double target, double predicted)
        {
            double error = Math.Abs(predicted - target);
            return error < beta ? error * error / (2 * beta) : error - (beta / 2);
        }

        protected override double Derivative(double target, double predicted)
        {
            double error = predicted - target;
            return Math.Abs(error) < beta ? error / beta : Math.Sign(error);
        }
    }
}
