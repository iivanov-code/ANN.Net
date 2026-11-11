using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean of ln(cosh(predicted - target))
    /// </summary>
    internal sealed class LogCoshLossFunction : PointwiseLossFunction
    {
        public LogCoshLossFunction()
            : base(true)
        {
        }

        protected override double Value(double target, double predicted)
        {
            // ln(cosh(e)) = |e| + ln(1 + exp(-2|e|)) - ln 2, which does not overflow for large errors
            double error = Math.Abs(predicted - target);
            return error + Math.Log(1 + Math.Exp(-2 * error)) - Math.Log(2);
        }

        protected override double Derivative(double target, double predicted)
        {
            return Math.Tanh(predicted - target);
        }
    }
}
