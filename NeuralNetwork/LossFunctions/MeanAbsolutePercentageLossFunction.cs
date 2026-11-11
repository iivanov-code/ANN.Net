using System;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// 100 times the mean of |predicted - target| / |target|
    /// </summary>
    internal sealed class MeanAbsolutePercentageLossFunction : PointwiseLossFunction
    {
        public MeanAbsolutePercentageLossFunction()
            : base(true)
        {
        }

        protected override double Value(double target, double predicted)
        {
            return 100 * Math.Abs(predicted - target) / Math.Max(Math.Abs(target), Epsilon);
        }

        protected override double Derivative(double target, double predicted)
        {
            return 100 * Math.Sign(predicted - target) / Math.Max(Math.Abs(target), Epsilon);
        }
    }
}
