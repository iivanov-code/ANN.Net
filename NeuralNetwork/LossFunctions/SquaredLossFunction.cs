
namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Half the sum of the squared errors; its gradient is predicted - target
    /// </summary>
    internal sealed class SquaredLossFunction : PointwiseLossFunction
    {
        public SquaredLossFunction()
            : base(false)
        {
        }

        protected override double Value(double target, double predicted)
        {
            double error = predicted - target;
            return error * error / 2;
        }

        protected override double Derivative(double target, double predicted)
        {
            return predicted - target;
        }
    }
}
