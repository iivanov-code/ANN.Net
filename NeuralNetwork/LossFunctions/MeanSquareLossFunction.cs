
namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Mean squared error
    /// </summary>
    internal sealed class MeanSquareLossFunction : PointwiseLossFunction
    {
        public MeanSquareLossFunction()
            : base(true)
        {
        }

        protected override double Value(double target, double predicted)
        {
            double error = predicted - target;
            return error * error;
        }

        protected override double Derivative(double target, double predicted)
        {
            return 2 * (predicted - target);
        }
    }
}
