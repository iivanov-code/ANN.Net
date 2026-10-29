namespace ANN.Net.Abstractions.Interfaces
{
    public interface IActivationFunction
    {
        Quad Activation(ref Quad x);

        /// <summary>
        /// Derivative of the activation at x, where x is the value before activation
        /// </summary>
        Quad Derivative(Quad x);

        Quad MinValue { get; }
        Quad MaxValue { get; }
    }
}
