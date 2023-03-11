namespace ANN.Net.Abstractions.Interfaces
{
    public interface IActivationFunction
    {
        Quad Activation(ref Quad x);

        Quad Derivative(Quad x);

        Quad MinValue { get; }
        Quad MaxValue { get; }
    }
}
