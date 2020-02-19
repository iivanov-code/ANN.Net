namespace NeuralNetwork.Interfaces
{
    public interface IActivationFunction
    {
        float Activation(float x);
        float Prime(float x);
        float MinValue { get; }
        float MaxValue { get; }
    }
}