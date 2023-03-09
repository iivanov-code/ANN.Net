namespace ANN.Net.Abstractions.Interfaces
{
    public interface INetwork
    {
        float[] Propagate(float[] values);
        float Backpropagate(float[] targets);
        float[] Values { get; }
        float GetNormalizedValue(float value);
    }
}