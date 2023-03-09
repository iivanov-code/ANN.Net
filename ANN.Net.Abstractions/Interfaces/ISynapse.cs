namespace ANN.Net.Abstractions.Interfaces
{
    public interface ISynapse
    {
#if DEBUG
        INeuron Input { get; }
        INeuron Output { get; }
        float Weight { get; }
#endif

        void Propagate(float value);
        void UpdateWeight(float gradient);
        void Backpropagate(float error);
    }
}
