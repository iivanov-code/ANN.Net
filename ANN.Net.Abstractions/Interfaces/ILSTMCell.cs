namespace ANN.Net.Abstractions.Interfaces
{
    public interface ILSTMCell : IHiddenNeuron
    {
        void ForgetGate(float value);
        void InputGate(float value);
        void OutputGate(float value);
    }
}
