using System.Collections.Generic;

namespace NeuralNetwork.Interfaces
{
    public interface ISynapseCollection<T> : IReadOnlyCollection<T>
    {
        ushort ActivatedCount { get; }
        bool CheckCountAndReset();
        void ResetCounter();
        void AccountSignal();
        bool Zeroed();
    }
}
