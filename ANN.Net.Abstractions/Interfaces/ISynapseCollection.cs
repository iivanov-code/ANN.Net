using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
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
