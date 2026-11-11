using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
{
    public interface ISynapseCollection<T> : IEnumerable<T>
    {
        Quad Value { get; set; }

        void Add(T axon);

        void AddRecurrent(T axon);

        void ClearValue();
    }
}
