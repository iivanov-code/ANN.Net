using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Abstractions.Interfaces
{
    public interface ILayer<T> : IEnumerable<T>
       where T : class, INeuron
    {
        T this[int index] { get; }
        int Count { get; }

        ILayer<TNeuron> AsLayerOf<TNeuron>() where TNeuron : class, INeuron;
    }
}
