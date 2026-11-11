using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
{
    public interface ITrainingStrategy
    {
        void Train(IReadOnlyCollection<Quad[]> data, ushort totalEpochs, ushort epochStep = 2);

        INetwork Network { get; }
        Rate MinValue { get; }
        Rate MaxValue { get; }
    }
}
