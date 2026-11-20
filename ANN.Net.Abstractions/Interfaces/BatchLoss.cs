using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Loss of a whole batch, computed from the outputs of all its samples together, as ranking, embedding and
    /// reinforcement learning losses are. Writes the gradient of the loss with respect to the outputs of sample i to
    /// <paramref name="gradients"/>[i] (an array of the same length as <paramref name="outputs"/>[i]) and returns the loss
    /// </summary>
    public delegate Quad BatchLoss(IReadOnlyList<Quad[]> outputs, Quad[][] gradients);
}
