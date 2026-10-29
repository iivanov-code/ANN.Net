namespace ANN.Net.Transformers
{
    /// <summary>
    /// Position-wise feed forward sub-layer: the same transformation applied to every vector of a sequence
    /// </summary>
    internal interface IFeedForwardNetwork
    {
        Quad[][] Propagate(Quad[][] inputs);

        /// <summary>
        /// Propagates the gradient of the outputs back to the inputs and updates the weights
        /// </summary>
        /// <returns>Gradient with respect to the inputs of the last propagation</returns>
        Quad[][] Backpropagate(Quad[][] outputGradients);
    }
}
