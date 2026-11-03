namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Layer of a <see cref="ConvolutionalNetwork"/>. Backpropagate takes the gradient of the loss with respect to the
    /// outputs of the last propagation, updates the parameters and returns the gradient with respect to the inputs,
    /// computed with the parameters before the update
    /// </summary>
    internal interface IConvolutionalLayer
    {
        Shape InputShape { get; }

        Shape OutputShape { get; }

        Quad[] Propagate(Quad[] input);

        Quad[] Backpropagate(Quad[] outputGradient);
    }
}
