using System;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Average of every channel over all its positions
    /// </summary>
    [Serializable]
    internal sealed class GlobalAveragePoolingLayer : IConvolutionalLayer
    {
        public GlobalAveragePoolingLayer(Shape inputShape)
        {
            InputShape = inputShape;
            OutputShape = new Shape(inputShape.Channels, 1, 1);
        }

        public IConvolutionalLayer Replicate(Optimizers.ReplicaContext context) => new GlobalAveragePoolingLayer(InputShape);

        public Shape InputShape { get; }

        public Shape OutputShape { get; }

        private int Positions => InputShape.Height * InputShape.Width;

        public Quad[] Propagate(Quad[] input)
        {
            var output = new Quad[InputShape.Channels];
            for (int c = 0; c < output.Length; c++)
            {
                double sum = 0;
                for (int i = c * Positions; i < (c + 1) * Positions; i++) sum += (float)input[i];
                output[c] = (Quad)(sum / Positions);
            }

            return output;
        }

        public Quad[] Backpropagate(Quad[] outputGradient)
        {
            var inputGradient = new Quad[InputShape.Size];
            for (int c = 0; c < outputGradient.Length; c++)
            {
                Quad share = outputGradient[c] / Positions;
                for (int i = c * Positions; i < (c + 1) * Positions; i++) inputGradient[i] = share;
            }

            return inputGradient;
        }
    }
}
