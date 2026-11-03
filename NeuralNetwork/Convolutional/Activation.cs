using System;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Utils;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Activation of a whole layer; remembers the values before activation for the derivative
    /// </summary>
    [Serializable]
    internal sealed class Activation
    {
        private readonly IActivationFunction function;
        private Quad[] preActivations;

        public Activation(ActivationTypes type)
        {
            Type = type;
            function = NetworkUtils.GetActivation(type);
        }

        public ActivationTypes Type { get; }

        public Quad[] Apply(Quad[] values)
        {
            preActivations = values;
            var outputs = new Quad[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                Quad value = values[i];
                outputs[i] = function.Activation(ref value);
            }

            return outputs;
        }

        /// <summary>
        /// Gradient with respect to the values before activation
        /// </summary>
        public Quad[] Backpropagate(Quad[] outputGradient)
        {
            var gradient = new Quad[outputGradient.Length];
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] = outputGradient[i] * function.Derivative(preActivations[i]);
            }

            return gradient;
        }
    }
}
