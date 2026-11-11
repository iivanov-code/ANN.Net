using System;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    internal class SwishActivation : BaseActivation<SwishActivation>, IActivationFunction
    {
        public Quad MinValue => float.MinValue;

        public Quad MaxValue => float.MaxValue;

        public Quad Activation(ref Quad x)
        {
            // x * sigmoid(x)
            return (Quad)(x / (1 + Math.Exp(-x)));
        }

        /// <summary>
        /// swish(x) + sigmoid(x) * (1 - swish(x))
        /// </summary>
        public Quad Derivative(Quad x)
        {
            double value = x;
            double sigmoid = 1 / (1 + Math.Exp(-value));
            double swish = value * sigmoid;
            return (Quad)(swish + (sigmoid * (1 - swish)));
        }
    }
}
