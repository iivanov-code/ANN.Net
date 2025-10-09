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
            //return (Quad)(x / (1 + Math.Exp(-x)));
            return (Quad)(x / (1 + Math.Exp(x) - x));
        }

        //Derivative of Swish is Swish(x) + sigmoid(x) * (1 - Swish(x))
        public Quad Derivative(Quad x)
        {
            //return (Quad)(Activation(ref x) + (1 / (1 + Math.Exp(-x))) * (1 - Activation(ref x)));
            return (Quad)(1 / Math.Exp(x) - 1);
        }
    }
}
