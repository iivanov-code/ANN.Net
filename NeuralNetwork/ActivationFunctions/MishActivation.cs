using System;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    internal class MishActivation : BaseActivation<MishActivation>, IActivationFunction
    {
        public Quad MinValue => -1;

        public Quad MaxValue => float.MaxValue;
        public Quad alpha = 1;

        public MishActivation(Quad? alpha = default)
        {
            this.alpha = alpha.HasValue ? alpha.Value : 1;
        }

        public Quad Activation(ref Quad x)
        {
            return x * (Quad)Math.Tanh(Math.Log((1 / alpha) + Math.Exp(alpha * x)));
        }

        public Quad Derivative(Quad x)
        {
            //Not sure if this is correct
            return (Quad)Math.Tanh(Math.Log(Math.Exp(alpha * x)));
        }
    }
}
