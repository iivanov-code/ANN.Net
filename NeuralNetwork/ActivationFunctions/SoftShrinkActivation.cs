using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.SoftShrink)]
    internal class SoftShrinkActivation : BaseActivation<SoftShrinkActivation>, IActivationFunction
    {
        public Quad MinValue => -1;

        public Quad MaxValue => 1;

        private Quad alpha;

        public SoftShrinkActivation(Quad alpha)
        {
            this.alpha = alpha;
        }

        public Quad Activation(ref Quad x)
        {
            if (x > alpha)
            {
                return x - alpha;
            }
            else if (x < -alpha)
            {
                return x + alpha;
            }
            else
            {
                return 0;
            }
        }

        public Quad Derivative(Quad x)
        {
            return 1;
        }
    }
}
