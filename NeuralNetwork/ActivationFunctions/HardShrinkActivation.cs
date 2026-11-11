using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.HardShrink)]
    internal class HardShrinkActivation : BaseActivation<HardShrinkActivation>, IActivationFunction
    {
        public Quad MinValue => Quad.MinValue;

        public Quad MaxValue => Quad.MaxValue;

        private Quad alpha;

        public HardShrinkActivation(Quad alpha)
        {
            this.alpha = alpha;
        }

        public Quad Activation(ref Quad x)
        {
            if (x > alpha)
            {
                return x;
            }
            else if (x < -alpha)
            {
                return x;
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
