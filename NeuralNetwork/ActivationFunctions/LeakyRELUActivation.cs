using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.LeakyRELU)]
    internal class LeakyRELUActivation : BaseActivation<LeakyRELUActivation>, IActivationFunction
    {
        public Quad MinValue => Quad.NegativeInfinity;

        public Quad MaxValue => Quad.PositiveInfinity;

        public Quad Activation(ref Quad x)
        {
            if (x < 0)
            {
                return 0.01f * x;
            }
            else
            {
                return x;
            }
        }

        public Quad Derivative(Quad x)
        {
            if (x < 0)
            {
                return 0.01f;
            }
            else
            {
                return 1f;
            }
        }
    }
}
