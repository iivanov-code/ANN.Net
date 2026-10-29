using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Passes the value through unchanged; used for linear inputs and outputs
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.Identity)]
    internal class IdentityActivation : BaseActivation<IdentityActivation>, IActivationFunction
    {
        public Quad MinValue => Quad.NegativeInfinity;

        public Quad MaxValue => Quad.PositiveInfinity;

        public Quad Activation(ref Quad x)
        {
            return x;
        }

        public Quad Derivative(Quad x)
        {
            return 1;
        }

        public static Quad Activate(ref Quad x)
        {
            return x;
        }
    }
}
