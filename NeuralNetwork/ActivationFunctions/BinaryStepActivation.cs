using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.BinaryStep)]
    internal class BinaryStepActivation : BaseActivation<BinaryStepActivation>, IActivationFunction
    {
        public Quad MinValue => 0;

        public Quad MaxValue => 1;

        public Quad Activation(ref Quad x)
        {
            return x < 0 ? 0 : 1;
        }

        public Quad Derivative(Quad x)
        {
            return x != 0 ? 0 : throw new ArgumentException();
        }
    }
}
