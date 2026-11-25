using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Heaviside step: 0 below 0, 1 from 0 on. Used for both <see cref="ActivationTypes.BinaryStep"/> and
    /// <see cref="ActivationTypes.BinaryRELU"/>
    /// </summary>
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

        /// <summary>
        /// 0 everywhere: the step is flat on both sides, and at 0, where it has no derivative, 0 is taken as well so
        /// that training does not stop on a sum of exactly 0
        /// </summary>
        public Quad Derivative(Quad x)
        {
            return 0;
        }
    }
}
