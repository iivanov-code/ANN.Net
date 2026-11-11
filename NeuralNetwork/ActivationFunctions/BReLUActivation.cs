using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Bipolar Rectified Linear Unit
    /// </summary>
    ///
    [Serializable]
    [ActivationType(Type = ActivationTypes.BReLU)]
    internal class BReLUActivation : BaseActivation<BReLUActivation>, IActivationFunction
    {
        public Quad MinValue => Quad.NegativeInfinity;
        public Quad MaxValue => Quad.PositiveInfinity;

        private int i = 0;

        public Quad Activation(ref Quad x)
        {
            if (i % 2 == 0)
            {
                i++;
                return RELUActivation.Activate(ref x);
            }
            else
            {
                i++;
                x = -x;
                return RELUActivation.Activate(ref x);
            }
        }

        public Quad Derivative(Quad x)
        {
            if (i % 2 == 0)
            {
                return RELUActivation.Activate(ref x);
            }
            else
            {
                x = -x;
                return RELUActivation.Activate(ref x);
            }
        }
    }
}
