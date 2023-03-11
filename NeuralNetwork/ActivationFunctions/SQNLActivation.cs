using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Square Nonlinearity
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.SQNL)]
    internal class SQNLActivation : BaseActivation<SQNLActivation>, IActivationFunction
    {
        public Quad MinValue => -1;

        public Quad MaxValue => 1;

        public Quad Activation(ref Quad x)
        {
            if (x > 2)
            {
                return 1;
            }
            else if (x >= 0 && x <= 2)
            {
                return (Quad)(x - Math.Pow(x, 2) / 4);
            }
            else if (x >= -2 && x < 0)
            {
                return (Quad)(x + Math.Pow(x, 2) / 4);
            }
            else// if (x < -2)
            {
                return -1;
            }
            //throw new NotImplementedException();
        }

        public Quad Derivative(Quad x)
        {
            if (x > 2)
            {
                return 1 + x / 2;
            }
            else if (x >= 0 && x <= 2)
            {
                return 1 - x / 2;
            }
            else if (x >= -2 && x < 0)
            {
                return 1 + x / 2;
            }
            else// if (x < -2)
            {
                return 1 - x / 2;
            }
        }
    }
}
