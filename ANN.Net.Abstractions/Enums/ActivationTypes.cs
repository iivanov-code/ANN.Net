using System;

namespace ANN.Net.Abstractions.Enums
{
    [Serializable]
    public enum ActivationTypes
    {
        Sigmoid = 0,
        SoftPlus = 1,
        SoftStep = 2,
        HyperbolicTangens = 3,
        ArcusTangens = 4,
        RELU = 5,
        LeakyRELU = 6,
        ParametricRELU = 7,
        SoftMax = 8,
        SWISH = 9,
        BinaryRELU = 10,
        BinaryStep = 11,
        BReLU = 12,
        Gaussian = 13,
        ISRU = 14,
        Sinusoid = 15,
        SoftSign = 16,
        SQNL = 17
    }
}
