using System;

namespace NeuralNetwork.Enums
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
        SWISH = 9
    }
}
