using System;

namespace ANN.Net.Abstractions.Enums
{
    [Serializable]
    public enum NodeType : byte
    {
        INPUT = 0,
        HIDDEN = 1,
        OUTPUT = 2,
        BIAS = 3
    }
}
