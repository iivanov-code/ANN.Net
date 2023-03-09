using System;

namespace ANN.Net.Abstractions.Enums
{
    [Serializable]
    internal enum NodeType : byte
    {
        INPUT = 0,
        HIDDEN = 1,
        OUTPUT = 2
    }
}
