using System;

namespace ANN.Net.Abstractions.Enums
{
    [Flags()]
    public enum RecurrentDirection : byte
    {
        Forward = 1,
        Backward = 2,
        Both = 3,
    }
}
