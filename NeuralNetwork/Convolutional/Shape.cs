using System;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Size of a channels x height x width volume stored as a flat array in channel, row, column order
    /// </summary>
    [Serializable]
    internal readonly record struct Shape(int Channels, int Height, int Width)
    {
        public int Size => Channels * Height * Width;

        public int Index(int channel, int row, int column) => (((channel * Height) + row) * Width) + column;

        public override string ToString() => $"{Channels}x{Height}x{Width}";
    }
}
