using System;

namespace ANN.Net.Remote
{
    /// <summary>
    /// A part of a breadth split failed or could not be reached, so the propagation cannot complete
    /// </summary>
    public sealed class RemotePartitionException : Exception
    {
        public RemotePartitionException(int part, string message)
            : base($"Part {part}: {message}")
        {
            this.Part = part;
        }

        public RemotePartitionException(int part, string message, Exception innerException)
            : base($"Part {part}: {message}", innerException)
        {
            this.Part = part;
        }

        public int Part { get; }
    }
}
