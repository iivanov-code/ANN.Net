using System;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Whether a convolutional network is training, shared with the layers that behave differently then
    /// </summary>
    [Serializable]
    internal sealed class NetworkMode
    {
        public bool Training { get; set; } = true;
    }
}
