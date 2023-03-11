using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    public class RecurrentLayerSettings : HiddenLayerSettings
    {
        public NeuronType CellType { get; set; }
        public bool HasSelfConnection { get; set; }
        public bool FullyConnectLayer { get; set; }
        public bool ActiveConnections { get; set; }
        public RecurrentDirection Direction { get; set; }

        public ushort WindowSize { get; set; }

        public ushort Features
        {
            get
            {
                return this.NeuronsCount;
            }
            set
            {
                this.NeuronsCount = value;
            }
        }
    }
}
