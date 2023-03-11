using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    public class HiddenLayerSettings : CommonLayerSettings
    {
        public ushort LayersCount { get; set; }
        public LayerConnectionType PrevLayerConnectionType { get; set; }
    }
}
