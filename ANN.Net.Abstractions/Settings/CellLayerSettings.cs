using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    public class CellLayerSettings : HiddenLayerSettings
    {
        public CellType CellType { get; set; }

        public bool HasSelfConnection { get; set; }
        public RecurrentDirection Direction { get; set; }
    }
}
