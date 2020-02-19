using NeuralNetwork.Enums;

namespace NeuralNetwork.Settings
{
    public class CellLayerSettings : HiddenLayerSettings
    {
        public CellType CellType { get; set; }

        public bool HasSelfConnection { get; set; }
        public RecurrentDirection Direction { get; set; }
    }
}
