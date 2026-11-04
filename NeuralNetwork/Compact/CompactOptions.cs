using ANN.Net.Abstractions.Enums;

namespace ANN.Net
{
    /// <summary>
    /// How <see cref="NetworkCompactor"/> stores the weights of a compact network
    /// </summary>
    public sealed class CompactOptions
    {
        /// <summary>
        /// Number format of the weight matrices. Biases, normalization and batch normalization values stay 32-bit
        /// </summary>
        public WeightPrecision Precision { get; set; } = WeightPrecision.Single;
    }
}
