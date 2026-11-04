namespace ANN.Net.Abstractions.Enums
{
    /// <summary>
    /// Number format of the weights of a compact network or an ONNX export
    /// </summary>
    public enum WeightPrecision
    {
        /// <summary>
        /// 32-bit floats: 4 bytes per weight, the values of the trained network
        /// </summary>
        Single = 0,

        /// <summary>
        /// 16-bit floats: 2 bytes per weight, about 3 significant digits
        /// </summary>
        Half = 1,

        /// <summary>
        /// 8-bit integers with one scale per neuron or filter (symmetric quantization): 1 byte per weight,
        /// 255 levels between -max|w| and max|w|
        /// </summary>
        Int8 = 2
    }
}
