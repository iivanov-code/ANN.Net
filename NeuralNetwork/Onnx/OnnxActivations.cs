using System;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Onnx
{
    /// <summary>
    /// ONNX operators that compute the ANN.Net activations
    /// </summary>
    internal static class OnnxActivations
    {
        private const float GELUActivationAlpha = ActivationFunctions.GELUActivation.DefaultAlpha;

        public static string Apply(OnnxGraph graph, string value, ActivationTypes activation)
        {
            switch (activation)
            {
                case ActivationTypes.Identity:
                    return value;
                case ActivationTypes.Sigmoid:
                    return graph.Node("Sigmoid", new[] { value });
                case ActivationTypes.HyperbolicTangens:
                    return graph.Node("Tanh", new[] { value });
                case ActivationTypes.ArcusTangens:
                    return graph.Node("Atan", new[] { value });
                case ActivationTypes.RELU:
                    return graph.Node("Relu", new[] { value });
                case ActivationTypes.LeakyRELU:
                    return graph.Node("LeakyRelu", new[] { value }, OnnxGraph.FloatAttribute("alpha", 0.01f));
                case ActivationTypes.BinaryRELU:
                    string positive = graph.Node("GreaterOrEqual", new[] { value, Scalar(graph, 0f) });
                    return graph.Node("Cast", new[] { positive }, OnnxGraph.IntAttribute("to", 1));
                case ActivationTypes.Sinusoid:
                    return graph.Node("Sin", new[] { value });
                case ActivationTypes.SoftSign:
                    return graph.Node("Softsign", new[] { value });
                case ActivationTypes.Gaussian:
                    // e^(-x²)
                    return graph.Node("Exp", new[] { graph.Node("Neg", new[] { Square(graph, value) }) });
                case ActivationTypes.SWISH:
                    // x * sigmoid(x)
                    return graph.Node("Mul", new[] { value, graph.Node("Sigmoid", new[] { value }) });
                case ActivationTypes.ISRU:
                    // x / sqrt(1 + x²), alpha = 1
                    return graph.Node("Div", new[] { value, graph.Node("Sqrt", new[] { graph.Node("Add", new[] { Square(graph, value), Scalar(graph, 1f) }) }) });
                case ActivationTypes.SQNL:
                    // with c = clip(x, -2, 2): c - c * |c| / 4 covers all four pieces
                    string clipped = graph.Node("Clip", new[] { value, Scalar(graph, -2f), Scalar(graph, 2f) });
                    string quarter = graph.Node("Mul", new[] { graph.Node("Mul", new[] { clipped, graph.Node("Abs", new[] { clipped }) }), Scalar(graph, 0.25f) });
                    return graph.Node("Sub", new[] { clipped, quarter });
                case ActivationTypes.GELU:
                    // tanh approximation used by ANN.Net: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + alpha * x³)))
                    string cube = graph.Node("Mul", new[] { Square(graph, value), value });
                    string inner = graph.Node("Add", new[] { value, graph.Node("Mul", new[] { cube, Scalar(graph, GELUActivationAlpha) }) });
                    string tanh = graph.Node("Tanh", new[] { graph.Node("Mul", new[] { inner, Scalar(graph, (float)Math.Sqrt(2 / Math.PI)) }) });
                    string half = graph.Node("Mul", new[] { value, Scalar(graph, 0.5f) });
                    return graph.Node("Mul", new[] { half, graph.Node("Add", new[] { tanh, Scalar(graph, 1f) }) });
                default:
                    throw new NotSupportedException($"The {activation} activation cannot be exported to ONNX");
            }
        }

        private static string Scalar(OnnxGraph graph, float value)
        {
            return graph.FloatTensor("constant", new[] { value });
        }

        private static string Square(OnnxGraph graph, string value)
        {
            return graph.Node("Mul", new[] { value, value });
        }
    }
}
