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
                case ActivationTypes.SoftStep:
                    return graph.Node("Sigmoid", new[] { value });
                case ActivationTypes.SoftPlus:
                    return graph.Node("Softplus", new[] { value });
                case ActivationTypes.HardSigmoid:
                    return graph.Node("HardSigmoid", new[] { value }, OnnxGraph.Attributes(
                                      OnnxGraph.FloatAttribute("alpha", ActivationFunctions.HardSigmoidActivation.Slope),
                                      OnnxGraph.FloatAttribute("beta", ActivationFunctions.HardSigmoidActivation.Offset)));
                case ActivationTypes.HyperbolicTangens:
                    return graph.Node("Tanh", new[] { value });
                case ActivationTypes.ArcusTangens:
                    return graph.Node("Atan", new[] { value });
                case ActivationTypes.RELU:
                    return graph.Node("Relu", new[] { value });
                case ActivationTypes.LeakyRELU:
                    return graph.Node("LeakyRelu", new[] { value }, OnnxGraph.FloatAttribute("alpha", 0.01f));
                case ActivationTypes.ParametricRELU:
                    // the slope is fixed, so a leaky ReLU with that slope
                    return graph.Node("LeakyRelu", new[] { value }, OnnxGraph.FloatAttribute("alpha", ActivationFunctions.PReLUActivation.DefaultAlpha));
                case ActivationTypes.BinaryRELU:
                case ActivationTypes.BinaryStep:
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
                case ActivationTypes.CubicBSpline:
                    // (relu(2 - |x|)³ - 4 relu(1 - |x|)³) / 6 covers both pieces
                    string absolute = graph.Node("Abs", new[] { value });
                    string outer = Cube(graph, graph.Node("Relu", new[] { graph.Node("Sub", new[] { Scalar(graph, 2f), absolute }) }));
                    string core = Cube(graph, graph.Node("Relu", new[] { graph.Node("Sub", new[] { Scalar(graph, 1f), absolute }) }));
                    string difference = graph.Node("Sub", new[] { outer, graph.Node("Mul", new[] { core, Scalar(graph, 4f) }) });
                    return graph.Node("Div", new[] { difference, Scalar(graph, 6f) });
                case ActivationTypes.TanHShrink:
                    return graph.Node("Sub", new[] { value, graph.Node("Tanh", new[] { value }) });
                case ActivationTypes.SoftShrink:
                    // Shrink: x - bias above lambd, x + bias below -lambd, 0 between
                    return graph.Node("Shrink", new[] { value }, OnnxGraph.Attributes(
                                      OnnxGraph.FloatAttribute("bias", ActivationFunctions.SoftShrinkActivation.DefaultLambda),
                                      OnnxGraph.FloatAttribute("lambd", ActivationFunctions.SoftShrinkActivation.DefaultLambda)));
                case ActivationTypes.HardShrink:
                    return graph.Node("Shrink", new[] { value }, OnnxGraph.Attributes(
                                      OnnxGraph.FloatAttribute("bias", 0f),
                                      OnnxGraph.FloatAttribute("lambd", ActivationFunctions.HardShrinkActivation.DefaultLambda)));
                default:
                    // SoftMax and BReLU are no per-neuron activations, so no network has neurons with them
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

        private static string Cube(OnnxGraph graph, string value)
        {
            return graph.Node("Mul", new[] { Square(graph, value), value });
        }
    }
}
