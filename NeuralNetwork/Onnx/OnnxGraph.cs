using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Compact;

namespace ANN.Net.Onnx
{
    /// <summary>
    /// Collects the nodes and initializers of an ONNX graph and encodes them as a ModelProto
    /// (field numbers from onnx.proto)
    /// </summary>
    internal sealed class OnnxGraph
    {
        public const int OpsetVersion = 13;
        private const long IrVersion = 7;

        private const int FloatType = 1;
        private const int Int64Type = 7;
        private const int Int8Type = 3;
        private const int Float16Type = 10;

        private const int AttributeFloat = 1;
        private const int AttributeInt = 2;
        private const int AttributeTensor = 4;
        private const int AttributeInts = 7;

        private readonly List<Action<ProtoWriter>> nodes = new List<Action<ProtoWriter>>();
        private readonly List<Action<ProtoWriter>> initializers = new List<Action<ProtoWriter>>();
        private int counter;

        /// <summary>
        /// Unique value name
        /// </summary>
        public string Name(string prefix)
        {
            return string.Create(CultureInfo.InvariantCulture, $"{prefix}_{counter++}");
        }

        public string FloatTensor(string prefix, float[] values, params long[] dims)
        {
            string name = Name(prefix);
            initializers.Add(w => WriteTensor(w, name, FloatType, MemoryMarshal.AsBytes(values.AsSpan()).ToArray(), dims));
            return name;
        }

        /// <summary>
        /// Float weights stored in <paramref name="precision"/>: float16 initializers are cast to float, int8
        /// initializers are dequantized with one scale per index of <paramref name="channelAxis"/>, quantized exactly as
        /// compact networks quantize them
        /// </summary>
        public string WeightTensor(string prefix, float[] values, WeightPrecision precision, int channelAxis, params long[] dims)
        {
            switch (precision)
            {
                case WeightPrecision.Half:
                {
                    string name = Name(prefix + "_f16");
                    Half[] half = WeightMatrix.ToHalf(values);
                    initializers.Add(w => WriteTensor(w, name, Float16Type, MemoryMarshal.AsBytes(half.AsSpan()).ToArray(), dims));
                    return Node("Cast", new[] { name }, IntAttribute("to", FloatType));
                }

                case WeightPrecision.Int8:
                {
                    int channels = (int)dims[channelAxis];
                    int stride = (int)dims.Skip(channelAxis + 1).Aggregate(1L, (a, b) => a * b);
                    float[] scales = WeightMatrix.Int8Scales(values, channels, stride);
                    sbyte[] quantized = WeightMatrix.ToInt8(values, scales, channels, stride);

                    string name = Name(prefix + "_int8");
                    initializers.Add(w => WriteTensor(w, name, Int8Type, MemoryMarshal.AsBytes(quantized.AsSpan()).ToArray(), dims));
                    string zero = Name(prefix + "_zero_point");
                    initializers.Add(w => WriteTensor(w, zero, Int8Type, new byte[channels], channels));
                    return Node("DequantizeLinear", new[] { name, FloatTensor(prefix + "_scale", scales, channels), zero }, IntAttribute("axis", channelAxis));
                }

                default:
                    return FloatTensor(prefix, values, dims);
            }
        }

        public string Int64Tensor(string prefix, long[] values, params long[] dims)
        {
            string name = Name(prefix);
            initializers.Add(w => WriteTensor(w, name, Int64Type, MemoryMarshal.AsBytes(values.AsSpan()).ToArray(), dims));
            return name;
        }

        public string Node(string opType, string[] inputs, Action<ProtoWriter> attributes = null, string output = null)
        {
            output ??= Name(opType);
            string name = Name(opType + "_node");

            nodes.Add(w =>
            {
                foreach (string input in inputs) w.String(1, input);
                w.String(2, output);
                w.String(3, name);
                w.String(4, opType);
                attributes?.Invoke(w);
            });

            return output;
        }

        public static Action<ProtoWriter> IntAttribute(string name, long value)
        {
            return w => w.Message(5, a => a.String(1, name).Int64(3, value).Int64(20, AttributeInt));
        }

        public static Action<ProtoWriter> IntsAttribute(string name, params long[] values)
        {
            return w => w.Message(5, a =>
            {
                a.String(1, name);
                foreach (long value in values) a.Int64(8, value);
                a.Int64(20, AttributeInts);
            });
        }

        public static Action<ProtoWriter> FloatAttribute(string name, float value)
        {
            return w => w.Message(5, a => a.String(1, name).Float(2, value).Int64(20, AttributeFloat));
        }

        public static Action<ProtoWriter> FloatTensorAttribute(string name, float value)
        {
            byte[] data = BitConverter.GetBytes(value);
            return w => w.Message(5, a => a.String(1, name).Message(5, t => WriteTensor(t, null, FloatType, data, 1)).Int64(20, AttributeTensor));
        }

        /// <summary>
        /// Encodes the model: one float input and one float output of shape [batch, width]
        /// </summary>
        public byte[] ToModel(string graphName, string inputName, int inputWidth, string outputName, int outputWidth, string docString)
        {
            var model = new ProtoWriter();
            model.Int64(1, IrVersion)
                 .String(2, "ANN.Net")
                 .String(3, typeof(OnnxGraph).Assembly.GetName().Version?.ToString() ?? "1.0")
                 .Message(7, graph =>
                 {
                     foreach (var node in nodes) graph.Message(1, node);
                     graph.String(2, graphName);
                     foreach (var initializer in initializers) graph.Message(5, initializer);
                     if (!string.IsNullOrEmpty(docString)) graph.String(10, docString);
                     graph.Message(11, v => WriteValueInfo(v, inputName, inputWidth));
                     graph.Message(12, v => WriteValueInfo(v, outputName, outputWidth));
                 })
                 .Message(8, opset => opset.String(1, string.Empty).Int64(2, OpsetVersion));

            return model.ToArray();
        }

        private static void WriteTensor(ProtoWriter w, string name, int dataType, byte[] rawData, params long[] dims)
        {
            foreach (long dim in dims) w.Int64(1, dim);
            w.Int64(2, dataType);
            if (name != null) w.String(8, name);
            w.Bytes(9, rawData);
        }

        private static void WriteValueInfo(ProtoWriter w, string name, int width)
        {
            w.String(1, name);
            w.Message(2, type => type.Message(1, tensor =>
            {
                tensor.Int64(1, FloatType);
                tensor.Message(2, shape =>
                {
                    shape.Message(1, dim => dim.String(2, "batch"));
                    shape.Message(1, dim => dim.Int64(1, width));
                });
            }));
        }
    }
}
