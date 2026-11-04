using System;
using System.Buffers.Binary;
using System.IO;
using System.Text;

namespace ANN.Net.Onnx
{
    /// <summary>
    /// Minimal Protocol Buffers encoder, enough to write ONNX models without a dependency on a protobuf library
    /// </summary>
    internal sealed class ProtoWriter
    {
        private const int VarintType = 0;
        private const int Fixed32Type = 5;
        private const int LengthDelimitedType = 2;

        private readonly MemoryStream stream = new MemoryStream();

        public byte[] ToArray() => stream.ToArray();

        public ProtoWriter Int64(int field, long value)
        {
            Tag(field, VarintType);
            Varint(unchecked((ulong)value));
            return this;
        }

        public ProtoWriter Float(int field, float value)
        {
            Tag(field, Fixed32Type);
            Span<byte> bytes = stackalloc byte[4];
            BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
            stream.Write(bytes);
            return this;
        }

        public ProtoWriter String(int field, string value)
        {
            return Bytes(field, Encoding.UTF8.GetBytes(value));
        }

        public ProtoWriter Bytes(int field, ReadOnlySpan<byte> value)
        {
            Tag(field, LengthDelimitedType);
            Varint((ulong)value.Length);
            stream.Write(value);
            return this;
        }

        public ProtoWriter Message(int field, Action<ProtoWriter> write)
        {
            var nested = new ProtoWriter();
            write(nested);
            return Bytes(field, nested.ToArray());
        }

        private void Tag(int field, int wireType)
        {
            Varint((ulong)((field << 3) | wireType));
        }

        private void Varint(ulong value)
        {
            while (value >= 0x80)
            {
                stream.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }

            stream.WriteByte((byte)value);
        }
    }
}
