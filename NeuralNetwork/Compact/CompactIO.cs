using System;
using System.IO;
using System.Runtime.InteropServices;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Compact
{
    /// <summary>
    /// Arrays and checks of the compact network file format; arrays are a length (-1 for none) and the raw
    /// little-endian values
    /// </summary>
    internal static class CompactIO
    {
        public static void WriteArray<T>(this BinaryWriter writer, T[] values)
            where T : unmanaged
        {
            if (values == null)
            {
                writer.Write(-1);
                return;
            }

            writer.Write(values.Length);
            writer.Write(MemoryMarshal.AsBytes(values.AsSpan()));
        }

        public static T[] ReadArray<T>(this BinaryReader reader, bool allowNull = false)
            where T : unmanaged
        {
            int length = reader.ReadInt32();
            if (length == -1 && allowNull) return null;
            Require(length >= 0, $"Invalid array length {length}");

            Stream stream = reader.BaseStream;
            long bytes = (long)length * Marshal.SizeOf<T>();
            if (stream.CanSeek) Require(bytes <= stream.Length - stream.Position, "The file ends inside an array");

            var values = new T[length];
            stream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
            return values;
        }

        public static ActivationTypes ReadActivation(this BinaryReader reader)
        {
            var activation = (ActivationTypes)reader.ReadInt32();
            Require(Enum.IsDefined(activation), $"Unknown activation {(int)activation}");
            if (activation == ActivationTypes.Identity) return activation;

            try
            {
                Utils.NetworkUtils.GetActivation(activation);
            }
            catch (ArgumentException exception)
            {
                throw new InvalidDataException($"Unsupported activation {activation}", exception);
            }

            return activation;
        }

        public static void Require(bool condition, string message)
        {
            if (!condition) throw new InvalidDataException(message);
        }
    }
}
