using System;
using System.IO;
using System.Text;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Compact;

namespace ANN.Net
{
    /// <summary>
    /// Saves compact networks as binary files and loads them back. The weights are written as they are stored (32-bit,
    /// 16-bit or 8-bit, dense or sparse), so a file is about <see cref="ICompactNetwork.SizeInBytes"/> long, and a loaded
    /// network computes bit for bit what the saved one computes. Files are checked on loading; a damaged file throws
    /// <see cref="InvalidDataException"/>. To keep training a network later, save the network itself with
    /// <see cref="NetworkSerializer"/>: a compact network has no optimizer state
    /// </summary>
    public static class CompactNetworkSerializer
    {
        public const string FileExtension = "annc";

        private const int Version = 1;
        private static readonly byte[] Magic = Encoding.ASCII.GetBytes("ANNC");

        public static void Save(ICompactNetwork network, string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = File.Create(filePath);
            Save(network, stream);
        }

        public static void Save(ICompactNetwork network, Stream stream)
        {
            ArgumentNullException.ThrowIfNull(network);
            ArgumentNullException.ThrowIfNull(stream);
            EnsureLittleEndian();

            using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
            writer.Write(Magic);
            writer.Write(Version);
            writer.Write((byte)network.Precision);

            switch (network)
            {
                case CompactGraphNetwork graph:
                    writer.Write((byte)1);
                    graph.Write(writer);
                    break;

                case CompactConvolutionalNetwork convolutional:
                    writer.Write((byte)2);
                    convolutional.Write(writer);
                    break;

                default:
                    throw new NotSupportedException($"Saving a {network.GetType().Name} is not supported");
            }
        }

        public static void Save(ICompactSequenceNetwork network, string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = File.Create(filePath);
            Save(network, stream);
        }

        public static void Save(ICompactSequenceNetwork network, Stream stream)
        {
            ArgumentNullException.ThrowIfNull(network);
            ArgumentNullException.ThrowIfNull(stream);
            EnsureLittleEndian();

            if (network is not CompactTransformer transformer)
                throw new NotSupportedException($"Saving a {network.GetType().Name} is not supported");

            using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
            writer.Write(Magic);
            writer.Write(Version);
            writer.Write((byte)network.Precision);
            writer.Write((byte)3);
            transformer.Write(writer);
        }

        public static ICompactNetwork Load(string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = File.OpenRead(filePath);
            return Load(stream);
        }

        public static ICompactNetwork Load(Stream stream)
        {
            return Read<ICompactNetwork>(stream, (reader, precision, kind) => kind switch
            {
                1 => CompactGraphNetwork.Read(reader, precision),
                2 => CompactConvolutionalNetwork.Read(reader, precision),
                3 => throw new InvalidDataException("The file holds a sequence network; load it with LoadSequence"),
                _ => throw new InvalidDataException($"Unknown compact network kind {kind}")
            });
        }

        public static ICompactSequenceNetwork LoadSequence(string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = File.OpenRead(filePath);
            return LoadSequence(stream);
        }

        public static ICompactSequenceNetwork LoadSequence(Stream stream)
        {
            return Read<ICompactSequenceNetwork>(stream, (reader, precision, kind) => kind == 3
                ? CompactTransformer.Read(reader, precision)
                : throw new InvalidDataException("The file does not hold a sequence network; load it with Load"));
        }

        private static T Read<T>(Stream stream, Func<BinaryReader, WeightPrecision, byte, T> read)
        {
            ArgumentNullException.ThrowIfNull(stream);
            EnsureLittleEndian();

            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            try
            {
                CompactIO.Require(reader.ReadBytes(Magic.Length).AsSpan().SequenceEqual(Magic), "The file is not an ANN.Net compact network");
                int version = reader.ReadInt32();
                CompactIO.Require(version == Version, $"Unsupported compact network version {version}");

                var precision = (WeightPrecision)reader.ReadByte();
                CompactIO.Require(Enum.IsDefined(precision), $"Unknown precision {(int)precision}");

                return read(reader, precision, reader.ReadByte());
            }
            catch (EndOfStreamException exception)
            {
                throw new InvalidDataException("The file ends too early", exception);
            }
        }

        private static void EnsureLittleEndian()
        {
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("Compact network files are written on little-endian machines only");
        }
    }
}
