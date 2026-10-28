using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters;
using System.Runtime.Serialization.Formatters.Binary;

using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net
{
    [Serializable]
    public abstract class ANNetwork
    {
        public const string FILE_EXTENSION = "ann";

        public static INetwork Load(string fullFilePath)
        {
            using (FileStream fileStream = new FileStream(fullFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                return Formatter.Deserialize(fileStream) as INetwork;
            }
        }

        public static INetwork BuildNetwork(IEnumerable<IInputNeuron> inputNeurons, IEnumerable<IHiddenNeuron> hiddenNeurons, IEnumerable<IOutputNeuron> outputNeurons)
        {
            var network = new Network(new NetworkSettings
            {
                InputSettings = new InputLayerSettings
                {
                    HasBiasNeuron = false,
                    InitializationType = WeightInitType.Random,
                    NeuronsCount = (ushort)inputNeurons.Count(),
                    ActivationType = ActivationTypes.Sigmoid,
                    OptimizerType = OptimizerTypes.Normal
                },
                OutputSettings = new OutputLayerSettings
                {
                    ActivationType = ActivationTypes.Sigmoid,
                    HasBiasNeuron = false,
                    InitializationType = WeightInitType.Random,
                    NeuronsCount = (ushort)outputNeurons.Count(),
                    OptimizerType = OptimizerTypes.Normal
                },
                LossType = LossFunctionTypes.Difference,
                OutputApplySoftMax = true,
                ShouldNormalize = true,
            });

            network.InputNeurons = new Layer<IInputNeuron>(inputNeurons);
            network.HiddenLayers = new List<ILayer<IHiddenNeuron>>();
            network.HiddenLayers.Add(new Layer<IHiddenNeuron>(hiddenNeurons));
            network.OutputNeurons = new Layer<IOutputNeuron>(outputNeurons);

            return network;
        }

        public void Save(string fullDirPath, string fileName)
        {
            string fullFilePath = Path.Combine(fullDirPath, $"{fileName}.{FILE_EXTENSION}");
            using FileStream fileStream = new FileStream(fullFilePath, FileMode.Create, FileAccess.ReadWrite, FileShare.Read);
            Formatter.Serialize(fileStream, this);
        }

#pragma warning disable SYSLIB0011 // Type or member is obsolete
        private static BinaryFormatter Formatter
        {
            get
            {
                BinaryFormatter formatter = new BinaryFormatter();
#pragma warning disable SYSLIB0050 // Type or member is obsolete
                formatter.TypeFormat = FormatterTypeStyle.XsdString;
#pragma warning restore SYSLIB0050 // Type or member is obsolete
#pragma warning disable SYSLIB0050 // Type or member is obsolete
                formatter.AssemblyFormat = FormatterAssemblyStyle.Simple;
#pragma warning restore SYSLIB0050 // Type or member is obsolete
#pragma warning disable SYSLIB0050 // Type or member is obsolete
                formatter.FilterLevel = TypeFilterLevel.Full;
#pragma warning restore SYSLIB0050 // Type or member is obsolete
                return formatter;
            }
        }
#pragma warning restore SYSLIB0011 // Type or member is obsolete
    }
}
