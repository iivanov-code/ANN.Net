using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
        public const string FILE_EXTENSION = NetworkSerializer.FileExtension;

        /// <summary>
        /// Loads a network saved with <see cref="Save(string, string)"/> or <see cref="NetworkSerializer"/>
        /// </summary>
        public static INetwork Load(string fullFilePath)
        {
            return NetworkSerializer.Load(fullFilePath);
        }

        /// <summary>
        /// Loads a network saved with <see cref="Save(string, string)"/> or <see cref="NetworkSerializer"/>
        /// </summary>
        public static INetwork Load(Stream stream)
        {
            return NetworkSerializer.Load(stream);
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

        /// <summary>
        /// Saves the network as JSON to <paramref name="fullDirPath"/>/<paramref name="fileName"/>.ann
        /// </summary>
        public void Save(string fullDirPath, string fileName)
        {
            NetworkSerializer.Save((INetwork)this, Path.Combine(fullDirPath, $"{fileName}.{FILE_EXTENSION}"));
        }

        /// <summary>
        /// Saves the network as JSON to a stream
        /// </summary>
        public void Save(Stream stream)
        {
            NetworkSerializer.Save((INetwork)this, stream);
        }
    }
}
