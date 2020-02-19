using NeuralNetwork.Connections;
using NeuralNetwork.Enums;
using NeuralNetwork.HelperClasses;
using NeuralNetwork.Interfaces;
using NeuralNetwork.Neurons;
using NeuralNetwork.Settings;
using NeuralNetwork.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    internal class Network : INetwork
    {
        public readonly float MinValue, MaxValue;

        private bool ShouldNormalize;

        protected readonly internal IReadOnlyList<IInputNeuron> InputNeurons;
        protected internal ICollection<ICollection<IHiddenNeuron>> HiddenLayers;
        protected readonly internal IReadOnlyList<IOutputNeuron> OutputNeurons;

        public Network(NetworkSettings settings)
        {
            this.MinValue = settings.MinNetworkValue;
            this.MaxValue = settings.MaxNetworkValue;
            this.ShouldNormalize = settings.ShouldNormalize;
            this.InputNeurons = new List<IInputNeuron>();
            this.OutputNeurons = new List<IOutputNeuron>();
        }

        internal virtual IList<ICollection<IHiddenNeuron>> BuildHiddenLayers(IEnumerable<INeuron> previousLayer, HiddenLayerSettings settings, ushort recurrentInputs = 0)
        {
            var tempLayer = new List<ICollection<IHiddenNeuron>>(settings.LayersCount);

            IEnumerable<INeuron> prevLayer = previousLayer;

            for (int i = 0; i < settings.LayersCount; i++)
            {
                var tempHidden = new List<IHiddenNeuron>(settings.NeuronsCount);
                for (int j = 0; j < settings.NeuronsCount; j++)
                {
                    var neuron = new HiddenNeuron(settings.FunctionType, recurrentInputs);
                    tempHidden.Add(neuron);
                    foreach (var inNeuron in prevLayer)
                    {
                        ConnectAxon(inNeuron, neuron, settings.NeuronsCount);
                    }
                }

                if (settings.HasBiasNeuron)
                {
                    var biasNeuron = new BiasNeuron();

                    tempHidden.Add(biasNeuron);

                    foreach (var inNeuron in previousLayer)
                    {
                        ConnectAxon(inNeuron, biasNeuron, settings.NeuronsCount);
                    }
                }
                prevLayer = tempHidden;
                tempLayer.Add(tempHidden);
            }

            return tempLayer;
        }

        internal IList<ICollection<ILSTMCell>> BuildLSTMCells(IEnumerable<INeuron> prevLayer, CellLayerSettings settings)
        {
            var tempLayers = new List<ICollection<ILSTMCell>>(settings.LayersCount);

            ushort recurrentInputs = (ushort)(settings.HasSelfConnection ? 1 : 0);

            for (int i = 0; i < settings.LayersCount; i++)
            {
                var tempHidden = new List<ILSTMCell>(settings.NeuronsCount);
                for (int j = 0; j < settings.NeuronsCount; j++)
                {
                    ushort rn = (ushort)(recurrentInputs + (settings.Direction == RecurrentDirection.Both ? settings.NeuronsCount : j));
                    var neuron = new LSTMCell(settings.FunctionType, rn);

                    foreach (var inNeuron in prevLayer)
                    {
                        ConnectAxon(inNeuron, neuron, settings.NeuronsCount);
                    }
                    tempHidden.Add(neuron);
                }

                if (settings.Direction == RecurrentDirection.Forward || settings.Direction == RecurrentDirection.Both)
                {
                    for (int k = 0; k < settings.NeuronsCount; k += 2)
                    {
                        if (k + 1 < settings.NeuronsCount)
                            ConnectRecurrentAxon(tempHidden[k], tempHidden[k + 1], settings.HasSelfConnection);
                    }
                }

                if (settings.Direction == RecurrentDirection.Backward || settings.Direction == RecurrentDirection.Both)
                {
                    for (int k = settings.NeuronsCount - 1; k >= 0; k -= 2)
                    {
                        if (k - 1 < settings.NeuronsCount)
                            ConnectRecurrentAxon(tempHidden[k], tempHidden[k - 1], settings.HasSelfConnection);
                    }
                }
            }
            return tempLayers;
        }

        private void ConnectRecurrentAxon(ILSTMCell first, ILSTMCell second, bool selfConnection)
        {
            var axon = new RecurrentSynapse(first, second, RandomWeight());
            (first.Outputs as SynapseCollection<ISynapse>).Add(axon);
            (second.Inputs as SynapseCollection<ISynapse>).Add(axon);
        }

        internal IEnumerable<IInputNeuron> BuildInputNeurons(NetworkSettings settings, ushort recurrentInputs = 0)
        {
            var tempInputNeurons = new List<IInputNeuron>(settings.InputNeuronsCount);

            for (int i = 0; i < settings.InputNeuronsCount; i++)
            {
                tempInputNeurons.Add(new InputNeuron(settings.InputLayerFunction, recurrentInputs));
            }

            if (settings.HasBiasNeuron)
                tempInputNeurons.Add(new BiasNeuron());

            ((List<IInputNeuron>)this.InputNeurons).AddRange(tempInputNeurons);
            return tempInputNeurons;
        }

        internal IEnumerable<IOutputNeuron> BuildOutputNeurons(IEnumerable<INeuron> prevLayer, ushort count, ActivationTypes type)
        {
            var tempOutputLayer = new List<IOutputNeuron>(count);

            for (int i = 0; i < count; i++)
            {
                var neuron = new OutputNeuron(type);
                tempOutputLayer.Add(neuron);
                foreach (var inNeuron in prevLayer)
                {
                    ConnectAxon(inNeuron, neuron, count);
                }
            }

            ((List<IOutputNeuron>)this.OutputNeurons).AddRange(tempOutputLayer);
            return tempOutputLayer;
        }

        public float[] Values
        {
            get
            {
                return OutputNeurons.Select(x => x.Value).ToArray();
            }
        }

        public float[] Propagate(float[] values)
        {
            //if (ShouldNormalize)
            //{
            //    float[] normalizedValues = new float[values.Length];
            //    for (int i = 0; i < values.Length; i++)
            //    {
            //        normalizedValues[i] = NetworkUtils.Normalize(values[i], MinValue, MaxValue);
            //    }
            //    values = normalizedValues;
            //}

            for (int i = 0; i < values.Length; i++)
            {
                InputNeurons[i].Propagate(values[i]);
            }

            if (InputNeurons.Count > values.Length)
            {
                InputNeurons[values.Length].Propagate(1);
            }

            return Values;
        }

        public float GetNormalizedValue(float value)
        {
            return NetworkUtils.UnNormalize(value, MinValue, MaxValue);
        }


        public float Backpropagate(float[] targets)
        {
            //if (ShouldNormalize)
            //{
            //    float[] normalizedValues = new float[targets.Length];
            //    for (int i = 0; i < targets.Length; i++)
            //    {
            //        normalizedValues[i] = NetworkUtils.Normalize(targets[i], MinValue, MaxValue);
            //    }
            //    targets = normalizedValues;
            //}

            float totalError = 0;

            for (int i = 0; i < targets.Length; i++)
            {
                float error = MeanSquaredError(targets[i], Values[i]);
                OutputNeurons[i].Backpropagate(error);
                totalError += error;
            }

            return totalError;
        }

        private static float MeanSquaredError(float target, float predicted)
        {
            return (float)Math.Pow(predicted - target, 2) / 2f;
            //return predicted - target;
        }

        public void ConnectAxon(INeuron input, INeuron output, int neuronsCount)
        {
            var axon = new Synapse(input, output, XavierWeight(neuronsCount));
            (output.Inputs as SynapseCollection<ISynapse>).Add(axon);
            (input.Outputs as SynapseCollection<ISynapse>).Add(axon);
        }

        private static float RandomWeight()
        {
            return NetworkUtils.GetRandomNumber(0, 1);
        }

        private static float XavierWeight(int neuronCount)
        {
            return (1 / NetworkUtils.GetRandomNumber(1, neuronCount)) + NetworkUtils.GetRandomNumber(0.001f, 0.009f, 4);
        }
    }
}
