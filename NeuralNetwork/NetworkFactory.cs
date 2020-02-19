using NeuralNetwork.Interfaces;
using NeuralNetwork.Settings;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public static class NetworkFactory
    {
        public static INetwork BuildFFN(NetworkSettings settings, HiddenLayerSettings hiddenSettings)
        {
            Network ffn = new Network(settings);

            ffn.BuildInputNeurons(settings);
            ffn.HiddenLayers = ffn.BuildHiddenLayers(ffn.InputNeurons, hiddenSettings);
            ffn.BuildOutputNeurons(ffn.HiddenLayers.Last(), settings.OutputNeuronsCount, settings.OutputLayerFunction);
            return ffn;
        }

        public static INetwork BuildRNN(NetworkSettings settings, HiddenLayerSettings hiddenSettings)
        {
            Network rnn = new Network(settings);

            rnn.BuildInputNeurons(settings);
            rnn.HiddenLayers = rnn.BuildHiddenLayers(rnn.InputNeurons, hiddenSettings);
            var prevLayer = rnn.BuildOutputNeurons(rnn.HiddenLayers.Last(), settings.OutputNeuronsCount, settings.OutputLayerFunction);
            var outputLayer = rnn.BuildInputNeurons(settings, 1);
            var prevLayerEnumerator = prevLayer.GetEnumerator();
            var outputLayerEnumerator = outputLayer.GetEnumerator();
            while (prevLayerEnumerator.MoveNext())
            {
                outputLayerEnumerator.MoveNext();
                rnn.ConnectAxon(prevLayerEnumerator.Current, outputLayerEnumerator.Current, settings.OutputNeuronsCount);
            }

            return rnn;
        }

        public static INetwork BuildLSTM(NetworkSettings settings, CellLayerSettings lstmSettings, HiddenLayerSettings prevHiddenSettings, HiddenLayerSettings lastHiddenSettings)
        {
            Network lstm = new Network(settings);

            IEnumerable<INeuron> prevLayer = lstm.BuildInputNeurons(settings);

            if (prevHiddenSettings != null)
                lstm.HiddenLayers = lstm.BuildHiddenLayers(prevLayer, prevHiddenSettings);

            if (lstm.HiddenLayers != null)
            {
                prevLayer = lstm.HiddenLayers.Last();
            }
            else
            {
                lstm.HiddenLayers = new List<ICollection<IHiddenNeuron>>();
                prevLayer = lstm.InputNeurons;
            }

            var lstmLayers = lstm.BuildLSTMCells(prevLayer, lstmSettings);
            foreach (ICollection<IHiddenNeuron> layer in lstmLayers)
            {
                lstm.HiddenLayers.Add(layer);
                prevLayer = layer;
            }

            if (lastHiddenSettings != null)
            {
                var hiddenLayers = lstm.BuildHiddenLayers(prevLayer, lastHiddenSettings);
                foreach (ICollection<IHiddenNeuron> layer in hiddenLayers)
                {
                    lstm.HiddenLayers.Add(layer);
                }
            }

            return lstm;
        }

        public static INetwork BuildCellRNN(NetworkSettings settings, CellLayerSettings[] cellSettings)
        {


            return null;
        }

    }
}
