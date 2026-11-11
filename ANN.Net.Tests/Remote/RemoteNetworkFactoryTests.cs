using System;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Neurons;
using ANN.Net.Remote;
using Xunit;

namespace ANN.Net.Tests.Remote
{
    public class RemoteNetworkFactoryTests
    {
        [Fact]
        public void Split_ExpandsLayersCountAndCutsAfterTheBoundaryLayer()
        {
            NetworkSettings whole = SplitHelpers.Whole(3, 2, ActivationTypes.Sigmoid, inputBias: true);
            RecurrentLayerSettings[] layers =
            {
                SplitHelpers.Layer(4, ActivationTypes.HyperbolicTangens, bias: true, layersCount: 2),
                SplitHelpers.Layer(5, ActivationTypes.Sigmoid)
            };

            NetworkSplit split = RemoteNetworkFactory.Split(whole, layers, 1);

            Assert.Equal(1, split.BoundaryLayer);
            Assert.Equal(4, split.BoundaryNeurons);

            Assert.Same(whole.InputSettings, split.HeadSettings.InputSettings);
            Assert.Single(split.HeadHiddenLayers);
            Assert.Equal(4, split.HeadHiddenLayers[0].NeuronsCount);
            Assert.Equal(1, split.HeadHiddenLayers[0].LayersCount);
            Assert.Equal(4, split.HeadSettings.OutputSettings.NeuronsCount);
            Assert.Equal(ActivationTypes.HyperbolicTangens, split.HeadSettings.OutputSettings.ActivationType);
            Assert.False(split.HeadSettings.OutputApplySoftMax);

            Assert.Equal(4, split.TailSettings.InputSettings.NeuronsCount);
            Assert.Equal(ActivationTypes.Identity, split.TailSettings.InputSettings.ActivationType);
            Assert.True(split.TailSettings.InputSettings.HasBiasNeuron, "the boundary layer's bias feeds the next layer, so it belongs to the tail");
            Assert.Single(split.TailHiddenLayers);
            Assert.Equal(5, split.TailHiddenLayers[0].NeuronsCount);
            Assert.Same(whole.OutputSettings, split.TailSettings.OutputSettings);
            Assert.Equal(whole.LossType, split.TailSettings.LossType);
            Assert.Same(whole.IdentityGenerator, split.TailSettings.IdentityGenerator);
        }

        [Fact]
        public void Split_AtTheFirstAndLastLayer_LeavesOneSideWithoutHiddenLayers()
        {
            NetworkSettings whole = SplitHelpers.Whole(2, 1);
            RecurrentLayerSettings[] layers = { SplitHelpers.Layer(3, ActivationTypes.Sigmoid), SplitHelpers.Layer(4, ActivationTypes.Sigmoid) };

            NetworkSplit first = RemoteNetworkFactory.Split(whole, layers, 0);
            NetworkSplit last = RemoteNetworkFactory.Split(whole, layers, 1);

            Assert.Empty(first.HeadHiddenLayers);
            Assert.Equal(3, first.HeadSettings.OutputSettings.NeuronsCount);
            Assert.Single(first.TailHiddenLayers);

            Assert.Single(last.HeadHiddenLayers);
            Assert.Equal(4, last.HeadSettings.OutputSettings.NeuronsCount);
            Assert.Empty(last.TailHiddenLayers);
        }

        [Fact]
        public void Split_RejectsBadInput()
        {
            NetworkSettings whole = SplitHelpers.Whole(2, 1);
            RecurrentLayerSettings[] layers = { SplitHelpers.Layer(3, ActivationTypes.Sigmoid) };

            Assert.Throws<ArgumentOutOfRangeException>(() => RemoteNetworkFactory.Split(whole, layers, -1));
            Assert.Throws<ArgumentOutOfRangeException>(() => RemoteNetworkFactory.Split(whole, layers, 1));
            Assert.Throws<ArgumentException>(() => RemoteNetworkFactory.Split(whole, Array.Empty<RecurrentLayerSettings>(), 0));
            Assert.Throws<ArgumentNullException>(() => RemoteNetworkFactory.Split(null, layers, 0));

            RecurrentLayerSettings lstm = SplitHelpers.Layer(3, ActivationTypes.Sigmoid);
            lstm.CellType = NeuronType.LSTM;
            Assert.Throws<NotSupportedException>(() => RemoteNetworkFactory.Split(whole, new[] { lstm }, 0));

            NetworkSettings withoutIds = SplitHelpers.Whole(2, 1);
            withoutIds.IdentityGenerator = null;
            Assert.Throws<ArgumentException>(() => RemoteNetworkFactory.Split(withoutIds, layers, 0));
        }

        [Fact]
        public void BuildTail_HasAnInputPerBoundaryNeuronAndTheRealOutputs()
        {
            NetworkSettings whole = SplitHelpers.Whole(3, 2);
            RecurrentLayerSettings[] layers = { SplitHelpers.Layer(4, ActivationTypes.Sigmoid, bias: true), SplitHelpers.Layer(5, ActivationTypes.Sigmoid) };

            Network tail = (Network)RemoteNetworkFactory.BuildTail(whole, layers, 0);

            Assert.Equal(4, tail.InputNeurons.Count(n => !(n is BiasNeuron)));
            Assert.Single(tail.InputNeurons.Where(n => n is BiasNeuron));
            Assert.Single(tail.HiddenLayers);
            Assert.Equal(5, tail.HiddenLayers.Single().Count);
            Assert.Equal(2, tail.OutputNeurons.Count);
        }

        [Fact]
        public void BuildHead_AttachesOneRemoteSynapsePerBoundaryNeuron()
        {
            NetworkSettings whole = SplitHelpers.Whole(3, 2);
            RecurrentLayerSettings[] layers = { SplitHelpers.Layer(4, ActivationTypes.Sigmoid, bias: true), SplitHelpers.Layer(5, ActivationTypes.Sigmoid) };

            SplitNetwork split = SplitHelpers.BuildInProcess(whole, layers, 0, out Network tail);
            Network head = (Network)split.Head;

            Assert.Equal(3, split.InputCount);
            Assert.Equal(4, split.BoundaryCount);
            Assert.Equal(2, split.OutputCount);
            Assert.Null(head.HiddenLayers);
            Assert.Equal(4, head.OutputNeurons.Count);
            Assert.Equal(4, split.Connection.Synapses.Count);
            Assert.All(head.OutputNeurons, neuron => Assert.IsType<RemoteSynapse>(Assert.Single(neuron.Outputs)));
            Assert.Equal(split.RemoteSynapses.Select(s => s.Id), split.Connection.Synapses.Select(s => s.Id));
            Assert.Same(tail, ((RemoteNetworkPart)split.Connection.Remote).LocalNetwork);
        }
    }
}
