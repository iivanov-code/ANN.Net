using System;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Remote;
using ANN.Net.Remote.Abstactions.Models;
using Xunit;

namespace ANN.Net.Tests.Remote
{
    public class RemoteNetworkPartTests
    {
        private static RemoteNetworkPart Part()
        {
            NetworkSettings whole = SplitHelpers.Whole(2, 1);
            RecurrentLayerSettings[] layers = { SplitHelpers.Layer(3, ActivationTypes.Sigmoid) };
            return RemoteNetworkPart.Create(RemoteNetworkFactory.BuildTail(whole, layers, 0));
        }

        private static RemotePropagationDTO[] Signals(params double[] values)
        {
            return values.Select((v, i) => new RemotePropagationDTO { Id = $"s{i}", Values = new[] { v }, ShouldActivate = true }).ToArray();
        }

        [Fact]
        public void GetInfo_ReportsTheShape()
        {
            RemoteNetworkInfo info = Part().GetInfo();

            Assert.Equal(3, info.InputCount);
            Assert.Equal(1, info.OutputCount);
        }

        [Fact]
        public void Propagate_RunsTheNetworkAndReturnsItsOutputs()
        {
            RemoteNetworkPart part = Part();

            RemotePropagationResult result = part.Propagate(Signals(0.1, 0.2, 0.3));

            Assert.Single(result.Values);
            Assert.Equal((float)part.LocalNetwork.Values[0], (float)result.Values[0], 6);
        }

        [Fact]
        public void Propagate_WithTheWrongNumberOfSignals_Throws()
        {
            RemoteNetworkPart part = Part();

            Assert.Throws<ArgumentException>(() => part.Propagate(Signals(0.1, 0.2)));
            Assert.Throws<ArgumentNullException>(() => part.Propagate(null));
        }

        [Fact]
        public void Backpropagate_ReturnsOneGradientPerSignalWithItsId()
        {
            RemoteNetworkPart part = Part();
            part.Propagate(Signals(0.1, 0.2, 0.3));

            RemoteBackpropagationResult result = part.Backpropagate(new[] { 0.9 });

            Assert.Equal(new[] { "s0", "s1", "s2" }, result.Gradients.Select(g => g.Id));
            Assert.Equal(part.LocalNetwork.InputGradients.Select(g => (float)g), result.Gradients.Select(g => (float)g.ErrorSignal));
            Assert.Single(result.Errors);
            Assert.True(result.Loss >= 0);
        }

        [Fact]
        public void Backpropagate_BeforePropagate_Throws()
        {
            Assert.Throws<InvalidOperationException>(() => Part().Backpropagate(new[] { 0.9 }));
        }

        [Fact]
        public void Create_RejectsAnUnknownNetworkType()
        {
            Assert.Throws<NotSupportedException>(() => RemoteNetworkPart.Create(new UnknownNetwork()));
        }

        private sealed class UnknownNetwork : INetwork
        {
            public Quad[] Values => Array.Empty<Quad>();

            public Quad[] Errors => Array.Empty<Quad>();

            public Quad[] InputGradients => Array.Empty<Quad>();

            public System.Collections.Generic.IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues => null;

            public Quad Backpropagate(params Quad[] targets) => 0;

            public Quad[] Propagate(params Quad[] values) => values;

            public void Save(string fullDirPath, string fileName)
            { }
        }
    }
}
