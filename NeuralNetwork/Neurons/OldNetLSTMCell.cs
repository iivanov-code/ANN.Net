using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Utils;

namespace ANN.Net.Neurons
{
    [Serializable]
    internal class OldNetLSTMCell : HiddenNeuron, ILSTMCell
    {
        private Quad[] ct, ht; //ht - output vector, ct - hidden cell state
        private INetwork inputGate, outputGate, forgetGate, cellStateGate;
        private Network inputGateRecurrent, outputGateRecurrent, forgetGateRecurrent, cellStateGateReccurent;

        public OldNetLSTMCell(ushort inputs, ushort outputs)
             : base(ActivationTypes.Sigmoid)
        {
            this.Inputs = new Synapses();
            this.Outputs = new Synapses();

            var builder = NetworkBuilder.Create();
            inputGateRecurrent = CreateInstance(builder, outputs, ActivationTypes.Sigmoid);
            outputGateRecurrent = CreateInstance(builder, outputs, ActivationTypes.Sigmoid);
            forgetGateRecurrent = CreateInstance(builder, outputs, ActivationTypes.Sigmoid);
            cellStateGateReccurent = CreateInstance(builder, outputs, ActivationTypes.HyperbolicTangens);

            inputGate = CreateNetwork(inputs, outputs, ActivationTypes.Sigmoid, inputGateRecurrent);
            outputGate = CreateNetwork(inputs, outputs, ActivationTypes.Sigmoid, outputGateRecurrent);
            forgetGate = CreateNetwork(inputs, outputs, ActivationTypes.Sigmoid, forgetGateRecurrent);
            cellStateGate = CreateNetwork(inputs, outputs, ActivationTypes.HyperbolicTangens, cellStateGateReccurent);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
        }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
        }

        public void Propagate(Quad[] values)
        {
            var it = inputGate.Propagate(values);
            var ft = forgetGate.Propagate(values);
            var ot = outputGate.Propagate(values);
            var cth = cellStateGate.Propagate(values);

            var temp = MatrixUtils.MatrixHadamard(ft, ct);
            var temp1 = MatrixUtils.MatrixHadamard(it, cth);
            ct = MatrixUtils.VectorSum(temp, temp1);

            ht = MatrixUtils.MatrixHadamard(MatrixUtils.VectorTanh(ct), ot);

            foreach (var output in this.Outputs)
            {
                //output.Propagate(ht);
            }

            inputGateRecurrent.Propagate(ht);
            outputGateRecurrent.Propagate(ht);
            forgetGateRecurrent.Propagate(ht);
            cellStateGateReccurent.Propagate(ht);
        }

        public void Backpropagate(Quad[] errors)
        {
            inputGateRecurrent.Backpropagate(errors);
            outputGateRecurrent.Backpropagate(errors);
            forgetGateRecurrent.Backpropagate(errors);
            cellStateGateReccurent.Backpropagate(errors);
            inputGate.Backpropagate(errors);
            outputGate.Backpropagate(errors);
            forgetGate.Backpropagate(errors);
            cellStateGate.Backpropagate(errors);
        }

        private static INetwork CreateNetwork(ushort inputs, ushort outputs, ActivationTypes activationType, Network network)
        {
            var settings = new NetworkSettings
            {
                InputSettings = new InputLayerSettings
                {
                    NeuronsCount = inputs,
                    ActivationType = activationType
                },
                OutputSettings = new OutputLayerSettings
                {
                    NeuronsCount = outputs,
                    ActivationType = activationType
                }
            };

            var newNetwork = NetworkFactory.BuildFFN(settings, network.InputNeurons) as Network;

            network.OutputNeurons = newNetwork.OutputNeurons;

            return newNetwork;
        }

        private static Network CreateInstance(NetworkBuilder builder, ushort outputs, ActivationTypes activationType)
        {
            var network = new Network(new NetworkSettings
            {
                OutputSettings = new OutputLayerSettings
                {
                    NeuronsCount = outputs
                }
            });

            network.InputNeurons = builder.BuildInputNeurons(new InputLayerSettings
            {
                NeuronsCount = outputs,
                ActivationType = activationType
            });

            return network;
        }

        public void Propagate(CellPropagateEventArgs value)
        {
            throw new NotImplementedException();
        }
    }
}
