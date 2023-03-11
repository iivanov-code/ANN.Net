using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Neurons
{
    internal class NetLSTMCell : BaseNeuron, ILSTMCell
    {
        private INetwork forgetGate, inputGate, outputGate, cellStateGate;
        private ushort hiddenUnits, windowSize;
        private Quad[] prevValues, prevCellState;

        public NetLSTMCell(bool initInputs, bool initOutputs, ushort hiddenUnits, ushort windowSize)
             : base(initOutputs, initOutputs)
        {
            this.hiddenUnits = hiddenUnits;
            this.windowSize = windowSize;
            this.prevCellState = new Quad[hiddenUnits];
            this.prevValues = new Quad[windowSize];
            this.forgetGate = BuildNetwork(windowSize, hiddenUnits, ActivationTypes.Sigmoid);
            this.inputGate = BuildNetwork(windowSize, hiddenUnits, ActivationTypes.Sigmoid);
            this.outputGate = BuildNetwork(windowSize, hiddenUnits, ActivationTypes.Sigmoid);
            this.cellStateGate = BuildNetwork(windowSize, hiddenUnits, ActivationTypes.HyperbolicTangens);
        }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            throw new NotImplementedException();
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            //   this.Inputs.
        }

        public void Propagate(CellPropagateEventArgs value)
        {
            Quad[] forgetValues = this.forgetGate.Propagate(value.Values);
            Quad[] inputValues = this.inputGate.Propagate(value.Values);
            Quad[] outputValues = this.outputGate.Propagate(value.Values);
            Quad[] cellStateValues = this.cellStateGate.Propagate(value.Values);

            cellStateValues = PointwiseMultiply(inputValues, cellStateValues);
        }

        protected override void ActivateNeuron(NeuronPropagateEventArgs value)
        {
            throw new NotImplementedException();
        }

        protected override void FeedbackError(BackpropagateEventArgs errorSignal)
        {
            throw new NotImplementedException();
        }

        private static Quad[] PointwiseMultiply(Quad[] a, Quad[] b)
        {
            for (int i = 0; i < a.Length; i++)
            {
                a[i] *= b[i];
            }

            return a;
        }

        private static Quad[] PointwiseSum(Quad[] a, Quad[] b)
        {
            for (int i = 0; i < a.Length; i++)
            {
                a[i] += b[i];
            }

            return a;
        }

        private static INetwork BuildNetwork(ushort windowSize, ushort hiddenUnits, ActivationTypes type)
        {
            return NetworkFactory.BuildFFN(new NetworkSettings
            {
                InputSettings = new InputLayerSettings
                {
                    NeuronsCount = windowSize,
                    ActivationType = type,
                    InitializationType = WeightInitType.Xavier
                },
                OutputSettings = new OutputLayerSettings
                {
                    NeuronsCount = hiddenUnits,
                    ActivationType = type,
                    InitializationType = WeightInitType.Xavier
                }
            }, null);
        }
    }
}
