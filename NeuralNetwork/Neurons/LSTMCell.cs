using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Utils;

namespace ANN.Net.Neurons
{
    public class LSTMCell : Cell, ILSTMCell
    {
        private INetwork inputGate, outputGate, forgetGate, inputGateTanh;

        public LSTMCell(ushort inputNeurons, ushort outputNeurons)
            : base(ref inputNeurons, outputNeurons)
        {
            this.inputGate = BuildGate(inputNeurons, outputNeurons);
            this.inputGateTanh = BuildGate(inputNeurons, outputNeurons, ActivationTypes.HyperbolicTangens);

            this.outputGate = BuildGate(inputNeurons, outputNeurons);
            this.forgetGate = BuildGate(inputNeurons, outputNeurons);
        }

        public void Propagate(NeuronPropagateEventArgs value)
        {
            Quad[] forgetResult;

            if (cellState != null)
            {
                forgetResult = MatrixUtils.MatrixHadamard(this.forgetGate.Propagate(value.Values), cellState);
            }
            else
            {
                forgetResult = this.forgetGate.Propagate(value.Values);
            }

            Quad[] inputResult = MatrixUtils.MatrixHadamard(this.inputGate.Propagate(value.Values), this.inputGateTanh.Propagate(value.Values));

            cellState = MatrixUtils.MatrixSum(forgetResult, inputResult);

            Quad[] ht = MatrixUtils.MatrixHadamard(this.outputGate.Propagate(value.Values), MatrixUtils.VectorTanh(cellState));

            foreach (var synapse in this.Outputs)
            {
                synapse.Propagate(new NeuronPropagateEventArgs
                {
                    Values = ht,
                    ClearValue = false
                });
            }
        }

        public void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            throw new System.NotImplementedException();
        }

    }
}
