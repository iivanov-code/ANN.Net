using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Utils;

namespace ANN.Net.Neurons
{
    internal class GRUCell : Cell, IGRUCell
    {
        private INetwork resetGate, updateGate, targetGate;
        private ushort inputSize, outputSize;

        public GRUCell(ref ushort inputNeurons, ushort outputNeurons)
            : base(ref inputNeurons, outputNeurons)
        {
            this.outputSize = outputNeurons;
            this.inputSize = (ushort)(inputNeurons - outputNeurons);
            
            this.resetGate = BuildGate(inputNeurons, outputNeurons);
            this.updateGate = BuildGate(inputNeurons, outputNeurons);
            this.targetGate = BuildGate(inputNeurons, outputNeurons, ActivationTypes.HyperbolicTangens);
        }

        public void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            throw new NotImplementedException();
        }

        public void Propagate(NeuronPropagateEventArgs value)
        {
            // value.Values typically contains [x_t, h_{t-1}] concatenated when cellState exists
            // On first call (cellState == null), it may contain only x_t
            // Split into input and previous hidden state
            Quad[] xt = new Quad[inputSize];
            Quad[] htPrev = null;
            
            if (cellState != null && value.Values.Length == inputSize + outputSize)
            {
                Array.Copy(value.Values, 0, xt, 0, inputSize);
                htPrev = new Quad[outputSize];
                Array.Copy(value.Values, inputSize, htPrev, 0, outputSize);
            }
            else if (value.Values.Length >= inputSize)
            {
                Array.Copy(value.Values, 0, xt, 0, inputSize);
            }
            else
            {
                // Fallback for unexpected input size - use what we have
                xt = value.Values;
            }
            
            // Compute reset gate: r_t = sigmoid(W_r * [x_t, h_{t-1}])
            Quad[] resetResult = this.resetGate.Propagate(value.Values);
            
            // Compute update gate: z_t = sigmoid(W_z * [x_t, h_{t-1}])
            Quad[] updateResult = this.updateGate.Propagate(value.Values);
            
            // Compute candidate hidden state: h~_t = tanh(W_h * [x_t, r_t ⊙ h_{t-1}])
            // Apply reset gate to previous hidden state
            Quad[] candidateInput;
            if (htPrev != null)
            {
                Quad[] resetHidden = MatrixUtils.MatrixHadamard(resetResult, htPrev);
                // Concatenate x_t with reset-gated hidden state
                candidateInput = new Quad[inputSize + outputSize];
                Array.Copy(xt, 0, candidateInput, 0, inputSize);
                Array.Copy(resetHidden, 0, candidateInput, inputSize, outputSize);
            }
            else
            {
                candidateInput = value.Values;
            }
            
            Quad[] candidateHidden = this.targetGate.Propagate(candidateInput);
            
            // Compute final hidden state: h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ h~_t
            Quad[] ht;
            if (cellState != null)
            {
                // h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ h~_t
                Quad[] updatePrevious = MatrixUtils.MatrixHadamard(updateResult, cellState);
                
                // Compute (1 - z_t)
                Quad[] oneMinusUpdate = new Quad[updateResult.Length];
                for (int i = 0; i < updateResult.Length; i++)
                {
                    oneMinusUpdate[i] = 1.0 - updateResult[i];
                }
                
                Quad[] updateCandidate = MatrixUtils.MatrixHadamard(oneMinusUpdate, candidateHidden);
                
                ht = MatrixUtils.MatrixSum(updatePrevious, updateCandidate);
            }
            else
            {
                ht = candidateHidden;
            }
            
            // Update cell state with new hidden state
            cellState = ht;
            
            // Propagate to next layer
            foreach (var synapse in this.Outputs)
            {
                synapse.Propagate(new NeuronPropagateEventArgs
                {
                    Values = ht,
                    ClearValue = false
                });
            }
        }
    }
}
