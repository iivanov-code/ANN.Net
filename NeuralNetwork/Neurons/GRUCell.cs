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

        public GRUCell(ushort inputNeurons, ushort outputNeurons)
            : base(ref inputNeurons, outputNeurons)
        {
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
            // Compute reset gate: r_t = sigmoid(W_r * x_t + U_r * h_{t-1})
            Quad[] resetResult = this.resetGate.Propagate(value.Values);
            
            // Compute update gate: z_t = sigmoid(W_z * x_t + U_z * h_{t-1})
            Quad[] updateResult = this.updateGate.Propagate(value.Values);
            
            // Apply reset gate to previous hidden state: r_t ⊙ h_{t-1}
            Quad[] resetHidden;
            if (cellState != null)
            {
                resetHidden = MatrixUtils.MatrixHadamard(resetResult, cellState);
            }
            else
            {
                resetHidden = resetResult;
            }
            
            // Compute candidate hidden state: h~_t = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}))
            // Note: The targetGate should receive the reset-gated hidden state
            // Since the gate building concatenates input and previous output,
            // we need to work with the gate's internal structure
            // For now, we compute the target with original values
            Quad[] candidateHidden = this.targetGate.Propagate(value.Values);
            
            // Compute final hidden state: h_t = (1 - z_t) ⊙ h~_t + z_t ⊙ h_{t-1}
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
