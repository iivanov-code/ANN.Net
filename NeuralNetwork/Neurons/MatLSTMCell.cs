using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Utils;
using static ANN.Net.Utils.MatrixUtils;

namespace ANN.Net.Neurons
{
    internal class MatLSTMCell : HiddenNeuron, ILSTMCell
    {
        public MatLSTMCell(int inputs, int outputs, ActivationTypes activationType)
            : base(activationType)
        {
            Wf = MatrixUtils.RandomMatrix(inputs, outputs, 0, 1);
            Wi = CopyMatrix(Wf);
            Wo = CopyMatrix(Wf);
            Wc = CopyMatrix(Wf);
            Uf = MatrixUtils.RandomMatrix(outputs, outputs, 0, 1);
            Ui = CopyMatrix(Uf);
            Uo = CopyMatrix(Uf);
            Uc = CopyMatrix(Uf);
            inputCounter = (ushort)inputs;
        }

        private Quad[,] xt, ft, it, ot, ct, ht; //xt - input vector, ht - output vector
        private Quad[,] Wf, Wi, Wo, Wc; // Weights of: forget input, output, cell state gate
        private Quad[,] Uf, Ui, Uo, Uc; // Weight of prev h gates
        private Quad[,] ht_prev, ct_prev;

        private ushort inputCounter;

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            //base.Backpropagate(0, 0, null);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            //_value += value;
            //Inputs.AccountSignal();
            //if (Inputs.CheckCountAndReset())
            //{
            //    _value = TanHActivation.Activate(ref _value);

            //    xt[xt.Length - inputCounter, 0] = _value;
            //    inputCounter--;
            //    if (inputCounter == 0)
            //    {
            //        ht = ComputeOutputMatrix();

            //        foreach (var val in ht)
            //        {
            //            base.Propagate(val);
            //        }
            //    }
            //}
        }

        private Quad[,] ComputeOutputMatrix()
        {
            ft = MatrixSigmoid(MatrixSum(MatrixProduct(Wf, xt),
             MatrixProduct(Uf, ht_prev)));

            it = MatrixSigmoid(MatrixSum(MatrixProduct(Wi, xt),
            MatrixProduct(Ui, ht_prev)));

            ot = MatrixSigmoid(MatrixSum(MatrixProduct(Wo, xt),
            MatrixProduct(Uo, ht_prev)));

            ct = MatrixSum(MatrixHadamard(ft, ct_prev),
           MatrixHadamard(it, MatrixTanh(MatrixSum(MatrixProduct(Wc, xt),
           MatrixProduct(Uc, ht_prev)))));
            ht = MatrixHadamard(ot, MatrixTanh(ct));

            ht_prev = ht;
            ct_prev = ct;
            return ht;
        }

        public void Propagate(CellPropagateEventArgs value)
        {
            throw new NotImplementedException();
        }
    }
}
