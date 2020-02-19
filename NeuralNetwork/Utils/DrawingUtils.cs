using NeuralNetwork.Interfaces;
using NeuralNetwork.Neurons;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace NeuralNetwork.Utils
{
    public static class DrawingUtils
    {
        public const int X_OFFSET = 5;
        public const int Y_OFFSET = 5;
#if DEBUG
        public static NetworkDrawing GetDrawing(this INetwork network, float width = 1920, float height = 1080)
        {
            Network net = (Network)network;
            BaseNetworkDrawing drawing = new NetworkDrawing();

            int layersCount = net.HiddenLayers.Count + 2;

            int columnsCount = (layersCount * 2) - 1;
            int maxRowsCount = Math.Max(net.InputNeurons.Count, net.OutputNeurons.Count);
            maxRowsCount = Math.Max(maxRowsCount, net.HiddenLayers.Select(x => x.Count).Max()) * 2;

            int circleDiameter = (int)(Math.Min(height, width) - 10) / Math.Max(maxRowsCount, columnsCount);
            Size circleSize = new Size(circleDiameter, circleDiameter);

            int x = X_OFFSET;

            AddLayerNodes(drawing, net.InputNeurons, x, maxRowsCount, circleSize);

            foreach (var layer in net.HiddenLayers)
            {
                x += (int)(width / columnsCount) * 2;
                AddLayerNodes(drawing, layer, x, maxRowsCount, circleSize);
            }

            x += (int)(width / columnsCount) * 2;
            AddLayerNodes(drawing, net.OutputNeurons, x, maxRowsCount, circleSize);

            return drawing as NetworkDrawing;
        }
#else
        public static NetworkDrawing GetDrawing(this Network network, float width = 1920, float height = 1080)
        {
            throw new NotImplementedException("CAN NOT DRAW IMAGE IN RELEASE MODE");
        }
#endif


#if DEBUG
        private static void AddLayerNodes(BaseNetworkDrawing drawing, IEnumerable<INeuron> neurons, int x, int maxNeuronsCount, Size circleSize)
        {
            int y = Y_OFFSET;

            y += (int)Math.Floor((maxNeuronsCount - (double)(neurons.Count() * 2)) / 2) * circleSize.Height;

            int cellHeight = circleSize.Height;

            int half = circleSize.Height / 2;

            foreach (Neuron neuron in neurons)
            {
                var rect = new Rectangle(new Point(x, y), circleSize);

                Point nodeCenterPoint = new Point(x + half, y + half);

                drawing.NodeCircles.Add(new Node(neuron.ID, rect));

                if (neuron.Outputs != null)
                {
                    foreach (ISynapse connection in neuron.Outputs)
                    {
                        var line = new Line(neuron.ID, connection.Output.ID, connection.Weight)
                        {
                            From = nodeCenterPoint
                        };
                        drawing.Connections.Add(line, line);
                    }
                }

                if (neuron.Inputs != null)
                {
                    foreach (ISynapse connection in neuron.Inputs)
                    {
                        var key = new LineKey(connection.Input.ID, neuron.ID);
                        if (drawing.Connections.ContainsKey(key))
                        {
                            drawing.Connections[key].To = nodeCenterPoint;
                        }
                    }
                }

                y += cellHeight * 2;
            }
        }
#endif

    }

    public class NetworkDrawing : BaseNetworkDrawing
    {
        internal NetworkDrawing() : base() { }

        public new IEnumerable<Rectangle> NodeCircles
        {
            get
            {
                return base.NodeCircles.Select(x => x.Circle);
            }
        }

        public new IEnumerable<ILine> Connections
        {
            get
            {
                return base.Connections.Select(x => x.Value);
            }
        }
    }

    public class BaseNetworkDrawing
    {
        public BaseNetworkDrawing()
        {
            this.Connections = new Dictionary<LineKey, ILine>();
            this.NodeCircles = new HashSet<INode>();
        }

        public HashSet<INode> NodeCircles { get; set; }
        public Dictionary<LineKey, ILine> Connections { get; set; }
    }

    public interface INode
    {
        Rectangle Circle { get; set; }
    }

    public class Node : INode
    {
        public Node(Guid id, Rectangle circle)
        {
            this.ID = id;
            this.Circle = circle;
        }

        public Guid ID { get; set; }
        public Rectangle Circle { get; set; }
    }

    public interface ILine
    {
        Point From { get; }
        Point To { get; set; }
        float Weight { get; }
    }

    public class LineKey : IEquatable<LineKey>
    {
        public LineKey(Guid fromId, Guid toId)
        {
            this.FromID = fromId;
            this.ToID = toId;
        }

        public Guid FromID { get; set; }
        public Guid ToID { get; set; }

        public override bool Equals(object obj)
        {
            return Equals((LineKey)obj);
        }

        public bool Equals(LineKey other)
        {
            return this.GetHashCode() == other.GetHashCode();
        }

        public override int GetHashCode()
        {
            return this.FromID.GetHashCode() ^ this.ToID.GetHashCode();
        }

    }

    public class Line : LineKey, ILine
    {
        public Line(Guid fromID, Guid toID, float weight)
            : base(fromID, toID)
        {
            this.Weight = weight;
        }

        public Point From { get; set; }
        public Point To { get; set; }
        public float Weight { get; set; }
    }
}
