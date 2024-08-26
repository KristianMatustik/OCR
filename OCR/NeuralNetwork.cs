using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading;

[Serializable]
class NeuralNetwork
{
    private Layer[] Layers;

    public NeuralNetwork(params int[] numNeuronsInLayer)
    {
        Layers = new Layer[numNeuronsInLayer.Length-1];
        for (int i = 0; i<numNeuronsInLayer.Length-1; i++)
        {
            Layers[i] = new Layer(numNeuronsInLayer[i], numNeuronsInLayer[i + 1]);
        }
        initializationNormal();
    }

    public double[] calculateOutput(Data data)
    {
        double[] outputs = data.input;
        for (int i=0; i<Layers.Length; i++)
        {
            outputs = Layers[i].calculateOutput(outputs);
        }
        return outputs;
    }

    public double[] calculateOutput(double[] inputs)
    {
        for (int i = 0; i < Layers.Length; i++)
        {
            inputs = Layers[i].calculateOutput(inputs);
        }
        return inputs;
    }

    public double[] probabilities(double[] output)
    {
        double sum = 0;
        for (int i=0; i < output.Length; i++)
        {
            sum += output[i];
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = output[i]/sum;
        }
        return output;
    }

    public int classify(double[] output)
    {
        double max = 0;
        int indexMax = 0;
        for(int i = 0;i<output.Length;i++)
        {
            if (output[i]>max)
            {
                max = output[i];
                indexMax = i;
            }
        }
        return indexMax;
    }

    public void saveToFile(string filename)
    {
        try
        {
            using (FileStream fs = new FileStream(filename, FileMode.Create))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(fs, this);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error saving the neural network to file: " + ex.Message);
        }
    }

    public bool loadFromFile(string filename)
    {
        NeuralNetwork loadedNetwork = null;
        try
        {
            using (FileStream fs = new FileStream(filename, FileMode.Open))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                loadedNetwork = (NeuralNetwork)formatter.Deserialize(fs);
            }
            this.Layers = loadedNetwork.Layers;
            return true;
        }
        catch (FileNotFoundException)
        {
            Console.WriteLine("File not found: " + filename);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error loading the neural network from file: " + ex.Message);
        }
        return false;
    }

    public void testAccuracy(Data[] testData, out double accuracy, out double error)
    {
        accuracy = 0;
        error = 0;
        double[] output;
        for (int i = 0; i<testData.Length; i++)
        {
            output = calculateOutput(testData[i]);
            error += testData[i].cost(output);
            if(testData[i].correctOutput[classify(output)]==1)
            {
                accuracy++;
            }
        }
        accuracy /= testData.Length;
        error /= testData.Length;
    }

    
    public void testAccuracyMT(Data[] testData, out double accuracy, out double error, int numThreads = 1)
    {
        if (numThreads < 1 || numThreads > 10)
            numThreads = 1;

        double[] acc = new double[numThreads];
        double[] err = new double[numThreads];
        Thread[] threads = new Thread[numThreads];
        accuracy = 0;
        error = 0;

        int n = testData.Length/numThreads;
        int remainingData = testData.Length % numThreads;

        for (int i = 0;i<numThreads;i++)
        {
            int threadDataLength = (i == numThreads - 1) ? n + remainingData : n;

            acc[i] = 0;
            err[i] = 0;

            Data[] threadData = new Data[threadDataLength];

            int index = i;
            Array.Copy(testData, i*n, threadData, 0, threadDataLength);
            threads[i] = new Thread(() => testAccuracy(threadData, out acc[index], out err[index]));
        }

        foreach (Thread thread in threads)
        {
            thread.Start();
        }
        foreach (Thread thread in threads)
        {
            thread.Join();
        }

        for (int i=0;i<numThreads-1;i++)
        {
            accuracy += acc[i]*n;
            error += err[i]*n;
        }
        accuracy += acc[numThreads - 1]*(n+remainingData);
        error += err[numThreads - 1] * (n + remainingData);
        accuracy /= testData.Length;
        error /= testData.Length;
    }
    

    public void train(Data[] trainingData, double requiredAcc=0.8, int numInBatch=-1, double learningRate = 0.01, double rho = 0, Data[] testingData=null)
    {
        if (numInBatch<1 || numInBatch>trainingData.Length)
            numInBatch = trainingData.Length;
        if (requiredAcc > 1 || requiredAcc<0)
            requiredAcc = 0.8;
        if (learningRate < 0 || learningRate > 1)
            learningRate = 0.01;
        if (rho < 0 || rho > 1)
            rho = 0;

        Data[] batch = new Data[numInBatch];
        int n = trainingData.Length;
        Stopwatch sw = Stopwatch.StartNew();
        double acc = 0;
        double err = 0;
        int epochs = 0;
        double minLR = learningRate/128;

        testAccuracyMT(testingData, out acc, out err, 6);
        Console.WriteLine("Epoch " + epochs + ". Test accuracy: {0:N4}%. Error: {1:N4}.\n", acc * 100, err);
        while (acc<requiredAcc)
        {
            long t = sw.ElapsedMilliseconds;
            epochs++;           
            for (int j = 0; j < n / numInBatch; j++)
            {
                for (int k = 0; k<numInBatch; k++)
                {
                    batch[k] = trainingData[j*numInBatch+k];
                }
                trainIteration(batch, learningRate, rho);
            }

            double oldErr = err;
            testAccuracyMT(testingData, out acc, out err, 6);
            Console.WriteLine("Epoch " + epochs + ". Test accuracy: {0:N4}%. Error: {1:N5}. Used LR: {2:N5}", acc * 100, err, learningRate);
            Console.WriteLine("Last epoch training time: {0:N2} seconds. Total: {1:N2} seconds\n", (double)(sw.ElapsedMilliseconds - t) / 1000, (double)sw.ElapsedMilliseconds / 1000);
            if (oldErr<err)
            {
                learningRate /= 2;
            }
            if (learningRate < minLR)
               break;
        }
    }

    private void trainIteration(Data[] data, double learningRate = 0.01, double rho = 0)
    {
        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i].clearGradient(rho);
        }
        for (int i = 0; i < data.Length; i++)
        {
            updateGradient(data[i]);
        }
        for (int j = 0; j < Layers.Length; j++)
        {
            Layers[j].applyGradients(learningRate / data.Length);
        }
    }


    private void updateGradient(Data data)
    {
        double[] output = calculateOutput(data);

        double[] nodeValues = Layers[Layers.Length - 1].outputLayerNodeValues(data.correctOutput);
        Layers[Layers.Length - 1].updateGradients(nodeValues);

        for (int i = Layers.Length - 2; i > -1; i--)
        {
            nodeValues = Layers[i].hiddenLayerNodeValues(Layers[i + 1], nodeValues);
            Layers[i].updateGradients(nodeValues);
        }
    }

    private void initializationNormal(double muW=0, double stdW=1, double muB = 0, double stdB = 1)
    {
        Random rg = new Random(0);      //change seed, default for testing
        for (int i=0; i<Layers.Length;i++)
        {
            int nIn = Layers[i].numNodesIn;
            int nOut = Layers[i].numNodesOut;

            double u1, u2, rn;

            for (int j=0; j<nOut; j++)
            {
                for(int k=0; k<nIn; k++)
                {
                    u1 = rg.NextDouble();
                    u2 = rg.NextDouble();
                    rn = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    Layers[i].weights[k, j] = muW+rn*stdW;
                }
                u1 = rg.NextDouble();
                u2 = rg.NextDouble();
                rn = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                Layers[i].biases[j] = muB+rn*stdB;
            }
        }
    }

    ////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
    ////////////////////////////////////////////////////

    [Serializable]
    private class Layer
    {
        public int numNodesIn;
        public int numNodesOut;

        public double[,] weights;
        public double[] biases;

        public double[,] weightsGradient;
        public double[] biasesGradient;

        public double[] input;
        public double[] wInput;
        public double[] output;

        public Layer(int numIn, int numOut)
        {
            numNodesIn = numIn;
            numNodesOut = numOut;

            input = new double[numIn];
            wInput = new double[numOut];
            output = new double[numOut];

            weights = new double[numIn, numOut];
            biases = new double[numOut];

            weightsGradient = new double[numIn, numOut];
            biasesGradient = new double[numOut];
        }

        public double[] calculateOutput(double[] input)
        {
            this.input = input;
            double[] wInput = new double[numNodesOut]; ///addded for MT of testing accuracy
            double[] output = new double[numNodesOut]; ///MT
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                wInput[nodeOut] = biases[nodeOut];
                for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
                {
                    wInput[nodeOut] += weights[nodeIn,nodeOut] * input[nodeIn];
                }
                output[nodeOut] = activationFunction(wInput[nodeOut]);
            }
            this.wInput = wInput; ///MT
            this.output = output; ///MT
            return output;
        }

        public void updateGradients(double[] nodeValues)
        {
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
                {
                    weightsGradient[nodeIn, nodeOut] += input[nodeIn] * nodeValues[nodeOut]; //weight derivative WRT cost
                }
                biasesGradient[nodeOut] += 1 * nodeValues[nodeOut]; //biast derivative WRT cost
            }
        }

        public double[] outputLayerNodeValues(double[] correctOutput)
        {
            double[] nodeValues= new double[numNodesOut];
            for (int i=0;i<correctOutput.Length;i++)
            {
                nodeValues[i] = activationFunctionDerivative(wInput[i]) * nodeCostDerivative(output[i], correctOutput[i]);
            }
            return nodeValues;
        }

        public double[] hiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues)
        {
            double[] newNodeValues = new double[numNodesOut];
            double newNodeValue;
            for (int i = 0; i < newNodeValues.Length; i++)
            {
                newNodeValue = 0;
                for (int j=0;j<oldNodeValues.Length;j++)
                {
                    newNodeValue += oldLayer.weights[i, j] * oldNodeValues[j];
                }
                newNodeValues[i] = newNodeValue* activationFunctionDerivative(wInput[i]);  
            }
            return newNodeValues;
        }
      
        public void applyGradients(double learningRate=0.01)
        {
            for (int i=0;i<numNodesOut;i++)
            {
                biases[i] -= biasesGradient[i] * learningRate;
                for (int j=0;j<numNodesIn;j++)
                {
                    weights[j,i]  -= weightsGradient[j,i]*learningRate;
                }
            }
        }

        public void clearGradient(double rho=0)
        {
            for (int i = 0; i < numNodesOut; i++)
            {
                biasesGradient[i] *= rho;
                for (int j = 0; j < numNodesIn; j++)
                {
                    weightsGradient[j, i] *= rho;
                }
            }
        }
    }

    ////////////////////////////////////////////////////

    [Serializable]
    public class Data
    {
        public double[] input;
        public double[] correctOutput;

        public Data(double[] input=null, double[] correctOutput=null)
        {
            this.input = input;
            this.correctOutput = correctOutput;
        }

        public double cost(double[] predictedOutputs)
        {
            double cost = 0;
            for (int i = 0; i < correctOutput.Length; i++)
            {
                cost += nodeCost(predictedOutputs[i], correctOutput[i]);
            }
            return cost;
        }
    }


    ////////////////////////////////////////////////////

    public static double activationFunction(double input)
    {
        //SIGMOID
        return 1 / (1 + System.Math.Exp(-input));

        //cRELU leaky
        //if (input < 0)
        //    return 0.01 * input;
        //if (input < 1)
        //    return input;
        //return 1;

        //TANH
        //return Math.Tanh(input);
    }

    public static double activationFunctionDerivative(double input)
    {
        //SIGMOID
        double act = activationFunction(input);
        return act * (1 - act);

        //cRELU leaky
        //if (input < 0 || input > 1)
        //    return 0.01;
        //return 1;

        //TANH
        //return 1-Math.Pow(Math.Tanh(input),2);
    }

    /*
    public static double activationFunctionOL(double input)
    {
        return input > 0 ? input : 0; //zmenit na softmax
    }

    public static double activationFunctionDerivativeOL(double input)
    {
        return input > 0 ? 1 : 0; //zmenit na softmax
    }
    */

    public static double nodeCost(double output, double correctOutput)
    {
        return (output - correctOutput)*(output - correctOutput);
    }

    public static double nodeCostDerivative(double output, double correctOutput)
    {
        return 2 * (output - correctOutput);
    }
}

