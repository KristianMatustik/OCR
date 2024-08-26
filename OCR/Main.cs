﻿using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

class main
{
    [STAThread]
    static void Main(string[] args)
    {
        //trainMNIST();

        Application.EnableVisualStyles();
        Application.Run(new DigitRecognitionWindow(28, 14));
    }

    static void loadMNIST(out List<NeuralNetwork.Data> dataTrain, out List<NeuralNetwork.Data> dataTest)
    {
        dataTrain = new List<NeuralNetwork.Data>();
        using (var reader = new StreamReader("mnist_train.csv"))
        {
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');
                double[] correctOutput = new double[10];
                for (int i = 0; i < 10; i++)
                {
                    correctOutput[i] = 0;
                }
                correctOutput[Int32.Parse(values[0])] = 1;

                double[,] image = new double[28, 28];
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        image[j, i] = Double.Parse(values[1 + i * 28 + j]) / 255;
                    }
                }
                double[,] cropedImage = Functions.OCR(image, 14, 0);
                double[] input = new double[14 * 14];
                for (int i = 0; i < 14; i++)
                {
                    for (int j = 0; j < 14; j++)
                    {
                        input[i * 14 + j] = cropedImage[i, j];
                    }
                }
                dataTrain.Add(new NeuralNetwork.Data(input, correctOutput));
            }
        }

        dataTest = new List<NeuralNetwork.Data>();
        using (var reader = new StreamReader("mnist_test.csv"))
        {
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');
                double[] correctOutput = new double[10];
                for (int i = 0; i < 10; i++)
                {
                    correctOutput[i] = 0;
                }
                correctOutput[Int32.Parse(values[0])] = 1;

                double[,] image = new double[28, 28];
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        image[j, i] = Double.Parse(values[1 + i * 28 + j]) / 255;
                    }
                }
                double[,] cropedImage = Functions.OCR(image, 14, 0);
                double[] input = new double[14 * 14];
                for (int i = 0; i < 14; i++)
                {
                    for (int j = 0; j < 14; j++)
                    {
                        input[i * 14 + j] = cropedImage[i, j];
                    }
                }
                dataTest.Add(new NeuralNetwork.Data(input, correctOutput));
            }
        }
    }

    static void trainMNIST()
    {
        NeuralNetwork nn = new NeuralNetwork(14 * 14, 50, 10);
        loadMNIST(out List<NeuralNetwork.Data> dataTrain, out List<NeuralNetwork.Data> dataTest);

        Console.WriteLine("training started");
        nn.train(dataTrain.ToArray(), 0.96, 1, 0.5, 0, dataTest.ToArray());
        nn.saveToFile("MNIST2.bin");
    }

    static void testMNIST()
    {
        NeuralNetwork nn = new NeuralNetwork(14 * 14, 50, 10);
        nn.loadFromFile("MNIST.bin");
        loadMNIST(out List<NeuralNetwork.Data> dataTrain, out List<NeuralNetwork.Data> dataTest);

        double acc, err;
        nn.testAccuracyMT(dataTrain.ToArray(), out acc, out err, 6);
        Console.WriteLine("Training data acc: " + acc);
        nn.testAccuracyMT(dataTest.ToArray(), out acc, out err, 6);
        Console.WriteLine("Testing data acc: " + acc);
    }

    static void trainXOR()
    {
        NeuralNetwork nn = new NeuralNetwork(2, 2, 2);

        NeuralNetwork.Data[] td = new NeuralNetwork.Data[4];
        td[0] = new NeuralNetwork.Data(new double[] { 0, 0 }, new double[] { 0, 1 });
        td[1] = new NeuralNetwork.Data(new double[] { 1, 0 }, new double[] { 1, 0 });
        td[2] = new NeuralNetwork.Data(new double[] { 0, 1 }, new double[] { 1, 0 });
        td[3] = new NeuralNetwork.Data(new double[] { 1, 1 }, new double[] { 0, 1 });


        nn.train(td.ToArray(), 4, 100000);

        double[] result;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                result = nn.calculateOutput(new double[] { i, j });
                Console.WriteLine(i);
                Console.WriteLine(j);
                foreach (double x in result)
                {
                    Console.WriteLine(x);
                }
            }
        }
        nn.saveToFile("xor.bin");
    }

    static void loadXOR()
    {
        NeuralNetwork nn = new NeuralNetwork(1);
        nn.loadFromFile("xor.bin");
        double[] result;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                result = nn.calculateOutput(new double[] { i, j });
                Console.WriteLine(i);
                Console.WriteLine(j);
                foreach (double x in result)
                {
                    Console.WriteLine(x);
                }
            }
        }
    }

    static void test1()
    {
        NeuralNetwork nn = new NeuralNetwork(2, 2, 2);

        NeuralNetwork.Data d = new NeuralNetwork.Data();
        double[] inputs = { 1, 1 };
        d.input = inputs;
        double[] outputs = nn.calculateOutput(d);

        for (int i = 0; i < outputs.Length; i++)
        {
            Console.WriteLine(outputs[i]);
        }
    }
}