using System;
using System.Linq;
using System.Runtime.Remoting.Messaging;

class Functions
{
    public static int GCD(int a, int b)
    {
        while (b != 0)
        {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    public static int LCM(int a, int b)
    {
        if (a == 0 || b == 0)
            return 0;

        return (a / GCD(a, b)) * b;
    }

    public static double[] Softmax(double[] input)
    {
        double max = input.Max();
        double[] expValues = input.Select(val => Math.Exp(val - max)).ToArray();

        double sumExp = expValues.Sum();

        return expValues.Select(val => val / sumExp).ToArray();
    }

    public static double[,] OCR(double[,] grid1, int grid2_size, double treshold = 0)
    {
        int grid1_size = grid1.GetLength(0);

        double[,] grid2 = new double[grid2_size,grid2_size];
        Array.Clear(grid2, 0, grid2.Length);

        int l = grid1_size - 1;
        int r = 0;
        int u = grid1_size - 1;
        int d = 0;

        for (int i = 0; i < grid1_size; i++)
        {
            for (int j = 0; j < grid1_size; j++)
            {
                if (grid1[i, j] > treshold)
                {
                    if (l > i)
                        l = i;
                    if (r < i)
                        r = i;
                    if (u > j)
                        u = j;
                    if (d < j)
                        d = j;
                }
            }
        }

        int x = Functions.LCM(r - l + 1, grid2_size);
        int y = Functions.LCM(d - u + 1, grid2_size);

        int dx = x / (r - l + 1);
        int dy = y / (d - u + 1);

        int dx2 = x / grid2_size;
        int dy2 = y / grid2_size;

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                grid2[i / dx2, j / dy2] += grid1[l + i / dx, u + j / dy];
            }
        }

        for (int i = 0; i < grid2_size; i++)
        {
            for (int j = 0; j < grid2_size; j++)
            {
                grid2[i, j] /= dx2 * dy2;
            }
        }

        return grid2;
    }

}
