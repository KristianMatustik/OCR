using System;
using System.Drawing;
using System.Windows.Forms;

class DigitRecognitionWindow : Form
{
    private NeuralNetwork nn;
    private int cellSize1 = 20;
    private int cellSize2 = 40;
    private int grid1_size;
    private int grid2_size;
    private double[,] grid1;
    private double[,] grid2;
    private double penThickness = 15;

    private int drawing = 2; //2 not drawing, 1 drawing, 0 erasing
    System.Windows.Forms.TextBox tb = new System.Windows.Forms.TextBox();

    public DigitRecognitionWindow(int s1 = 1, int s2 = 1, int m = 1200, int n = 900)
    {
        InitializeComponent(s1, s2, m, n);
        this.Paint += new PaintEventHandler(paint);
    }

    private void InitializeComponent(int s1, int s2, int m, int n)
    {
        grid1_size = s1;
        grid2_size = s2;
        grid1 = new double[grid1_size, grid1_size];
        grid2 = new double[grid2_size, grid2_size];
        nn=new NeuralNetwork(1);
        nn.loadFromFile("MNIST.bin");
        resetGrids();
        SuspendLayout();
        ClientSize = new System.Drawing.Size(m, n);
        Name = "Digit Recognition";
        ResumeLayout(false);
        tb.Multiline = true;
        tb.Size = new Size(400, 400);
        tb.Font = new Font(tb.Font.FontFamily, 16);
        tb.Location = new Point(10, cellSize1 * grid1_size + 10);
        tb.Enabled = false;
        this.Controls.Add(tb);
    }

    private void resetGrids(bool g1 = true, bool g2 = true)
    {
        if (g1)
        {
            for (int i = 0; i < grid1_size; i++)
            {
                for (int j = 0; j < grid1_size; j++)
                {
                    grid1[i, j] = 0;
                }
            }
        }
        if (g2)
        {
            for (int i = 0; i < grid2_size; i++)
            {
                for (int j = 0; j < grid2_size; j++)
                {
                    grid2[i, j] = 0;
                }
            }
        }
    }

    private void setGrid(int x, int y, double value)
    {
        if (x < 0 || y < 0 || x > grid1_size - 1 || y > grid1_size - 1)
            return;
        if (drawing==0)
        {
            grid1[x, y] = 0;
            return;
        }
        if (grid1[x, y] > value)
        {
            return;
        }
        grid1[x,y] = value;
    }

    private double intensity(double x1, double y1, double x2, double y2)
    {
        double distance = Math.Sqrt(Math.Pow(x1 - x2, 2) + Math.Pow(y1 - y2, 2));
        if (distance < penThickness) { return 1; }
        else if (2 * penThickness - distance > 0) { return (2 * penThickness - distance) / penThickness; }
        else { return 0; }
    }

    private void paint(object sender, PaintEventArgs e)
    {
        Graphics gr = e.Graphics;
        for (int i = 0; i < grid1_size; i++)
        {
            for (int j = 0; j < grid1_size; j++)
            {
                System.Drawing.Drawing2D.GraphicsPath gp = new System.Drawing.Drawing2D.GraphicsPath();
                Rectangle rect = new Rectangle(i * cellSize1, j * cellSize1, cellSize1, cellSize1);
                gp.AddRectangle(rect);
                System.Drawing.Region r = new System.Drawing.Region(gp);
                int c = (int)(grid1[i, j] * 255);
                gr.FillRegion(new SolidBrush(Color.FromArgb(255, c, c, c)), r);
            }
        }

        for (int i = 0; i < grid2_size; i++)
        {
            for (int j = 0; j < grid2_size; j++)
            {
                System.Drawing.Drawing2D.GraphicsPath gp = new System.Drawing.Drawing2D.GraphicsPath();
                Rectangle rect = new Rectangle(i * cellSize2 + cellSize1*(grid1_size+1), j * cellSize2, cellSize2, cellSize2);
                gp.AddRectangle(rect);
                System.Drawing.Region r = new System.Drawing.Region(gp);
                int c = (int)(grid2[i, j] * 255);
                gr.FillRegion(new SolidBrush(Color.FromArgb(255, c, c, c)), r);
            }
        }

        string s = "";
        double[] input = new double[grid2_size * grid2_size];
        for (int i = 0; i < grid2_size; i++)
        {
            for (int j = 0; j < grid2_size; j++)
            {
                input[i * grid2_size + j] = grid2[i, j];
            }
        }
        double[] output = nn.calculateOutput(input);
        double[] prob = nn.probabilities(output);

        for (int i = 0;i < prob.Length; i++)
        {
            double max = 0;
            int iMax = 0;
            for (int j = 0; j<prob.Length; j++)
            {
                if (prob[j] > max)
                {
                    max = prob[j];
                    iMax = j;
                }
            }
            s += iMax + " : " + max*100 + "% "+"\r\n";
            prob[iMax] = 0;
        }
        tb.Text = s;
    }


    protected override void OnMouseMove(MouseEventArgs e)
    {
        if (drawing!=2)
        {
            int x = (int)((double)e.Location.X / cellSize1);
            int y = (int)((double)e.Location.Y / cellSize1);
            
            int min = (int)Math.Floor(-2*penThickness / (double)cellSize1);
            int max = (int)Math.Ceiling(2 * penThickness / (double)cellSize1);
            for (int i = x + min; i < x + max + 1; i++)
            {
                for (int j = y + min; j < y + max + 1; j++)
                {
                    setGrid(i, j, drawing*intensity(i * cellSize1 + cellSize1 / 2, j * cellSize1 + cellSize1 / 2, e.Location.X, e.Location.Y));
                }
            }

            grid2 = Functions.OCR(grid1,grid2_size,0);
            double[] input = new double[grid2_size*grid2_size];
            for(int i=0; i<grid2_size; i++)
            {
                for (int j=0;j<grid2_size; j++)
                {
                    input[i * grid2_size + j] = grid2[i, j];
                }
            }

            Console.WriteLine(nn.classify(nn.calculateOutput(input)));
            Invalidate();
        }
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        if (e.Button == MouseButtons.Left)
            drawing = 1;
        else if (e.Button == MouseButtons.Right)
            drawing = 0;
        OnMouseMove(e);
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        drawing = 2;
        OnMouseMove(e);
    }

    protected override void OnKeyUp(KeyEventArgs e)
    {
        base.OnKeyUp(e);
        if (e.KeyCode == Keys.Space)
        {
            resetGrids();
            Invalidate();
        }
    }
}