/**
 * Code from the MIT course "Performance engineering of software systems"
 *
 * Modified for Python3 by Moreno Marzolla
 * Last modified 2024-10-14 by Moreno Marzolla
 */

import java.util.Random;
import java.io.*;

public class matmul_test {
    final static int n = 4096;

    static double[][] p = new double[n][n];
    static double[][] q = new double[n][n];
    static double[][] r = new double[n][n];

    public static void main(String[] args)
    {
        Random rnd = new Random();

        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                p[i][j] = rnd.nextDouble();
                q[i][j] = rnd.nextDouble();
                r[i][j] = 0;
            }
        }

        System.out.printf("Matrix-Matrix multiplication (Java), %d x %d\n\n", n, n);

        final long start = System.nanoTime();
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                for (int k=0; k<n; k++) {
                    r[i][j] += p[i][k] * q[k][j];
                }
            }
        }
        final double elapsed = (System.nanoTime() - start) * 1e-9;
        final double Gflops = 2 * (n/1000.0) * (n/1000.0) * (n/1000.0) / elapsed;

        System.out.printf("      Time\t    Gflops\n");
        System.out.printf("----------\t----------\n");
        System.out.printf("%10.3f\t%10.3f\n", elapsed, Gflops);
    }
}
