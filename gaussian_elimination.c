#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */

#DEFINE MAX_SIZE 100



int main() {
	return 0;	
}

double *gauss_sequential (double **a, double *b) {
	double *x, sum, l[MAX_SIZE];
	int i, j, k, r;
	
	x = (double *) malloc(n * sizeof(double));
	for (k = 0; k < n - 1; k++) { /* Forward elimination */
		r = max_col(a, k);
		if (k != r)
			exchange_row(a, b, r, k);
		for (i = k + 1; i < n; i++) {
			l[i] = a[i][k] / a[k][k];
			for(j = k + 1; j < n; j++)
				a[i][j] = a[i][j] - l[i] * a[k][j];
			b[i] = b[i] - l[i] * b[k];
		}
	}
	
	for (k = n - 1; k >= 0; k--) { /* Backward elimination */
		sum = 0.0;
		for (j = k + 1; j < n; j++)
			sum = sum + a[k][j] * x[j];
		x[k] = 1 / a[k][k] * (b[k] - sum);
	}
	
	return x;
}
