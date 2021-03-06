#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>		/* fabs */

#DEFINE MAX_SIZE 100



int max_col_loc(double **a, int k, int n);
void exchange_row(double **a, double *b, int r, int k, int n);
double *gauss_sequential (double **a, double *b, int n);


int main() {
	return 0;	
}


double *gauss_sequential (double **a, double *b, int n) {
	double *x, sum, l[MAX_SIZE];
	int i, j, k, r;
	
	x = (double *) malloc(n * sizeof(double));
	for (k = 0; k < n - 1; k++) { /* Forward elimination */
		r = max_col(a, k, n);
		if (k != r)
			exchange_row(a, b, r, k, n);
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

int max_col_loc(double **a, int k, int n) {
	int index = -1, i = k;
	double max_val = 0.0, aux = 0.0;
	for (; i < n; i++) {
		if( max_val < (aux = fabs(a[k][i]))) {
			max_val = aux;
			index = i;
		}	
	}
	return index;
}

void exchange_row(double **a, double *b, int r, int k, int n) {

	double aux = b[r];
	b[r] = b[k];
	b[k] = aux;
	
	double* row = a[r];
	a[r] = a[k];
	a[k] = a[r];
}
