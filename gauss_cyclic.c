#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include "mpi.h"

#DEFINE MAX_SIZE 100


int main() {
	return 0;	
}

double *gauss_cyclic(double **a, double *b) {

	double *x, l[MAX - SIZE], *buf;
	int i,j,k,r, tag=42;
	MPI_Status status;
	
	struct { double val; int node; } z, y;
	
	x = (double *) malloc(n * sizeof(double));
	buf = (double *) malloc((n+1) * sizeof(double));
	
	/* Forward elimination */
	for (k=0 ; k<n-1 ; k++) { 
		r = max_col_loc(a, k);
		z.node = me;
		if (r != -1) 
			z.val = fabs(a[r][k]); 
		else 
			z.val = 0.0;
		MPI_Allreduce(&z, &y, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
		/* Pivot row and row k are on the same processor */
		if (k % p == y.node){ 
			if (k % p == me) {
				if (a[k][k] != y.val) 
					exchange_row(a, b, r, k);
				copy_row(a, b, k, buf);
			}
		}
		/* Pivot row and row k are owned by different processors */
		else if (k % p == me) {
			copy_row(a, b, k, buf);
			MPI_Send(buf+k, n-k+1, MPI_DOUBLE, y.node, tag,	MPI_COMM_WORLD);
		}
		else if (y.node == me) {
			MPI - Recv(buf+k, n-k+1, MPI_DOUBLE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
			copy_exchange_row(a, b, r, buf, k);
		}
		
		MPI_Bcast(buf+k, n-k+1, MPI_DOUBLE, y.node, MPI_COMM_WORLD);
		if ((k % p != y.node) && (k % p == me)) 
			copy_back_row(a, b, buf, k);
		i = k+1; 
		while (i % p != me) 
			i++;
		for (; i<n; i+=p) {
			l[i] = a[i][k] / buf[k];
			for (j=k+1; j<n; j++)
				a[i][j] = a[i][j] - l[i]*buf[j];
			b[i] = b[i] - l[i]*buf[n];
		}
	}
	
	/* Backward substitution */
	for (k=n-1; k>=0; k--) { 
		if (k % p == me) {
			sum = 0.0;
			for (j=k+1; j < n; j++) 
				sum = sum + a[k][j] * x[j];
			x[k] = 1/a[k][k] * (b[k] - sum); 
		}
		MPI_Bcast(&x[k], 1, MPI_DOUBLE, k%p, MPI_COMM_WORLD);
	}
	return x;
}
