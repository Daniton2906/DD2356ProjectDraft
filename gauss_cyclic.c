#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>		/* fabs */
#include "mpi.h"

#DEFINE MAX_SIZE 100

int* vec_rows;
int n = MAX_SIZE;

typedef struct{
	double val;
	int node;
} Pair;

int max_col_loc(double **a, int k);
void exchange_row(double **a, double *b, int r, int k);
void copy_row(double **a, double *b, int k, double *buf);
void copy_exchange_row(double **a, double *b, int r, double *buf, int k);
void copy_back_row(double **a, double *b, double *buf, int k);

int main() {

	n = 10;
	int i, j, k;
	int myrank, p;
	
	MPI_Init(NULL, NULL);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);	
	
	int vec_size = n/p;
	vec_rows = (int *) malloc((vec_size+1) * sizeof(int));
	
	k = myrank;
	for (i = 0; i <= vec_size; i++) {
		if(k < n)
			vec_rows[i] = k;
		else
			vec_rows[i] = -1;
		k += p;
	}
	
	
	MPI_Finalize();
	return 0;
}

/*
	Solve the linear system Ax = b
	a: an (n x n) matrix
	b: vector of size n
	n: size of the linear system

*/
double *gauss_cyclic(double **a, double *b, int n) {

	double *x, l[MAX_SIZE], *buf;
	int i, j, k, r, tag=42;
	int me, p;
	MPI_Status status;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &p);	
	
	Pair z, y;
	
	x = (double *) malloc(n * sizeof(double));
	buf = (double *) malloc((n+1) * sizeof(double));
	
	/* Forward elimination */
	for (k=0 ; k < n - 1 ; k++) {
		//obtain local max abs value between my rows 
		r = max_col_loc(a, k);
		//save my rank and the value (if exists)
		z.node = me;
		if (r != -1) 
			z.val = fabs(a[r][k]); 
		else 
			z.val = 0.0;
		//send the value and receive the max global
		MPI_Allreduce(&z, &y, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
		
		//Several cases:
		/* Pivot row and row k are on the same processor */
		if (k % p == y.node){ //I have the row with the max global value
			if (k % p == me) { //also, I have the row k of the column k
				//then I just exchange my rows :)
				if (a[k][k] != y.val) 
					exchange_row(a, b, r, k);
				copy_row(a, b, k, buf); //copy the pivot row in the buffer
			}
		}
		/* Pivot row and row k are owned by different processors */
		//I have the k row and I will send it to the guy with the max global row
		else if (k % p == me) { 
			copy_row(a, b, k, buf); //copy the k row in the buffer
			MPI_Send(buf+k, n-k+1, MPI_DOUBLE, y.node, tag,	MPI_COMM_WORLD);
		}
		//I have the row with the global max and I will receive the row k
		else if (y.node == me) { 
			MPI_Recv(buf+k, n-k+1, MPI_DOUBLE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
			copy_exchange_row(a, b, r, buf, k); //exchange the buffer with the pivot row
		}
		
		//broadcast the k row, just from the k index, all receive the pivot row
		MPI_Bcast(buf+k, n-k+1, MPI_DOUBLE, y.node, MPI_COMM_WORLD);
		
		//if I am not the owner of the pivot row but I owned the k row
		//copy the buffer in the k row (is the pivot row dude)
		if ((k % p != y.node) && (k % p == me)) 
			copy_back_row(a, b, buf, k);
			
		//search the next row which I own
		i = k+1; 
		while (i % p != me) 
			i++;
			
		//compute the eliminator factor and new arrays a and b
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

int max_col_loc(double **a, int k) {
	int index = -1, i = 0, j = k;
	double max_val = 0.0, aux = 0.0;
	
	int myrank, p;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);	
	
	while (j % p != myrank) 
		j++;
		
	for(; j < n; j += p) {
		
		if( max_val < (aux = fabs(a[k][j]))) {
			max_val = aux;
			index = j;
		}	
		
	}
}

void exchange_row(double **a, double *b, int r, int k) {

	double aux = b[r];
	b[r] = b[k];
	b[k] = aux;
	
	double* row = a[r];
	a[r] = a[k];
	a[k] = a[r];
}

void copy_row(double **a, double *b, int k, double *buf) {
	int i = 0;
	for (; i < n; i++) {
		buf[i] = a[k][i];
	}
	buf[n] = b[k];
}

void copy_exchange_row(double **a, double *b, int r, double *buf, int k) {
	//buffer has the k row, I own the pivot row
	//we want to exchange the data in the pivot row with the data in the buffer
	
	double aux = buf[n];
	buf[n] = b[r];
	b[r] = aux;
	
	int i = k;
	for (; i < n; i++) {
		aux = buf[i];
		buf[i] = a[r][i];
		a[r][i] = aux;
	}
	
}

void copy_back_row(double **a, double *b, double *buf, int k) {
	
}


