#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>		/* fabs */
#include <string.h>		/* strcmp, strlen */
#include "mpi.h"

#include "mmio.h"

#define MAX_SIZE 100

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
double* gauss_cyclic(double **a, double *b, int n);


int main(int argc, char *argv[]) {

    int ret_code;
    MM_typecode matcode;
    FILE *f_input, *f_output;
    int M, N, nz;   
    int i, *I, *J;
    double val;
    int p1, p2;

    if (argc != 7)
	{
		fprintf(stderr, "Usage: mpiexec -n [number-processor] %s -i [input-file] -o [output-file] -g [p1]x[p2] \n", argv[0]);
		exit(1);
	}        
	
	i = 1;
	char* aux = argv[i];
	while(i < 7) {
		//printf("command: \"%s\"\n", argv[i]);
		if(strcmp(aux, "-i") == 0) {
			if ((f_input = fopen(argv[i+1], "r")) == NULL){
				printf("Input file \"%s\" not found.\n", argv[i+1]);
    			exit(1);  
    		}
			else if (mm_read_banner(f_input, &matcode) != 0) {
				printf("Could not process Matrix Market banner.\n");
				exit(1);
			}
			else if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
				    mm_is_sparse(matcode) ) {
				printf("Sorry, this application does not support ");
				printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
				exit(1);
			}
			else if ((ret_code = mm_read_mtx_array_size(f_input, &M, &N)) !=0)
				exit(1);
		}
		else if(strcmp(aux, "-o") == 0) {
			if ((f_output = fopen(argv[i+1], "w")) == NULL) {
				printf("Error with output file.\n");
    			exit(1);
    		}
		}
		else if(strcmp(aux, "-g") == 0) {
			char* aux = argv[i+1];
			int j = 0;
			while(*(aux++) != 'x')
				j++;
			char p[j+1];
			int k = 0;
			while(k < j) {
				p[k] = argv[i+1][k];
				k++;
			}
			p[j] = '\0';
			p1 = atoi(p);

			p2 = atoi(aux);
		}
		i += 2;
		aux = argv[i];

	}
	
	//printf("p1: %d, p2: %d\n", p1, p2);
	
	double* A[M];
    int j, k;
    //double val;
    
    //printf("hellooo, %d %d\n", M, N);
    /* init matrix */
    for (i = 0; i < M; i++) {
    	A[i] = (double * ) malloc(N * sizeof(double));	
    	for (j = 0; j < N; j++) {
    		A[i][j] = 0;
    	}
    }
    
    for (k = 0; k< M*N; k++)
    {
        fscanf(f_input, "  %lg\n", &val);       
        A[k/N][k%N] = val;
    }

	fclose(f_input);
	
	n = M;
	//int i, j, k;
	int myrank, p;
	
	MPI_Init(NULL, NULL);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	if(myrank == 0){
		printf("A: \n");
		for (i = 0; i < M; i++) {    	
			for (j = 0; j < N; j++) {
				fprintf(stdout, "%10.3g", A[i][j]);
			}
			fprintf(stdout, "\n");
		}
    }
	
	double b[M];
	for(i = 0; i < M; i++)
		b[i] = 1;
		
	if(myrank == 0){
		printf("b: \n");
		for(i = 0; i < M; i++)
			printf("%f \n", b[i]);
	}
		
	double* result;
	result = gauss_cyclic(A, b, M);
	
	if(myrank == 0){
		printf("x: \n");
		for(i = 0; i < M; i++)
			printf("%f \n", result[i]);
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
double* gauss_cyclic(double **a, double *b, int n) {

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
		printf("%d -> column %d: \n", me, k);
		//obtain local max abs value between my rows 
		r = max_col_loc(a, k);
		
		//save my rank and the value (if exists)
		z.node = me;
		if (r != -1) {
			printf("%d -> my max val (%f) is in row %d \n", me, fabs(a[r][k]), r);
			z.val = fabs(a[r][k]); 
		}
		else{
			//printf("%d -> no more for me baaabe \n", me);
			z.val = 0.0;
		}
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
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	/* Backward substitution */
	double sum;
	for (k=n-1; k>=0; k--) { 
		if (k % p == me) {
			sum = 0.0;
			for (j=k+1; j < n; j++) 
				sum = sum + a[k][j] * x[j];
			x[k] = 1/a[k][k] * (b[k] - sum); 
		}
		MPI_Bcast(&x[k], 1, MPI_DOUBLE, k%p, MPI_COMM_WORLD);
		//printf("%d: x[%d] = %f \n", me, k, x[k]);
	}
	
	//free(buf);
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
	return index;
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
	
	b[k] = buf[n];
	
	int i = k;
	for (; i < n; i++)
	
		a[k][i] = buf[i];
}


