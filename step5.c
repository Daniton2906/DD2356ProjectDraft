#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>		/* fabs */
#include <string.h>		/* strcmp, strlen */
#include "mpi.h"
#include "omp.h"

#include "lib/mmio.h"
#include "lib/buffer.h"

#define MAX_SIZE 100

#ifndef DEBUG_FULL
	#define DEBUG_FULL  0
#endif

#ifndef DEBUG_RESULT
	#define DEBUG_RESULT  0
#endif

#ifndef DEBUG_MATRIX
	#define DEBUG_MATRIX  0
#endif

#ifndef DEBUG_PROCESSOR
	#define DEBUG_PROCESSOR  0
#endif

int* vec_rows;
int n = MAX_SIZE;
StringBuffer sb;

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

	//if didn't receive 6 arguments
    if (argc != 7)
	{
		fprintf(stderr, "Usage: mpiexec -n [number-processor] %s -i [input-file] -o [output-file] -g [p1]x[p2] \n", argv[0]);
		exit(1);
	}        
	
	//Set inputfile, outputfile and amount of processors used by columns and rows
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
	
	//Initialize matrix A by reading input file
	double* A[M];
    int j, k;
    init_string_buffer(&sb);
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
        A[k%N][k/N] = val;
    }

	fclose(f_input);
	
	//Initialize b = [1, 1, ..., 1]
	double b[M];
	for(i = 0; i < M; i++)
		b[i] = 1;
	
	//Initialize MPI
	n = M;
	//int i, j, k;
	int myrank, p;
	
	MPI_Init(NULL, NULL);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);	
	
	//Print A and b, debbuging
	if(myrank == 0 && (DEBUG_MATRIX || DEBUG_FULL)){
		printf("A: \n");
		for (i = 0; i < M; i++) {    	
			for (j = 0; j < N; j++) {
				fprintf(stdout, "%10.3g", A[i][j]);
			}
			fprintf(stdout, "\n");
		}
		
		printf("b: \n");
		for(i = 0; i < M; i++)
			printf("%f \n", b[i]);
			
		//appendRank(myrank, &sb);
    }
    
	//test(A,b,n);
		
	//calculate solution linear system Ax = b
	
	
	double* result;
	result = gauss_cyclic(A, b, M);

	
	//write back the result in the output file
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(myrank == 0){
		mm_write_banner(f_output, matcode);
	    mm_write_mtx_array_size(f_output, M, N);
		printf("x: \n");

		for(i = 0; i < M; i++) {		
			fprintf(f_output,"%f \n", result[i]);
			printf("%f \n", result[i]);
		}
	}

	
	
	
	if(DEBUG_MATRIX) {
	
		MPI_Barrier(MPI_COMM_WORLD);
	
		if(myrank == 0) printf("Final A: \n");
		
		for (i = 0; i < M; i++) {
			if(i % p == myrank){
				printf("row %d: ", i);
				for (j = 0; j < N; j++) {
					fprintf(stdout, "%10.3g", A[i][j]);
				}
				fprintf(stdout, "\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	
		MPI_Barrier(MPI_COMM_WORLD);
	
		if(myrank == 0) printf("Final b: \n");
		
		for (i = 0; i < M; i++) {
			if(i % p == myrank){
				fprintf(stdout, "%10.3g", b[i]);
				fprintf(stdout, "\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	
		MPI_Barrier(MPI_COMM_WORLD);
	
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

	double *x, l[n], *buf;
	int i, j, k, r, tag=42;
	int me, p;
	MPI_Status status;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &p);	
	
	StringBuffer sb2;
	
	init_string_buffer(&sb2);
	
	//appendRank(me, &sb2);
	
	Pair z, y;
	
	x = (double *) malloc(n * sizeof(double));
	buf = (double *) malloc((n+1) * sizeof(double));
	
	/* Forward elimination */
	for (k=0 ; k < n - 1 ; k++) {
		if (DEBUG_PROCESSOR)
			printf("%d -> column %d: \n", me, k);
		//obtain local max abs value between my rows 
		r = max_col_loc(a, k);
		
		//save my rank and the value (if exists)
		z.node = me;
		if (r != -1) {
			if (DEBUG_PROCESSOR)
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
			
			//printf("%d has k row (%d) and pivot row (%d)\n", me, k, r);
		}
		/* Pivot row and row k are owned by different processors */
		//I have the k row and I will send it to the guy with the max global row
		else if (k % p == me) {
			copy_row(a, b, k, buf); //copy the k row in the buffer
			MPI_Send(buf+k, n-k+1, MPI_DOUBLE, y.node, tag,	MPI_COMM_WORLD);
			//printf("%d has k row (%d), sending to %d\n", me, k, y.node);
		}
		//I have the row with the global max and I will receive the row k
		else if (y.node == me) { 
			MPI_Recv(buf+k, n-k+1, MPI_DOUBLE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
			//printf("%d has pivot row (%d), reciving k row (%d)\n", me, r, k);
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
			
		//compute the eliminator factor for my rows
		//and new arrays a and b for my rows
		//PARALELIZABLE
		for (; i<n; i+=p) {
			l[i] = a[i][k] / buf[k];
			for (j=k+1; j<n; j++)
				a[i][j] = a[i][j] - l[i]*buf[j];
			b[i] = b[i] - l[i]*buf[n];
		}
		
		
			  
		//MPI_Barrier(MPI_COMM_WORLD);
	}
	
	/* Backward substitution */
	double sum;
	//PARALELIZABLE
	for (k=n-1; k>=0; k--) { 
		if (k % p == me) {
			sum = 0.0;
			for (j=k+1; j < n; j++){ 
				sum = sum + a[k][j] * x[j];
			}	
			x[k] = 1/a[k][k] * (b[k] - sum); 
		}
		MPI_Bcast(&x[k], 1, MPI_DOUBLE, k%p, MPI_COMM_WORLD);
	}
	
	//free(buf);
	return x;
}

/*
	max_col_loc 
	
		double** a: (n x n) matrix
		int k: column
	
		find in the column k the value with the
		greastet absolute value, return the row of this value


*/
int max_col_loc(double **a, int k) {
	int index = -1, i = 0, j = k;
	double max_val = 0.0, aux = 0.0;
	
	int myrank, p;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);	
	
	while (j % p != myrank) 
		j++;
	
	//PARALELIZABLE
	for(; j < n; j += p) {
		aux = fabs(a[j][k]);
		//printf("%d: |max_val| = |%f| < |a[%d][%d]| = |%f|?", myrank, max_val, j, k, aux);
		if( max_val < aux) {
			//printf("--> yep\n");
			max_val = aux;
			index = j;
		}
		//else		
			//printf("--> nop\n");
		
	}
	return index;
}

/*
	exchange_row:
	
		double** a: (n x n) matrix 
		double* b: vector of size n
		int r: row r
		int k: row k
	
		exchange the row k and the row r 
			in the matrix a and the vector b
	
*/
void exchange_row(double **a, double *b, int r, int k) {

	double aux = b[r];
	b[r] = b[k];
	b[k] = aux;
	
	double* row = a[r];
	a[r] = a[k];
	a[k] = row;
}

/*
	copy_row:
	
		double** a: (n x n) matrix 
		double* b: vector of size n
		int k: row k
		double* buf: buffer with size n
	
		copy the row k of the matrix a 
		and the element k of vector b in the buffer

*/
void copy_row(double **a, double *b, int k, double *buf) {
	int i = 0;
	//PARALELIZABLE
	for (; i < n; i++) {
		buf[i] = a[k][i];
	}
	buf[n] = b[k];
}

/*
	copy_exchange_row:
	
		double** a: (n x n) matrix 
		double* b: vector of size n
		int r: row r
		double* buf: buffer with size n + 1
		int k: index k
		
		exchange the data on the buffer with the data in the row r in the matrix a
		and the element r in the vector r
		The exchange starts from the index k

*/
void copy_exchange_row(double **a, double *b, int r, double *buf, int k) {
	//buffer has the k row, I own the pivot row
	//we want to exchange the data in the pivot row with the data in the buffer
	
	double aux = buf[n];
	buf[n] = b[r];
	b[r] = aux;
	
	int i = k;
	//PARALELIZABLE
	for (; i < n; i++) {
		aux = buf[i];
		buf[i] = a[r][i];
		a[r][i] = aux;
	}
	
}

/*
	copy_back_row:
	
		double** a: (n x n) matrix 
		double* b: vector of size n
		int k: row k
		double* buf: buffer with size n + 1
		
		Do the opposite of copy_row, i.e., copy the data of the buffer
		in the row k of the matrix a and in the element k of the vector b 
		The copy back starts from the index k

*/
void copy_back_row(double **a, double *b, double *buf, int k) {
	
	b[k] = buf[n];
	
	int i = k;
	//PARALELIZABLE
	for (; i < n; i++)
		a[k][i] = buf[i];
}


