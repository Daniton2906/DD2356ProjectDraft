#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <math.h>		/* fabs */
#include <string.h>		/* strcmp, strlen */
#include "mpi.h"

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

int n = MAX_SIZE;
StringBuffer sb;
int p1, p2, b1, b2;

MPI_Comm row_comm, column_comm;
MPI_Group row_group, column_group;

typedef struct{
	double val;
	int pvtline;
} Pair;

int Co (int k);
int Ro (int k);
int member (int me, int G,  char type);
int Cop (int q);
int Rop (int q);
int grp_leader (int G, char type);
int rank(int q, int G, char type, int p);

double * gauss_double_cyclic (double **a, double *b, int n);

int max_col_loc(double** a, int k);
void exchange_row_loc(double** a, double* b, int r, int k);
void copy_row_loc(double** a, double* b, int k, double* buf);
int compute_partner(int G, char type, int me);

void exchange_row_buf(double** a, double* b, int r, double* buf, int k);

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
	n = M;
	//calculate b1 (row data block) and b2 (column data blocks)
	//virtual processor mesh has size b1xb2
	b1 = n / p1;
	b2 = n / p2;
	
	
	
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
	if(myrank == 0 && (1 || DEBUG_MATRIX || DEBUG_FULL)){
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
	
	MPI_Comm_split(MPI_COMM_WORLD, myrank % p2, myrank / p2, &row_comm);
	
	MPI_Comm_group (row_comm, &row_group);
	
	MPI_Comm_split(MPI_COMM_WORLD, myrank / p1, myrank % p1, &column_comm);
	
	MPI_Comm_group (column_comm, &column_group);
	
	if(myrank == 0 && 0){
		printf("((%d, %d)(%d, %d))\n", p1, b2, p2, b2);
		printf("p1 (row processors): %d, p2(column processors): %d\n", p1, p2);
		printf("b1 (row block size): %d, b2(column block processors): %d\n", b1, b2);
		
		//test Co, Ro
		for(i = 0; i < n; i++) {
			printf("Co(%d) = %d, Ro(%d) = %d\n", i, Co(i), i, Ro(i));	
		}
		
		//test member, Rop, Cop
		for(k = 0; k < p1*p2; k++) {
			
			for(j = 0; j < p1; j++) {
				if(member(k, j, 'r'))
					printf("proc %d is member of group R(%d)\n", k, j);	
			}
			
			for(j = 0; j < p2; j++) {
				if(member(k, j, 'c'))
					printf("proc %d is member of group C(%d)\n", k, j);	
			}
		}
		
		//test grp_leader
		for(j = 0; j < p1; j++) {
			printf("the group leader of group R(%d) is proc %d\n", j, grp_leader (j, 'r'));	
		}
		
		for(j = 0; j < p2; j++) {
			printf("the group leader of group C(%d) is proc %d\n", j, grp_leader (j, 'c'));	
		}
		
	}
	
	result = gauss_double_cyclic (A, b, M);
	
	//write back the result in the output file
	MPI_Barrier(MPI_COMM_WORLD);
	/*
	if(myrank == 0){
		mm_write_banner(f_output, matcode);
	    mm_write_mtx_array_size(f_output, M, N);

		appendString("x: \n", &sb);

		for(i = 0; i < M; i++) {
			appendFloat(result[i], 5, &sb);
			appendString("\n", &sb);		
			fprintf(f_output,"%f \n", result[i]);
		}
		
		
	}
	printf("%s \n", sb.buffer);
	
	*/
	/*
	int dims[2]; dims[0] = dims[1] = 1;
	int periods[2]; periods[0] = periods[1] = 1;
	MPI_Comm grid_comm;
	
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0 )
	*/
	
	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);
	
	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);

	printf("WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d \t COLUMN RANK/SIZE: %d/%d\n", myrank, p, row_rank, row_size, column_rank, column_size);
	
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


double * gauss_double_cyclic (double **a, double *b, int n) {
	double *x, *buf, *elim_buf;
	int i, j, k, r, q, ql, size, buf_size, elim_size, psz;
	int tag = 42;
	
	int me, p;
	
	Pair z, y;
	
	MPI_Status status;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	x = (double *) malloc(n * sizeof(double));
	buf = (double *) malloc((n+1) * sizeof(double));
	
	elim_buf = (double *) malloc((n+1) * sizeof(double));
	
	for (k = 0; k < n - 1; k++) {
	
		//if I am a member of the group which contains the column k
		//I could have the largest absolute value from this column
		//AQUUIII QUEDE
		if (member(me, Co(k), 'c')) {
			r = max_col_loc(a, k);
			z.pvtline = r; 
			z.val = fabs(a[r][k]);
			int leader = grp_leader(Co(k), 'c');
			MPI_Reduce(&z, &y, 1, MPI_DOUBLE_INT, MPI_MAXLOC, leader, /*comm(Co(k))*/ column_comm));
		} //otherwise, just have to just the largest value from some on this group
		
		//broadcast the row with the largest absolute value
		MPI_Bcast(&y, 1, MPI_DOUBLE_INT, grp_leader(Co(k), 'c'), MPI_COMM_WORLD);
		r = y.pvtline;
		
		if(Ro(k) == Ro(r)){
			/*pivot row and row k are in the same row group */
			if (member(me, Ro(k), 'r')) {
				if (r != k) 
					exchange_row_loc(a, b, r, k);
				copy_row_loc(a, b, k, buf); 
			} 
		}
		 /* pivot row and row k are in different row groups */
		else if (member(me, Ro(k), 'r')) {
			copy_row_loc(a, b, k, buf);
			q = compute_partner(Ro(r), me);
			psz = compute_size(n, k, Ro(k));
			MPI_Send(buf+k, psz, MPI_DOUBLE, q, tag, MPI_COMM_WORLD); 
		}
		else if (member(me,Ro(r), 'r')) {
			/* executing processor contains a part of the pivot row */
			q = compute_partner(Ro(k), me);
			psz = compute_size(n, k, Ro(r));
			MPI_Recv(buf+k, psz, MPI_DOUBLE, q, tag, MPI_COMM_WORLD, &status) ;
			exchange_row_buf(a, b, r, buf, k);
		}
		
		for (q=0; q<p; q++) /* broadcast of pivot row */
			if (member(q, Ro(r), 'r') && member(me, Cop(q), 'c')) {
				ql = rank(q, Cop(q)); 
				buf_size = compute_size(n,k,Ro(k));
				MPI_Bcast(buf+k, buf_size, MPI_DOUBLE, ql, comm(Cop(q)));
			}
			
		if ((Ro(k) != Ro(r)) && (member(me, Ro(k)))
			copy_row_loc(a, b, buf, k);
			
		if (member(me,Co(k))) 
			elim_buf = compute_elim_fact_loc(a, b, k, buf);
			
		for (q=0; q<p; q++) /* broadcast of elimination factors */
			if (member(q,Co(k)) && member(me,Rop(q))) {
				ql = rank(q,Rop(q)); 
				elim_size = compute_size(n,k,Co(k));
				MPI_Bcast(elim_buf, elim_size, MPI_DOUBLE, ql, comm(Rop(q))); 
			}
			
		compute_local_entries(a, b, k, elim_buf, buf); 
	
	}
	backward_substitution(a,b,x);
	return x;
}


/*
	Co:
		int k: column k
		
	return the column group which own the column k

*/
int Co (int k) {
	//printf("k = %d, b2 = %d, p2 = %d\n", k, b2, p2);
	int column_size;
	MPI_Comm_size(column_comm, &column_size);
	return (k / b2) % column_size;
}


/*
	Ro:
		int k: row k
		
	return the row group which own the row k

*/
int Ro (int k) {
	//printf("k = %d, b1 = %d, p1 = %d\n", k, b1, p1);
	int row_size;
	MPI_Comm_size(row_comm, &row_size);
	return (k / b1) % row_size;

}

/*
	member:
		int me: id processor
		int G: number of group
		char type: column group or row group
		
	return 1 if the processor belong to the given group, otherwise return 0

*/
int member(int me, int G, char type) {
	if (type == 'c')
		return Cop(me) == G;
	else if (type == 'r')
		return Rop(me) == G;
	else
		return 0;
}

/*
	grp_leader:	int index = -1, i = 0, j = k;
	double max_val = 0.0, aux = 0.0;
	
	int myrank, p;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
		int G: number of the group
		char type: column group or row group
		int p1: number of row processors
		int p2: number of column processors
		
	return the group leader of the given group

*/
int grp_leader(int G, char type) {
	int column_size;
	MPI_Comm_size(column_comm, &column_size);
	
	int row_size;
	MPI_Comm_size(row_comm, &row_size);
	
	int leader = 0;
	if (type == 'c'){
		while(!member(leader, G, 'c') && leader < row_size)
			leader++;
		return leader;
	}
	else if (type == 'r') {
		while(!member(leader, G, 'r') && leader < row_size*column_size)
			leader+=column_size;
		return leader;
	}
	else
		return -1;
}

/*
	Cop:
		int q: id processor
		
	return the column group of the given processor
*/
int Cop(int q) {
	int column_size;
	MPI_Comm_size(column_comm, &column_size);
	return q % column_size;
}

/*
	Rop:
		int q: id processor
		int p1: number of row processor
		
	return the row group of the given processor
*/
int Rop(int q) {
	int row_size;
	MPI_Comm_size(row_comm, &row_size);
	return q / row_size;
}

/*
	rank:
		int q: id processor
		int G: number of group
		char type: column group or row group
		int p: number of row/column processors
		
	return the rank of the given processor in the given group

*/
int rank(int q, int G, char type) {
	int p;
	if (type == 'c'){
		MPI_Comm_size(row_comm, &p);
		return q / p;
	}
	else if (type == 'r'){
		MPI_Comm_size(column_comm, &p);
		return q % p;
	}
	else
		return -1;
}

/*



*/
int max_col_loc(double** a, int k) {

	int index = -1, i = 0, j;
	double max_val = 0.0, aux = 0.0;
	
	int myrank, p;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	int column_size;
	MPI_Comm_size(column_comm, &column_size);
	
	int row = Ro(myrank);
	j = row * column_size;
	
	for(; j < (row + 1) * column_size && j < n; j++) {
		aux = fabs(a[j][k]);

		if( max_val < aux) {

			max_val = aux;
			index = j;
		}
	}
	
	return index;
}

void exchange_row_loc(double** a, double* b, int r, int k) {
	
	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);

	int j = column_rank * b2, end = (column_rank + 1) * b2;
	
	double aux = b[r];
	b[r] = b[k];
	b[k] = aux;
	
	for(j; j < end && j < n; j++) {
		aux = a[r][j];
		a[r][j] = a[k][j];
		a[k][j] = aux;
	}
}

void copy_row_loc(double** a, double* b, int k, double* buf) {
	
	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);

	int j = column_rank * b2, end = (column_rank + 1) * b2;
	
	while (j < k)
		j++;
		
	buf[n] = b[k];
	 
	for(j; j < end && j < n; j++) {
		buf[j] = a[k][j];
	}
}


int compute_partner(int G, char type, int me) {

	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);
	
	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);
	
	if (type == 'c') {
		int row_group = Cop(me);
		return row_group * column_size + G;
	}
	else if (type == 'r') {
		int col_group = Cop(me);
		return G * column_size + col_group;
	}
	else
		return -1;
}

int compute_size(int n, int k, int G, char type) {

	int rank, size;
	
	int size = 0, j = -1, end = -1;

	if (type == 'c') {
		MPI_Comm_rank(column_comm, &rank);
		MPI_Comm_size(column_comm, &size);
			
		j = rank * b2, end = (rank + 1) * b2;

	}
	else if (type == 'r') {
		MPI_Comm_rank(row_comm, &rank);
		MPI_Comm_size(row_comm, &size);
		
		j = rank * b1, end = (rank + 1) * b1;
	}
	else 
		return -1;
		
	while (j < k)
		j++;
		
	for(j; j < end && j < n; j++)
		size++;
		
	return size;
}

void exchange_row_buf(double** a, double* b, int r, double* buf, int k) {
	
	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);

	int j = column_rank * b2, end = (column_rank + 1) * b2;
	
	double aux = buf[n];
	buf[n] = b[r];
	b[r] = aux;
	
	while (j < k)
		j++;
	
	for(j; j < end && j < n; j++) {
		aux = buf[i];
		buf[i] = a[r][i];
		a[r][i] = aux;
	}
}

/*


*/
MPI_Comm comm(int G, char type, Co(k), p2) {
	
	int size;
	if(type == 'c')
		MPI_Comm_size(column_comm, &size);
	else if(type == 'r')
		MPI_Comm_size(row_comm, &size);
	else
		return MPI_COMM_NULL;
		
	int ranks[size];
	
	// Construct a group containing all of the prime ranks in world_group
	MPI_Group group;
	MPI_Group_incl(world_group, 7, ranks, &prime_group);

	// Create a new communicator based on the group
	MPI_Comm prime_comm;
	MPI_Comm_create_group(MPI_COMM_WORLD, prime_group, 0, &prime_comm);
}














































