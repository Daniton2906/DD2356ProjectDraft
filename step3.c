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

double * gauss_double_cyclic (double **a, double *b, int n);

int Co (int k);
int Ro (int k);
int Cop (int q);
int Rop (int q);
int member (int me, int G, char type);
int grp_leader (int G, char type);
int rank(int q, int G, char type);


int max_col_loc(double** a, int k, int n);
void exchange_row_loc(double** a, double* b, int r, int k);
void copy_row_loc(double** a, double* b, int k, double* buf);
int compute_partner(int G, char type, int me);
int compute_size(int n, int k, int G, char type);
void exchange_row_buf(double** a, double* b, int r, double* buf, int k);
void copy_back_row_loc(double** a, double* b, int k, double* buf);
void compute_elim_fact_loc (double** a, double* b, int k, double* buf, double* elim_buf);
void compute_local_entries(double** a, double* b, int k, double* elim_buf, double* buf);
void backward_substitution(double** a, double* b, double* x);

void printchunk(double** a, int me);
void print_buf(int me, double* buf, int k);

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
//	printf("here: %d!", myrank);
	
	double* result;
	
	if(myrank == 0){
		printf("(%d x %d)\n", M, N);
		printf("((%d, %d)(%d, %d))\n", p1, b2, p2, b2);
		printf("p1 (row processors): %d, p2(column processors): %d\n", p1, p2);
		printf("b1 (row block size): %d, b2(column block processors): %d\n", b1, b2);
	}
	
	int row = myrank / p2;
	//create row group: have p1 row group of size p2 each one
	int row_ranks[p2];
	int global_row_rank = row * p2;
	for(i = 0; i < p2; i++) {
		row_ranks[i] = global_row_rank;
		global_row_rank+=1;
	}
	
	int column = myrank % p2;
	//create column group: have p2 column group of size p1 each one
	int column_ranks[p1];
	int global_column_rank = column;
	for(i = 0; i < p1; i++) {
		column_ranks[i] = global_column_rank;
		global_column_rank+=p2;
	}
	
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	// Construct a group containing all of the row ranks in world
	MPI_Group_incl(world_group, p2, row_ranks, &row_group);
	MPI_Group_incl(world_group, p1, column_ranks, &column_group);

	// Create a new communicator based on the group
	MPI_Comm_create_group(MPI_COMM_WORLD, row_group, 0, &row_comm);
	MPI_Comm_create_group(MPI_COMM_WORLD, column_group, 1, &column_comm);

	if(myrank == 0 && 0){

		
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
	MPI_Barrier(MPI_COMM_WORLD);
	result = gauss_double_cyclic (A, b, M);
    
	//write back the result in the output file
	MPI_Barrier(MPI_COMM_WORLD);
	
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
	
	//row_rank: local rank in the row_group I am part of
	//row_size: number of elements of the row_group I am part of
	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);

	//column_rank: local rank in the column_group I am part of
	//column_size: number of elements of the column_group I am part of
	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);

	//printf("WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d \t COLUMN RANK/SIZE: %d/%d\n", myrank, p, row_rank, row_size, column_rank, column_size);
	
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
	//printf("here: %d !\n", me);

	for (k = 0; k < n - 1; k++) {
	
		//if I am a member of the group which contains the column k
		//I could have the largest absolute value from this column
		
		//printf("here: %d !", me);
		if (member(me, Co(k), 'c')) {
			r = max_col_loc(a, k, n);
			z.pvtline = r; 
			if (r != -1) 
				z.val = fabs(a[r][k]);
			else
				z.val = 0.0;
			if(k == 6 && 0)
				printf("phase %d: processor %d says the max is %f from row %d\n", k, me, z.val, r);
			int leader = grp_leader(Co(k), 'c');
			int s; MPI_Comm_size(row_comm, &s);
			int r; MPI_Comm_rank(column_comm, &r);
			//printf("global_rank: %d -> leader: %d -> Column group: %d of %d\n", me, leader, Cop(me), s);
			MPI_Reduce(&z, &y, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, column_comm); /*comm(Co(k))*/      
			if(me == leader && 0) 
				printf("phase %d, rank: %d -> pivot row: %d\n", k, me, y.pvtline);
		} //otherwise, just have to just the largest value from some on this group
		
		//broadcast the row with the largest absolute value		
		MPI_Bcast(&y, 1, MPI_DOUBLE_INT, grp_leader(Co(k), 'c'), MPI_COMM_WORLD);
		r = y.pvtline;
		
		
		
		/*pivot row and row k are in the same row group*/
		if(Ro(k) == Ro(r)){
			//if I am member of the row group of the row k
			//I have chunks of the k row and the pivot row
			if (member(me, Ro(k), 'r')) {
				//just exchange my rows only if are different rows
				if (r != k) 
					exchange_row_loc(a, b, r, k); //exchange my chunks of data
				//copy on the buffer to send it to others processors
				copy_row_loc(a, b, k, buf); 
				//print_buf(me, buf, k);
			} 
		}
		 /* pivot row and row k are in different row groups */
		else if (member(me, Ro(k), 'r')) { //I have the k row
			//copy my chunk of data in the buffer
			copy_row_loc(a, b, k, buf);
			//print_buf(me, buf, k);
			q = compute_partner(Ro(r), 'r', me);
			//printf("I am %d, my partner is %d who own the pivot row: %d\n", me, q, r);
			psz = compute_size(n, k, Ro(k), 'r');
			//printf("rank %d have to send %d elements to %d\n", me, psz, q);
			MPI_Send(buf+k, psz, MPI_DOUBLE, q, tag, MPI_COMM_WORLD); 
		}
		/* executing processor contains a part of the pivot row */
		else if (member(me, Ro(r), 'r')) { //i have the pivot row
			
			q = compute_partner(Ro(k), 'r', me);
			//printf("I am %d, my partner is %d who own the k row: %d\n", me, q, k);
			psz = compute_size(n, k, Ro(r), 'r');
			//printf("rank %d have to receive %d elements from %d\n", me, psz, q);
			MPI_Recv(buf+k, psz, MPI_DOUBLE, q, tag, MPI_COMM_WORLD, &status);
			//print_buf(me, buf, k);
			exchange_row_buf(a, b, r, buf, k);
		}
		
		//if(member(me, Ro(r), 'c'))
		if(k == 6 && 0)
			printchunk(a, me);

		
		/*
		int myrank = me;

		int column_size;
		MPI_Comm_size(column_comm, &column_size);
	
		int row_size;
		MPI_Comm_size(row_comm, &row_size);
		MPI_Barrier(MPI_COMM_WORLD);
		
		int M = n, N = n;
	
		if(myrank == 0) printf("A: \n");
		
		for (i = 0; i < p; i++) {
			if(member(myrank, i, 'c') && member(myrank, i, 'r')) {
				printf("row %d: ", i);
				for (j = 0; j < b2; j++) {
					fprintf(stdout, "%10.3g", a[i][j]);
				}
				fprintf(stdout, "\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	
		MPI_Barrier(MPI_COMM_WORLD);
	
		if(myrank == 0) printf("b: \n");
		
		for (i = 0; i < n; i++) {
			if(member(myrank, 0, 'c') && member(myrank, i, 'r')){
				fprintf(stdout, "%10.3g", b[i]);
				fprintf(stdout, "\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	
		MPI_Barrier(MPI_COMM_WORLD);
		*/
		/* broadcast of pivot row */
		
		buf_size = compute_size(n, k, Ro(k), 'r');
		//printf("rank %d have to broadcast %d elements, root %d\n", me, buf_size, Ro(r));
		MPI_Bcast(buf+k, buf_size, MPI_DOUBLE, Ro(r), column_comm);
		
		
		//print_buf(me, buf, k);
		//if the k row and pivot row are in different row groups
		//and I hadthe k row
		if ((Ro(k) != Ro(r)) && (member(me, Ro(k), 'r')))
			copy_back_row_loc(a, b, k, buf);
			
		if (member(me, Co(k), 'c')) 
			compute_elim_fact_loc(a, b, k, buf, elim_buf);
			
		/* broadcast of elimination factors */
			
		elim_size = compute_size(n, k, Co(k), 'c');
		//elim_size = n - k - 1;
		elim_size--;
		if(Cop(me) == Co(k) && 0)
			printf("phase(%d) rank %d column %d, broadcast %d elements, root %d\n", k, me, Cop(me), elim_size, Co(k));
		MPI_Bcast(elim_buf, elim_size, MPI_DOUBLE, Co(k), row_comm);
		//print_buf(me, elim_buf, k);	
		compute_local_entries(a, b, k, elim_buf, buf); 
		if(Rop(me) == Ro(k)){

			int row_rank, column_size;
			MPI_Comm_rank(row_comm, &row_rank);
			MPI_Comm_size(column_comm, &column_size);

			int i, end = (row_rank + 1) * b2; 	
	
			i = row_rank * b2;
			for(; i < end; i++) {
				printf("phase(%d) rank: %d, a[%d][%d] = %f\n", k, me, k, i, a[k][i]);	
			}
		}
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
	int row_size;
	MPI_Comm_size(row_comm, &row_size);
	return (k / b2) % row_size;
}


/*
	Ro:
		int k: row k
		
	return the row group which own the row k

*/
int Ro (int k) {
	//printf("k = %d, b1 = %d, p1 = %d\n", k, b1, p1);
	int column_size;
	MPI_Comm_size(column_comm, &column_size);
	return (k / b1) % column_size;

}

/*
	Cop:
		int q: id processor
		
	return the column group of the given processor
*/
int Cop(int q) {
	int row_size;
	MPI_Comm_size(row_comm, &row_size);
	return q % row_size;
}

/*member(me, Ro(k), 'r')MPI_COMM_WORLDcopy_back_row_loc(a, b, k, buf);
	Rop:
		int q: id processor
		
	return the row group of the given processor
*/
int Rop(int q) {
	int row_size;
	MPI_Comm_size(row_comm, &row_size);
	return q / row_size;
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
	grp_leader:	
		int G: number of the group
		char type: column group or row group
		
	return the group leader of the given group

*/
int grp_leader(int G, char type) {
	int column_size;
	MPI_Comm_size(column_comm, &column_size);
	
	int row_size;
	MPI_Comm_size(row_comm, &row_size);
	
	int leader = 0;
	if (type == 'c'){
		//printf("row size: %d \n", row_size);
		while(!member(leader, G, 'c') && leader < row_size)
			leader++;
		return leader;
	}
	else if (type == 'r') {
		//printf("column size: %d \n", column_size);
		while(!member(leader, G, 'r') && leader < row_size*column_size)
			leader+=row_size;
		return leader;
	}
	else
		return -1;
}

/*
	rank:
		int q: id processor
		int G: number of group
		char type: column group or row group
		
	return the rank of the given processor in the given group

*/
int rank(int q, int G, char type) {
	int p;
	MPI_Comm_size(row_comm, &p);
	if (type == 'c'){
		return q / p;
	}
	else if (type == 'r'){
		return q % p;
	}
	else
		return -1;
}

/*
	max_col_loc 
	
		double** a: (n x n) matrix
		int k: column
	
		find in the column k the value with the
		greastet absolute value
		only search between the rows the processor owns
		return the row of this value

*/
int max_col_loc(double** a, int k, int n) {

	int index = -1, i = 0, j;
	double max_val = 0.0, aux = 0.0;
	
	int myrank, p;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	int row_group = Rop(myrank), end = (row_group + 1) * b1;
	j = row_group * b1 < k? k: row_group * b1;
//	if(k == 6)
	//	printf("rank %d, init: %d, fin: %d\n", myrank, j, end);	
	for(; j <  end; j++) {
	//	if(k == 6)
		//	printf("rank %d, j: %d, a[%d][%d]: %f\n", myrank, j, j, k, a[j][k]);	
		aux = fabs(a[j][k]);

		if( max_val < aux) {
			max_val = aux;
			index = j;
		}
	}
	
	return index;
}


/*
	exchange_row_loc:
	
		double** a: (n x n) matrix 
		double* b: vector of size n
		int r: row r
		int k: row k
	member(me, Ro(k), 'r')
		Both rows owns to the processor
		exchange the row k and the row r in the matrix a and the vector b
		exchange only the chunks of data the processor owns

*/
void exchange_row_loc(double** a, double* b, int r, int k) {
/*
	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);

	int j, end = (row_rank + 1) * b2;
	j = row_rank * b2 < k ? k : row_rank * b2;
	*/
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int j, end = (Cop(rank) + 1) * b2;
	j = Cop(rank) * b2 < k ? k : Cop(rank) * b2;
	
	double aux = b[r];
	b[r] = b[k];
	b[k] = aux;
	
	for(j; j < end; j++) {
		//printf("asdsad\n");
		aux = a[r][j];
		a[r][j] = a[k][j];
		a[k][j] = aux;
	}
}

/*
	copy_row_loc:
	
		double** a: (n x n) matrix 
		double* b: vector of size n
		int k: row k
		double* buf: buffer with size n
	
		copy the chunk of data of row k of the matrix a
		and the element k of vector b in the buffer

*/
void copy_row_loc(double** a, double* b, int k, double* buf) {
	
	/*
	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);

	int j, end = (row_rank + 1) * b2;
	j = row_rank * b2 < k ? k : row_rank * b2;
*/
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int j, end = (Cop(rank) + 1) * b2;
	j = Cop(rank) * b2 < k ? k : Cop(rank) * b2;

	buf[end] = b[k];
	int i = 0;
	for(j; j < end; j++) {
		buf[k + i] = a[k][j];
		i++;
	}
}

/*
	exchange_row_buf:
	
		double** a: (n x n) matrix 
		double* b: vector of size n
		int r: row r
		double* buf: buffer with size n
		int k: index k
		
		exchange the data on the buffer with the data in the row r in the matrix a
		and the element r in the vector r
		The exchange starts from the minimum between index k 
		and the start index of the chunk of data

*/
void exchange_row_buf(double** a, double* b, int r, double* buf, int k) {
	/*
	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);

	int j, end = (row_rank + 1) * b2;
	
	j = row_rank * b2 < k? k: row_rank * b2;
	*/
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int j, end = (Cop(rank) + 1) * b2;
	j = Cop(rank) * b2 < k ? k : Cop(rank) * b2;
	
	double aux = buf[end];
	buf[end] = b[r];
	b[r] = aux;
	
	int i = 0;
	for(; j < end; j++) {
		aux = buf[k + i];
		buf[k + i] = a[r][j];
		a[r][j] = aux;
		i++;
	}
}

/*
	copy_back_row_loc:
	
		double** a: (n x n) matrix 
		double* b: vector of size n
		int k: row k
		double* buf: buffer with size n
	
		copy the chunk of data of row k of the matrix a
		and the element k of vector b in the buffer (the opposite now :P)

*/
void copy_back_row_loc(double** a, double* b, int k, double* buf) {
	
	/*
	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);

	int j, end = (row_rank + 1) * b2;
	j = row_rank * b2 < k ? k : row_rank * b2;
*/
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int j, end = (Cop(rank) + 1) * b2;
	j = Cop(rank) * b2 < k ? k : Cop(rank) * b2;

	b[k] = buf[end];
	int i = 0;
	for(j; j < end; j++) {
		a[k][j] = buf[k + i];
		i++;
	}
}

/*
	compute_partner:
	
		int G: group number
		char type: column ('c') or row ('r') group
		int me: id processor
		
		Depends on the case:
		type == c -> return the processor in the column group G
				which belong the same row group as me processor  
		type == r -> return the processor in the row group G
				which belong the same column group as me processor

*/
int compute_partner(int G, char type, int me) {

	int row_size, column_size;
	MPI_Comm_size(row_comm, &row_size);
	MPI_Comm_size(column_comm, &column_size);
	
	if (type == 'c') {
		int row_group = Rop(me);
		return row_group * row_size + G;
	}
	else if (type == 'r') {
		int col_group = Cop(me);
		return G * row_size + col_group;
	}
	else
		return -1;
}

/*
	compute_size:
	
		int n: size of the matrix
		int k: k row
		int G: group number
		char type: column ('c') or row ('r') group
		
		Compute the chunk size from the minimum between the k index column/row
		and the start index of the chunk of data
		until the index of the next processor

*/
int compute_size(int n, int k, int G, char type) {

	int rank, size = 1, j = -1, end = -1;

	if (type == 'c') {
		MPI_Comm_rank(column_comm, &rank);
//		MPI_Comm_size(column_comm, &size);
			
		j = rank * b1, end = (rank + 1) * b1;

	}
	else if (type == 'r') {
		MPI_Comm_rank(row_comm, &rank);
//		MPI_Comm_size(row_comm, &size);
		
		j = rank * b2, end = (rank + 1) * b2;
	}
	else 
		return -1;
		
	j = j < k? k : j;
	size += end - j > 0? end - j: 0;
	//size += (n - j);
		
	return size;
}

/*
	compute_elim_fact_loc:
		double** a: (n x n) matrix 
		double* b: vector of size n
		int k: column k
		double* buf: buffer with size n + 1
		double* l: eliminaton buffer with size n + 1

*/
void compute_elim_fact_loc (double** a, double* b, int k, double* buf, double* elim_buf) {

	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int i, end = (column_rank + 1) * b1; 
	
	i = column_rank * b1 < k+1? k+1: column_rank * b1;
	//printf("rank %d -> i = %d end= %d\n", rank, i, end);
	int j = 0;
	for (;i < end; i++) {
		elim_buf[j] = a[i][k] / buf[k];	
		//printf("phase(%d) row (%d) -> elim fact = %f\n", k, i, elim_buf[j]);	
		b[i] = b[i] - elim_buf[j]*buf[end];
		j++;
	}
}

/*


*/
void compute_local_entries(double** a, double* b, int k, double* elim_buf, double* buf) {

	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);

	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);

	int i, end = (column_rank + 1) * b1; 	
	int j, end2 = (row_rank + 1) * b2; 
	
	i = column_rank * b1 < k+1? k+1: column_rank * b1;
	j = row_rank * b2 < k+1? k+1: row_rank * b2;
	
	int me;
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	
	//printf("i = %d end= %d, j = %d end2 = %d\n", i, end,j,end2 );
	int h = j;
	int buf_index = b2 - (end2 - j), ebuf_index = 0;
	for (; i < end; i++) {
		//printf("phase(%d) rank %d, row (%d)\n", k, me, i);
		buf_index = b2 - (end2 - j);
		for (j = h; j < end2; j++){
			//printf("phase(%d) rank %d, column (%d)\n", k, me, j);
			printf("phase(%d) ->>> rank: %d, a[%d][%d] = %f - %f*%f \n", k, me, i, j, a[i][j], elim_buf[ebuf_index], buf[k + buf_index]);
			a[i][j] = a[i][j] - elim_buf[ebuf_index]*buf[k+ buf_index];	
			printf("phase(%d) ->>> rank: %d, a[%d][%d] = %f \n", k, me, i, j, a[i][j]);
			buf_index++;
		}
		ebuf_index++;
	}
}


/*



*/
void backward_substitution(double** a, double* b, double* x) {
		
	/* Backward substitution */
	int i, j, k, q, ql, buf_size, end;
	
	int row_rank, column_rank;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_rank(column_comm, &column_rank);
	
	int me;
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	
	double sum, sum_partial;
	for (k = n - 1; k >= 0; k--) {
	
		if (member(me, Ro(k), 'r')) {
			j = column_rank * b1 < k + 1? column_rank * b1: k + 1;
			end = (column_rank + 1) * b1; 
			sum = 0.0;
			for (j = k + 1; j < end; j++){ 
				sum = sum + a[k][j] * x[j];

			}
			MPI_Reduce(&sum_partial, &sum, 1, MPI_DOUBLE, MPI_SUM, Co(k), row_comm);
			
			if(member(me, Co(k), 'c'))
				x[k] = 1/a[k][k] * (b[k] - sum); 
			
			MPI_Bcast(&x[k], 1, MPI_DOUBLE, Co(k), row_comm);
			
		}
		
		MPI_Bcast(&x[k], 1, MPI_DOUBLE, Ro(k), column_comm);
	}
}

void printchunk(double** a, int me) {
	StringBuffer sb2;		
	init_string_buffer(&sb2);
	
	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);

	int column_rank, column_size;
	MPI_Comm_rank(column_comm, &column_rank);
	MPI_Comm_size(column_comm, &column_size);

	int i, end = (Rop(me) + 1) * b1; 	
	int j, end2 = (Cop(me) + 1) * b2; 
	
	i = Rop(me) * b1;
	j = Cop(me) * b2;
	
	int h = j;
	appendRank(me, &sb2);
	for (; i < end; i++) {

		//appendString("\n", &sb2);
		appendString("row ", &sb2);
		appendInt(i, &sb2);
		appendString(": ", &sb2);
		for (j = h; j < end2; j++){
			appendFloat(a[i][j], 5, &sb2);
			appendString(" ", &sb2);
		}
		appendString("\n", &sb2);
	}
	
	printf("%s \n", sb2.buffer);
}


void print_buf(int me, double* buf, int k) {

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	StringBuffer sb2;		
	init_string_buffer(&sb2);

	appendRank(me, &sb2);
	appendString("phase(", &sb2);
	appendInt(k,&sb2 );
	appendString(") buf: ", &sb2);
	appendFloatArray(buf, n+1, &sb2);
	appendString("\n", &sb2);
	printf("%s \n", sb2.buffer);
}





































