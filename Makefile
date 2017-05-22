CC=gcc
CFLAGS=-I.
	
example_read: example_read.o mmio.o
	$(CC) -o read example_read.o mmio.o -I.
     

example_write: example_write.o mmio.o
	$(CC) -o write example_write.o mmio.o -I.
	
test: test.o mmio.o
	$(CC) -o test test.o mmio.o -I.
	
gauss_cyclic: gauss_cyclic.c
	mpicc gauss_cyclic.c mmio.c -o gauss.x

     
