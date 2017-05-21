CC=gcc
CFLAGS=-I.
	
example_read: example_read.o mmio.o
	$(CC) -o read example_read.o mmio.o -I.
     

example_write: example_write.o mmio.o
	$(CC) -o write example_write.o mmio.o -I.


     
