CC=gcc
CFLAGS=-I.
showMatrix = -DDEBUG_MATRIX=1
showResult = -DDEBUG_RESULT=1 
	
example_read: example_read.o mmio.o
	$(CC) -o read example_read.o mmio.o -I.

example_write: example_write.o mmio.o
	$(CC) -o write example_write.o mmio.o -I.
	
test: test.o mmio.o
	$(CC) -o test test.o mmio.o -I.
	
test2: test2.o mmio.o buffer.o
	$(CC) -o test2 test2.o mmio.o buffer.o -I.	
	
step2_d0: step2.c
	mpicc step2.c mmio.c buffer.c -o step2.x $(showResult)
	
step2_d1: step2.c
	mpicc step2.c mmio.c buffer.c -o step2.x $(showMatrix) $(showResult)
	
step2_d2: step2.c
	mpicc step2.c mmio.c buffer.c -o step2.x -DDEBUG_PROCESSOR=1 $(showResult)
	
step2_d12: step2.c
	mpicc step2.c mmio.c buffer.c -o step2.x $(showMatrix) -DDEBUG_PROCESSOR=1 $(showResult)

     
