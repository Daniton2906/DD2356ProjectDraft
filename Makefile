CC=mpicc
LIB =lib

showMatrix = -DDEBUG_MATRIX=1
showResult = -DDEBUG_RESULT=1 
	
step2_d0: step2.c
	$(CC) step2.c $(LIB)/mmio.c $(LIB)/buffer.c -o step2.x $(showResult)
	
step2_d1: step2.c
	$(CC) step2.c $(LIB)/mmio.c $(LIB)/buffer.c -o step2.x $(showMatrix) $(showResult)
	
step2_d2: step2.c
	$(CC) step2.c $(LIB)/mmio.c $(LIB)/buffer.c -o step2.x -DDEBUG_PROCESSOR=1 $(showResult)
	
step2_d12: step2.c
	$(CC) step2.c $(LIB)/mmio.c $(LIB)/buffer.c -o step2.x $(showMatrix) -DDEBUG_PROCESSOR=1 $(showResult)
	
step3_d0: step3.c
	$(CC) step3.c $(LIB)/mmio.c $(LIB)/buffer.c -o step3.x $(showResult)

.PHONY: clean

clean:
	rm -f *~ *.o *.x
