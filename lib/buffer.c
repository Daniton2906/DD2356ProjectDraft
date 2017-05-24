#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "buffer.h"

void init_string_buffer(StringBuffer* sb) {
	sb->index = 0;
	(sb->buffer)[sb->index] = '\0';
	sb->size = BUFFER_SIZE;
}

void reset_string_buffer(StringBuffer* sb) {
	sb->index = 0;
	(sb->buffer)[sb->index] = '\0';
}


int lenNumber(int number) {
	if(number == 0) return 1;
	if(number < 0) number *= -1;
	int n = 0 , aux = 1;
	while(number >= aux) {
		n++;
		aux *= 10;
	}
	return n;
}

int append(char chr, StringBuffer* sb, int offset) {
	if(sb->index  + offset < sb->size){
		(sb->buffer)[sb->index + offset] = chr;
		return 1;
	}
	return 0;
}

int appendChar(char chr, StringBuffer* sb) {
	if(sb->index < sb->size){
		(sb->buffer)[sb->index] = chr;
		(sb->index)++;
		return 1;
	}
	return 0;
}

void appendInt(int number, StringBuffer* sb) {
	int n = lenNumber(number), aux;
	if(number < 0){
		appendChar('-', sb);
		number *= -1;
	}
	int i = n - 1, j = 0;
	while(i >= 0) {
		aux = number%10;
		if(append((char) (48 + aux), sb, i))
			j++;
		number /= 10;
		i--;
	}
	(sb->index) += j;
	(sb->buffer)[sb->index] =  '\0';
}

void appendFloat(double number, int precision, StringBuffer* sb) {
	if(number < 0) { 
		appendChar('-', sb);
		number *= -1;
	}
	//printf("number: %f, ", number);
	int auxInt = (int) number;
	//printf("intPart: %d, ", auxInt);
	double auxDec = number - auxInt;
	//printf("decPart: %f \n", auxDec);
	appendInt(auxInt, sb);
	appendChar('.', sb);
	int k = 0;
	while(k++ < precision) {
		auxDec *= 10;
		auxInt = (int) auxDec;
		appendInt(auxInt, sb);
		auxDec -= auxInt;
	}
	//(sb->buffer)[sb->index] =  '\0';
}

void appendString(char* string, StringBuffer* sb) {
	int n = strlen(string);
	int i = 0;
	while(i < n) {
		appendChar(string[i], sb);
		i++;
	}
	(sb->buffer)[sb->index] = '\0';
}

void appendIntArray(int* array, int length, StringBuffer* sb) {
	int i = 0;
	while(i < length) {
		appendInt(array[i], sb);
		appendString(" ", sb);
		i++;
	}
}

void appendFloatArray(double* array, int length, StringBuffer* sb) {
	int i = 0;
	while(i < length) {
		appendFloat(array[i], 5, sb);
		appendString(" ", sb);
		i++;
	}
}

void appendIntMatrix(int* array, int n, int m, StringBuffer* sb) {
	int i = 0;
	while(i < n) {
		int j;
		for (j = 0; j < m; j++){
			appendInt((int) array[i*m + j], sb);
			appendString(" ", sb);
		}
		appendString("\n", sb);
		i++;
	}
}

void appendFloatMatrix(double** array, int n, int m, StringBuffer* sb) {
	int i = 0;
	while(i < n) {
		int j;
		for (j = 0; j < m; j++){
			appendFloat(array[i][j], 5, sb);
			appendString(" ", sb);
		}
		appendString("\n", sb);
		i++;
	}
}

void appendRank(int rank, StringBuffer* sb) {
	appendString("[Rank ", sb);
	appendInt(rank, sb);
	appendString("] \n", sb);	
}
