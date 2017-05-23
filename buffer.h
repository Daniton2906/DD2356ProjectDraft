/*
BUFFER FUNCTIONS
JUST FOR DEBUGGING 
*/


#define BUFFER_SIZE 2048

typedef struct 
{
	int index;
	int size;
	char buffer[BUFFER_SIZE + 1];
} StringBuffer;

void init_string_buffer(StringBuffer* sb);
void reset_string_buffer(StringBuffer* sb);
void appendRank(int rank, StringBuffer* sb);
void appendIntArray(int* array, int length, StringBuffer* sb);
void appendFloatArray(double* array, int length, StringBuffer* sb);
void appendString(char* string, StringBuffer* sb);
void appendInt(int number, StringBuffer* sb);
void appendFloat(double number, int precision, StringBuffer* sb);
void appendIntMatrix(int* array, int n, int m, StringBuffer* sb);
void appendFloatMatrix(double** array, int n, int m, StringBuffer* sb);
