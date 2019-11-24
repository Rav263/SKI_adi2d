#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N  2050//(2*2*2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;

double A [N][N];

double relax();
void init();
void verify(); 


void print_matrix(){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%10lf ", A[i][j]);
        }
        printf("\n");
    }
}

int compare(const void *a, const void *b){
    return (*(double *)a < *(double *)b);
}

int main(int an, char **as)
{
	init();

    printf("NUM OF THREADS: %d\n", omp_get_num_threads());
	for (int it = 1; it <= itmax; it++) {
        double eps = relax();
        printf( "it=%4i eps=%f\n", it, eps);
		if (eps < maxeps) break;
	}

	verify();

	return 0;
}


void init()
{ 
	for(int i = 1; i < N - 1; i++) {
        #pragma omp for schedule(static)
        for(int j = 1; j < N - 1; j++) {
            A[i][j] = (1. + i + j) ;
        }
    }
} 

double relax() {
    double eps = 0;
    {
	    for(int i = 1; i <= N - 2; i++) {
            for(int j = 1; j <= N - 2; j++) {
		        A[i][j] = (A[i - 1][j] + A[i + 1][j]) / 2.;
	        }
        }

        #pragma omp parallel for
	    for(int i = 1; i <= N - 2; i++) {

	        for(int j = 1; j <= N - 2; j++) {
		        double e = A[i][j];
		    
                A[i][j] = (A[i][j - 1] + A[i][j + 1]) / 2.;
		        
                eps = Max(eps, fabs(e - A[i][j]));
            }
	    }
    }
    return eps;
}

void verify()
{ 
	double s = 0.;

    #pragma omp parallel for reduction (+:s)
	for(int i = 0; i <= N - 1; i++) {
	    for(int j = 0; j <= N - 1; j++) {
		    s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
        }
	}
	printf("  S = %f\n",s);
}
