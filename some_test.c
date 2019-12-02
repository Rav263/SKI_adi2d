#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))
#define  N  4096
double   maxeps = 0.1e-7;
int itmax = 100;
double A [N][N];
double relax(char *);
void init();
void verify();
int numt; 


int main(int an, char **as) {
    numt = omp_get_num_threads();	

    char isync[numt];
	double time = omp_get_wtime();
	
    omp_set_num_threads(atoi(as[1]));
	init();
    for(int it = 1; it <= itmax; it++) {
        double eps = relax(isync);
        printf( "it=%4i eps=%f\n", it, eps);
		if (eps < maxeps) break;
	}
	verify();
	
    printf("time = %f\n", omp_get_wtime()-time);
	return 0;
}


void init() {
#pragma omp parallel for
	for(int i = 1; i < N - 1; i++) {
	    for(int j = 1; j < N - 1; j++) {
		    A[i][j]= ( 1. + i + j ) ;
	    }
    }
} 
double relax(char *isync) {
    double eps = 0.0;
#pragma omp parallel shared(A,numt) reduction(max:eps)
    {	
	    int iam = omp_get_thread_num();
        for(int i = 1; i <= N - 2; i++) {
#pragma omp for
            for(int j = 1; j <= N - 2; j++) {
                A[i][j] = (A[i - 1][j] + A[i + 1][j]) / 2.;
            }
        }

        int limit = Min(numt-1, N-2);
	    isync[iam] = 0;


//#pragma omp barrier
#pragma omp for  nowait
	    for (int i = 1; i <= N - 2; i++) {
	        if ((iam > 0) && (iam <= limit)) {
	            for (;isync[iam - 1] == 0;) {
	                #pragma omp flush(isync)
		        }
		        isync[iam - 1] = 0;
                #pragma omp flush(isync)
	        }
            for(int j = 1; j <= N - 2; j++) {
                double e = A[i][j];
      	        A[i][j] = (A[i][j - 1] + A[i][j + 1]) / 2.;
	            eps = Max(eps, fabs(e - A[i][j]));
	        }
	        if (iam < limit) {
                for (;isync[iam] == 1;) {
                   #pragma omp flush(isync)
	            }
                isync[iam] = 1;
                #pragma omp flush(isync)
            }
    	}
    }

    return eps;
}
void verify() { 
	double s = 0.;
	#pragma omp parallel for shared(A), reduction(+:s)
	for(int i = 0; i <= N - 1; i++) {
	    for(int j = 0; j <= N - 1; j++) {
		    s += A[i][j] * (i + 1) * (j + 1) / (N * N);
	    }
    }
	printf("  S = %f\n",s); 
}
	
	
