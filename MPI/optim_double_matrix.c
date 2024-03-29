#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N  2048//(2*2*2*2*2*2+2)
double maxeps = 0.1e-7;
int itmax = 100;

double eps;
double A [N][N];
double B [N][N];

void relax();
void init();
void verify(); 

int world_rank;
int world_size;
int world_start, world_end;

void print_matrix() {
    for (int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(stderr, "%lf ", A[i][j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

void calc_start_end(int *start, int *end, int rank) {
    int for_one = N / world_size;
    if (rank != 0){
        *start = rank * for_one;
    } else {
        *start = 1;
    }
    if (rank != world_size - 1){
        *end = (rank + 1) * for_one;
    } else {
        *end = N - 1;
    }
}

void synchronize_string_A() {
    MPI_Request req[world_size];
    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) continue;
        
        MPI_Isend(A[world_start], N * (world_end - world_start), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req[i]);
        MPI_Request_free(&req[i]);
    }

    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) continue;
        
        int start, end;
        calc_start_end(&start, &end, i);
        MPI_Recv(A[start], N * (end - start), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void synchronize_string_B() {
    MPI_Request req[world_size];
    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) continue;
        
        MPI_Isend(B[world_start], N * (world_end - world_start), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req[i]);
        MPI_Request_free(&req[i]);
    }

    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) continue;
        
        int start, end;
        calc_start_end(&start, &end, i);
        MPI_Recv(B[start], N * (end - start), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}


int main(int an, char **as)
{
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    calc_start_end(&world_start, &world_end, world_rank);

    init();
    MPI_Barrier(MPI_COMM_WORLD);
	for(int it = 1; it <= itmax; it++) {
		eps = 0.;
		relax();
        if (world_rank == 0) {
            printf( "it=%4i   eps=%f\n", it,eps);
            if (eps < maxeps) break;
        }
    }
	verify();
    MPI_Finalize();
	return 0;
}


void init() {
    for (int i = world_start; i < world_end; i++) {
	    for(int j = 1; j < N - 1; j++) {
		    A[i][j] = (1. + i + j);
            B[j][i] = (1. + i + j);
        }
    }

    synchronize_string_A();
    synchronize_string_B();
} 


void relax() {
	for(int i = 1; i < N - 1; i++) {
	    for(int j = world_start; j < world_end; j++) {
		    B[j][i] = (A[i - 1][j] + A[i + 1][j]) / 2.;
        }
	}
    synchronize_string_B();
    double local_eps = 0;

	for(int i = world_start; i < world_end; i++) {
        for(int j = 1; j < N - 1; j++) {
		    double e = B[j][i];
		    A[i][j] = (B[j - 1][i] + B[j + 1][i]) / 2.;
		    local_eps = Max(local_eps, fabs(e - A[i][j]));
        }
	}

    synchronize_string_A();
    MPI_Reduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

void verify() { 
	double local_sum = 0.;
    double global_sum = 0.;
	for(int i = world_start; i < world_end; i++) {
	    for(int j = 0; j <= N - 1; j++) {
		    local_sum = local_sum + A[i][j] * (i + 1) * (j + 1) / (N * N);
	    }
    }
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
    if (world_rank == 0) {
        printf("  S = %f\n", global_sum);
    }
}
