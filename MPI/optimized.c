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

void synchronize_string() {
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

void print_tmp_matrix(double *tmp_matrix, int height, int width) {
    fprintf(stderr, "thread %d, sending matrix:\n", world_rank);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(stderr, "%lf ", tmp_matrix[i * width + j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

void synchronize_colums() {
    MPI_Request req[world_size];

    double tmp_matrix[(N - 2) * (world_end - world_start)];

    for (int i = 1; i < N - 1; i++) {
        for (int j = world_start; j < world_end; j++) {
            tmp_matrix[(i - 1) * (world_end - world_start) + (j - world_start)] = A[i][j];
        }
    }
    
    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) continue;

        MPI_Isend(tmp_matrix, (N - 2) * (world_end - world_start), MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &req[i]);
        MPI_Request_free(&req[i]);
    }

    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) continue;

        int start, end;
        calc_start_end(&start, &end, i);
        double an_tmp_matrix[(N - 2) * (end - start)];

        MPI_Recv(an_tmp_matrix, (N - 2) * (end - start), MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int k = 1; k < N - 1; k++) {
            for (int j = start; j < end; j++) {
                A[k][j] = an_tmp_matrix[(k - 1) * (end - start) + (j - start)];
            }
        }
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
      //      print_matrix();
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
		    A[i][j] = (1. + i + j) ;
        }
    }
    synchronize_string();
} 

void relax() {
	for(int i = 1; i < N - 1; i++) {
	    for(int j = world_start; j < world_end; j++) {
		    A[i][j] = (A[i - 1][j] + A[i + 1][j]) / 2.;
        }
	}
    synchronize_colums();
    double local_eps = 0;

	for(int i = world_start; i < world_end; i++) {
        for(int j = 1; j < N - 1; j++) {
		    double e = A[i][j];
		    A[i][j] = (A[i][j - 1] + A[i][j + 1]) / 2.;
		    local_eps = Max(local_eps, fabs(e - A[i][j]));
        }
	}

    synchronize_string();
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
