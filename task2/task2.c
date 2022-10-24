#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"

#define N 32
#define NNN (3*N)
#define VALUE (1.0/24.0)

static void handle_error(int errcode, char *str)
{
	char msg[MPI_MAX_ERROR_STRING];
	int resultlen;
	MPI_Error_string(errcode, msg, &resultlen);
	fprintf(stderr, "%s: %s\n", str, msg);
	MPI_Abort(MPI_COMM_WORLD, errcode);
}

inline void my_init(int *argc, char ***argv, int *size, int *rank) {
	int err = MPI_Init(argc, argv);
	if (err != MPI_SUCCESS) {
		handle_error(err, "cannot init MPI");
		MPI_Abort(MPI_COMM_WORLD, err);
		exit(err);
	}

	err = MPI_Comm_size(MPI_COMM_WORLD, size);
	if (err != MPI_SUCCESS) {
		handle_error(err, "cannot set comm size");
		MPI_Abort(MPI_COMM_WORLD, err);
		exit(err);
	}

	err = MPI_Comm_rank(MPI_COMM_WORLD, rank);
	if (err != MPI_SUCCESS) {
		handle_error(err, "cannot set rank");
		MPI_Abort(MPI_COMM_WORLD, err);
		exit(err);
	}
}

inline void my_scatter(double *sendbuf, double *recvbuf, int recvsize, int process_count) {
	int err = MPI_Scatter(
		sendbuf, recvsize, MPI_DOUBLE,
		recvbuf, recvsize, MPI_DOUBLE, 0, MPI_COMM_WORLD
	);

	if (err != MPI_SUCCESS) {
		handle_error(err, "cannot scatter");
		MPI_Abort(MPI_COMM_WORLD, err);
		exit(err);
	}
}

inline void my_sum(double *local, double *global) {
	int err = MPI_Reduce(local, global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (err != MPI_SUCCESS) {
		handle_error(err, "cannot reduce sum");
		MPI_Abort(MPI_COMM_WORLD, err);
		exit(err);
	}
}

inline void my_max(double *local, double *global) {
	int err = MPI_Reduce(local, global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (err != MPI_SUCCESS) {
		handle_error(err, "cannot reduce max");
		MPI_Abort(MPI_COMM_WORLD, err);
		exit(err);
	}
}

inline double f(double *point) {
	double x, y, z;
	x = point[0];
	y = point[1];
	z = point[2];
	return x * x * x * y * y * z;
}

inline double parse_arg(int argc, char **argv) {
	double result;
	if (argc != 2) {
		printf("cannot read eps: argc=%i\n", argc);
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}

	result = atof(argv[1]);
	if (result <= 0.0) {
		printf("cannot read eps from \"%s\": atof returned %g\n", argv[1], result);
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}

	return result;
}

int main(int argc, char *argv[]) {
	int process_count, process_rank;
	double local_points[NNN];

	my_init(&argc, &argv, &process_count, &process_rank);
	double local_time = MPI_Wtime();

	if (process_rank == 0) {
		// MASTER
		double all_points[NNN * process_count];
		long total_points = 0;
		double total_sum = 0.0;

		// master reads accuracy etc
		double error = INFINITY;
		double tolerance = parse_arg(argc, argv);
		srand48(2);

		while (error > tolerance) {
			// generate new points
			for (size_t i = NNN; i < NNN * process_count; i++) {
				all_points[i] = -drand48();
			}
			total_points += N * (process_count - 1);
			// send points to workers
			my_scatter(all_points, MPI_IN_PLACE, NNN, process_count);
			// master receives results from workers
			my_sum(MPI_IN_PLACE, &total_sum);
			error = fabs(total_sum / total_points - VALUE);
		}

		// scatter NAN as signal to stop
		for (
			double *first_point = all_points;
			first_point < all_points + NNN * process_count;
			first_point += NNN
		) {
			*first_point = NAN;
		}
		my_scatter(all_points, MPI_IN_PLACE, NNN, process_count);

		// reduce-max local_time
		local_time = MPI_Wtime() - local_time;
		my_max(MPI_IN_PLACE, &local_time);
		printf(
			"result\t%i\t%.2e\t%i\t%.2e\t%.2e\t%li\t%.2e\n",
			N,
			tolerance,
			process_count,
			total_sum / total_points,
			error,
			total_points,
			local_time
		);

	} else {
		// WORKER
		// get new points
		my_scatter(NULL, local_points, NNN, process_count);
		// stop if local_points[0] is NaN
		while (!isnan(local_points[0])) {
			// worker computes
			double local_result = 0.0;
			for (double *point = local_points; point < local_points + NNN; point+=3) {
				local_result += f(point);
			}
			// worker sends result to root
			my_sum(&local_result, NULL);
			// get new points
			my_scatter(NULL, local_points, NNN, process_count);
		}
		// reduce-max local_time
		local_time = MPI_Wtime() - local_time;
		my_max(&local_time, NULL);
	}

	MPI_Finalize();
	exit(0);
}
