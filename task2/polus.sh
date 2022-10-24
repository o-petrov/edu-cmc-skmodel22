#!/bin/sh
for rep in 1 2 3 4 5 6 7
do
	for eps in 3e-5 5e-6 1.5e-6
	do
		for procs in 02 04 16 28 32
		do
			name=eps"$eps"_procs"$procs"_r$rep
			mpisubmit.pl -p $procs -w 00:01 a.out --stdout $name.out --stderr $name.err -- $eps
		done
	done
done
