#!/bin/bash
# Run the solver on 10 MPI ranks inside Docker, reading input_y.dat from the
# current directory.  Build the image first:  docker build -t digitalrock-elasticity .
#
# Usage:  ./run9.sh [input_file] [number_of_mpi_ranks]
INPUT=${1:-input_y.dat}
NP=${2:-10}
docker run --rm -v "$(pwd)":/workspace -w /workspace digitalrock-elasticity \
    mpirun -n "$NP" python3 /opt/digitalrock/testRun9y.py "$INPUT"
