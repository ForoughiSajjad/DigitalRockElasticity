# Build:  docker build -t digitalrock-elasticity .
# Run:    docker run --rm -v "$(pwd)":/workspace -w /workspace digitalrock-elasticity \
#             mpirun -n 4 python3 /opt/digitalrock/testRun9y.py input_y.dat
#
# The solver is written against the DOLFINx 0.8/0.9 Python API
# (dolfinx.fem.functionspace, dolfinx.fem.petsc.LinearProblem). Newer DOLFINx
# releases changed LinearProblem, so keep the version pinned.
ARG DOLFINX_VERSION=v0.9.0
FROM dolfinx/dolfinx:${DOLFINX_VERSION}

RUN pip3 install --no-cache-dir scikit-image tifffile psutil \
 || pip3 install --no-cache-dir --break-system-packages scikit-image tifffile psutil

COPY testRun9y.py /opt/digitalrock/testRun9y.py
WORKDIR /workspace
