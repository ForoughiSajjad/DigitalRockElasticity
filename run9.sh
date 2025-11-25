sudo docker run -v $(pwd):/workspace -w /workspace my-dolfinx \
    bash -c "pip install psutil && mpirun -n 10 python3 testRun9y.py"
