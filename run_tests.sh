#!/bin/bash

# Command line arguments specify the sizes of the matrice under test.
# Possible values: 0008, 0016, 0064, 0256, 0512, 1024
#

# Specify where the input data are to be found, f.ex.
DATA_DIR=$(pwd)/data

while [ "$1" != "" ]; do
    SIZE=$1; shift 1
    echo "Testing ${DATA_DIR}/matrix_${SIZE}.mm"

    # Step 2 - row cyclic
    mpiexec -n 16 step_2/step_2.x -g 16x1 \
        -i ${DATA_DIR}/matrix_${SIZE}.mm -o step_2/res_${SIZE}_step_2.mm

    # Step 3 - total cyclic
    mpiexec -n 16 step_3/step_3.x -g 4x4 \
        -i ${DATA_DIR}/matrix_${SIZE}.mm -o step_3/res_${SIZE}_step_3_44.mm
    mpiexec -n 16 step_3/step_3.x -g 2x8 \
        -i ${DATA_DIR}/matrix_${SIZE}.mm -o step_3/res_${SIZE}_step_3_28.mm
    mpiexec -n 16 step_3/step_3.x -g 8x2 \
        -i ${DATA_DIR}/matrix_${SIZE}.mm -o step_3/res_${SIZE}_step_3_82.mm

    # Step 5 - OpenMP
    export OMP_NUM_THREADS=8
    step_5/step_5.x \
        -i ${DATA_DIR}/matrix_${SIZE}.mm -o step_5/res_${SIZE}_step_5_8.mm
    export OMP_NUM_THREADS=16
    step_5/step_5.x \
        -i ${DATA_DIR}/matrix_${SIZE}.mm -o step_5/res_${SIZE}_step_5_16.mm

    # Step 6 - column cyclic
    mpiexec -n 16 step_6/step_6.x -g 1x16 \
        -i ${DATA_DIR}/matrix_${SIZE}.mm -o step_6/res_${SIZE}_step_6.mm


    echo
done