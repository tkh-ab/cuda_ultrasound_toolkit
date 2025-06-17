#!/bin/bash

cuda_lib_header_path="../../cuda_toolkit/include"
output_dir="output"
mkdir -p ${output_dir}

header_file="./cuda_transfer.h"
output_header="${output_dir}/cuda_transfer_matlab.h"

cp -f ${header_file} ${output_header}
cp -f ../src/communication_params.h ${output_dir}/communication_params.h
cp -f ${cuda_lib_header_path}/cuda_beamformer_parameters.h ${output_dir}/cuda_beamformer_parameters.h

pushd ${output_dir}

cflags="-march=native -std=c11 -Wall  -Wno-unused-variable -Wno-unused-function -I./."
debug_flags="${cflags} -O0 -fuse-ld=lld -g -gcodeview -Wl,--pdb="
libcflags="${cflags} -fPIC -shared"
ldflags="${ldflags} -lm -lgdi32 -lwinmm -L./."

matlab_libcflags="${libcflags} -DMATLAB_CONSOLE"
matlab_ldflags="-llibmat -llibmex"

cc=${CC:-cc}

${cc} ${libcflags} ${debug_flags} ../cuda_transfer.c -o cuda_transfer_debug.dll ${ldflags}
${cc} ${matlab_libcflags} ../cuda_transfer.c -o cuda_transfer_matlab.dll -L'C:/Program Files/MATLAB/R2024a/extern/lib/win64/mingw64' ${matlab_ldflags}
${cc} ${cflags} ${debug_flags} ../test_lib.c -o test_lib.exe ${ldflags} -lcuda_transfer_debug

popd

