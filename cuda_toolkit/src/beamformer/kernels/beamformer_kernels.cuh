#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../defs.h"
#include "../beamformer_constants.cuh"

namespace bf_kernels
{
    __global__ void
    walsh_forces_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

    __global__ void
    per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

    __global__ void
    per_channel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, const float* hadamard);

    __global__ void
    mixes_beamform(const cuComplex* rfData, cuComplex* volume, u8 mixes_rows[128]);

    __global__ void
    forces_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

	__global__ void
    uforces_beamform(const cuComplex* rfData, cuComplex* volume, const short* uforces_elements);

	__global__ void
    hercules_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

	__global__ void
    walsh_hercules_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

    template<SequenceId SEQUENCE, EncodingMatrix READI_ORDER> __global__ void
    readi_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

	__global__ void
	tpw_beamform(const cuComplex* rfData, cuComplex* volume, const float* angles);
	
	__global__ void
	block_beamform(const cuComplex* rfData, cuComplex* volume);
}


