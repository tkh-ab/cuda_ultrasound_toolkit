#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../defs.h"
#include "../beamformer_constants.cuh"

namespace bf_kernels
{
    __global__ void
    walsh_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

    __global__ void
    per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

    __global__ void
    per_channel_beamform(const cuComplex* rfData, cuComplex* volume, uint readi_group_id, const float* hadamard);

    __global__ void
    mixes_beamform(const cuComplex* rfData, cuComplex* volume, u8 mixes_rows[128]);

    __global__ void
    forces_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);

    __host__ bool
	copy_kernel_constants(const BeamformerConstants& constants);

    template<SequenceId SEQUENCE, ReadiOrdering READI_ORDER> __global__ void
    readi_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard);
}


