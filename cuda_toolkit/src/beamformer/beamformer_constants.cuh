#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../defs.h"

static constexpr float CUDART_PI_F = 3.141592654F;
static constexpr uint MAX_TX_COUNT = 128;

static constexpr uint Y_BLOCK_SIZE = 1;
namespace bf_kernels
{
    enum class FocalDirection
    {
        PLANE_FOCUS = 0,
        XZ_FOCUS = 1,
        YZ_FOCUS = 2,
        SPHERE_FOCUS = 3,
    };  
    typedef struct
    {
        // Data Constants
        size_t sample_count;
        size_t channel_count;
        size_t tx_count;
        float2 xdc_mins;
        float2 xdc_maxes;
        float samples_per_meter;
        float3 focal_point;
        float2 pitches;
        int delay_samples;
        FocalDirection focal_direction;
        SequenceId sequence;

        // Render Constants
        uint3 voxel_dims;
        float3 volume_mins;
        float3 resolutions;
        float f_number;
		float coherency_weighting;

        // Sequence Constants
        u8 mixes_count;
        u8 mixes_offset;

        u8 readi_group_count;
        u8 readi_group_id;
        EncodingMatrix encoded_matrix;
    } BeamformerConstants;
}

// NOTE: The app must be built with relocatable device code for this to link properly
// Compile with nvcc -rdc=true, the definition is in beamformer.cu
extern __device__ __constant__ bf_kernels::BeamformerConstants Beamformer_Constants;