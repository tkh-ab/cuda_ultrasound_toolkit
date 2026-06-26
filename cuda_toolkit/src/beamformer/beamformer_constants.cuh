#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../defs.h"

static constexpr float CUDART_PI_F = 3.141592654F;
static constexpr uint MAX_TX_COUNT = 128;

static constexpr uint Y_BLOCK_SIZE = 1;
namespace bf_kernels
{

	enum class FocusType
	{
		NO_TX_FOCUS = 0,
		XZ_PLANE = 1,
		YZ_PLANE = 2,
		XZ_FOCUS = 3,
		YZ_FOCUS = 4,
		SPHERE_FOCUS = 5
	};

	enum class TrOscDirection
	{
		NONE = 0,
		X_AXIS = 1,
		Y_AXIS = 2,
		BOTH_AXES = 3,
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
		FocusType focus_type;
        RCAOrientation tx_orientation;
		RCAOrientation rx_orientation;
        SequenceId sequence;

		float lambda_0;

        // Render Constants
        uint3 voxel_dims;
        float3 volume_mins;
        float3 resolutions;
		float fn_tx; // Tx f-number
        float fn_rx; // Rx f-number
		float coherency_weighting;

        // Sequence Constants
        u8 mixes_count;
        u8 mixes_offset;

        u8 readi_group_count;
        u8 readi_group_id;
        EncodingMatrix encoded_matrix;

		ApoType apo_type;				// Type of apodization to apply during beamforming
		int to_power;					// Power to raise the TO apodization to
    } BeamformerConstants;
}

// NOTE: The app must be built with relocatable device code for this to link properly
// Compile with nvcc -rdc=true, the definition is in beamformer.cu
extern __device__ __constant__ bf_kernels::BeamformerConstants Beamformer_Constants;