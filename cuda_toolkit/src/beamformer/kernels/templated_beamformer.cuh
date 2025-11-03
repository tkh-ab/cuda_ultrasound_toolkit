#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../defs.h"
#include "../beamformer_constants.cuh"
#include "../beamformer_utils.cuh"

namespace bf_kernels
{

template <SequenceId SEQ>
concept SupportedDASSequence = (SEQ == SequenceId::FORCES) || (SEQ == SequenceId::HERCULES);

template<SequenceId SEQ, FocalDirection DIR> __device__ inline float3 
initial_tx_vec(const float2 xdc_mins, const float2 pitches, const float3 vox_loc, const float3 focal_point)
{
	if constexpr (SEQ == SequenceId::FORCES)
	{
		return make_float3(vox_loc.x - Beamformer_Constants.xdc_mins.x , 0.0f, vox_loc.z);
	}
	else if constexpr (SEQ == SequenceId::HERCULES)
	{
		if constexpr (DIR == FocalDirection::PLANE_FOCUS)
		{
			return make_float3(0.0f, 0.0f, vox_loc.z);
		}
		else if constexpr (DIR == FocalDirection::XZ_FOCUS)
		{
			return make_float3(vox_loc.x - focal_point.x, 0.0f, vox_loc.z - focal_point.z);
		}
		else if constexpr (DIR == FocalDirection::YZ_FOCUS)
		{
			return make_float3(0.0f, vox_loc.y - focal_point.y, vox_loc.z - focal_point.z);
		}
		else if constexpr (DIR == FocalDirection::SPHERE_FOCUS)
		{
			static_assert(false, "Spherical focusing not supported for HERCULES");
		}
	}
	else
	{
		static_assert(false, "Unsupported sequence for DAS beamforming");
	}
}

template<SequenceId SEQ> __device__ inline float3 
initial_rx_vec(const float2 xdc_mins, const float2 pitches, const float3 vox_loc)
{
	if constexpr (SEQ == SequenceId::FORCES)
	{
		return make_float3(Beamformer_Constants.xdc_mins.x -vox_loc.x, 0.0f, -vox_loc.z);
	}
	else if constexpr (SEQ == SequenceId::HERCULES)
	{
		return make_float3(Beamformer_Constants.xdc_mins.x - vox_loc.x,
							Beamformer_Constants.xdc_mins.y - vox_loc.y, -vox_loc.z);
	}
	else
	{
		static_assert(false, "Unsupported sequence for DAS beamforming");
	}
}

// Returns the unpacked decode bit in the top bit of the uint
template<EncodingMatrix MAT> __device__ __forceinline__ uint 
unpack_decode_bit(int signal, int sub_signal, u64 decode_row)
{
	if constexpr (MAT == EncodingMatrix::HADAMARD)
	{
		return ((decode_row >> sub_signal) & 1u ) << 31;
	}
	else if constexpr (MAT == EncodingMatrix::WALSH)
	{
		// Every other signal (with raw indicies) is encoded with the row in reverse order.
		// This is because the Walsh matrix isn't truly recursive like Hadamard.
		// This could be fixed in decoding as well.
		int decode_index = !(signal & 1u) ? sub_signal : (Beamformer_Constants.readi_group_count - 1 - sub_signal);

		return ((decode_row >> decode_index) & 1u ) << 31;
	}
	else
	{
		return 0;
	}
}

template<FocalDirection DIR, EncodingMatrix READI> __global__ void
hercules_beamform_new(const cuComplex* __restrict__ rf_data, cuComplex* volume, u64 decode_row = 0)
{
	// TODO: Check if inlining this to the vox_loc calculation drops the register count
	uint3 voxel_idx = { threadIdx.x + blockIdx.x * blockDim.x,
						threadIdx.y + blockIdx.y * blockDim.y,
						threadIdx.z + blockIdx.z * blockDim.z };

	const float3 vox_loc = {
		Beamformer_Constants.volume_mins.x + voxel_idx.x * Beamformer_Constants.resolutions.x,
		Beamformer_Constants.volume_mins.y + voxel_idx.y * Beamformer_Constants.resolutions.y,
		Beamformer_Constants.volume_mins.z + voxel_idx.z * Beamformer_Constants.resolutions.z,
	};

	if(!COMPARE_LT_V3(voxel_idx, Beamformer_Constants.voxel_dims))
	{
		return;
	}
	float3 focal_point = { 0.0f, 0.0f, Beamformer_Constants.focal_point.z };

	float3 tx_vec = initial_tx_vec<HERCULES, DIR>(Beamformer_Constants.xdc_mins,
													Beamformer_Constants.pitches,
													vox_loc,
													focal_point);

	float3 rx_vec = initial_rx_vec<HERCULES>(Beamformer_Constants.xdc_mins,
											Beamformer_Constants.pitches,
											vox_loc);

	float initial_rx_vec_y = rx_vec.y;

	float tx_distance = copysignf(NORM_F3(tx_vec), tx_vec.z);
	float incoherent_sum = 0.0f;
	cuComplex total = {0.0f, 0.0f};
	int total_transmits = Beamformer_Constants.tx_count * Beamformer_Constants.readi_group_count;
	int t_signal = 0;
	uint decode_bit = 0;
	int sub_sig_count;
	float current_tx_idx;

	// This should let the compiler remove the outer loop when not using READI (TODO confirm in SASS)
	if constexpr (READI != EncodingMatrix::NONE) {sub_sig_count = Beamformer_Constants.readi_group_count;}
	else {sub_sig_count = 1;}

	for (int readi_sub_signal = 0; readi_sub_signal < sub_sig_count; readi_sub_signal++)
	{
		for (int t_signal = 0; t_signal < Beamformer_Constants.tx_count; t_signal++)
		{
			if constexpr (READI == EncodingMatrix::HADAMARD)
			{
				current_tx_idx = (float)readi_sub_signal * Beamformer_Constants.tx_count + t_signal;
			}
			else if constexpr (READI == EncodingMatrix::WALSH)
			{
				current_tx_idx = (float)t_signal * Beamformer_Constants.readi_group_count + readi_sub_signal;
			}
			else
			{
				current_tx_idx = (float)t_signal;
			}
			rx_vec.y = initial_rx_vec_y + current_tx_idx * Beamformer_Constants.pitches.y;
			size_t channel_offset = Beamformer_Constants.channel_count * Beamformer_Constants.sample_count * t_signal;
			for (int c = 0; c < Beamformer_Constants.channel_count; c++)
			{
				static constexpr float APO_MIN = 0.1f;
				float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.f_number);

				if(apo > APO_MIN)
				{
					float scan_index = (tx_distance + NORM_F3(rx_vec) + Beamformer_Constants.focal_point.z)
										* Beamformer_Constants.samples_per_meter
										+ Beamformer_Constants.delay_samples;

					scan_index = utils::clampf(scan_index, 1.0f, (float)Beamformer_Constants.sample_count - 2.0f);
					
					
					cuComplex value = utils::lerp_read(scan_index, rf_data + channel_offset);	
					//cuComplex value = utils::fast_cubic_spline(scan_index, rf_data + channel_offset);					

					if constexpr (READI != EncodingMatrix::NONE)
					{
						decode_bit = unpack_decode_bit<READI>(t_signal, readi_sub_signal, decode_row);
						apo = __uint_as_float(__float_as_uint(apo) ^ decode_bit);
					}
					
					total.x = fmaf(apo, value.x, total.x);
					total.y = fmaf(apo, value.y, total.y);
					incoherent_sum += NORM_SQUARE_V2(value);
				}

				channel_offset += Beamformer_Constants.sample_count;
				rx_vec.x += Beamformer_Constants.pitches.x;
			}
			rx_vec.x -= Beamformer_Constants.pitches.x * Beamformer_Constants.channel_count;
		}
	}

	float coherency_factor = NORM_SQUARE_V2(total) / incoherent_sum;
	coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
	coherency_factor = utils::clear_nan(coherency_factor);
	//total = SCALE_V2(total, coherency_factor);

	size_t volume_offset = voxel_idx.z * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + voxel_idx.y * Beamformer_Constants.voxel_dims.x + voxel_idx.x;
	volume[volume_offset] = total;
}

// Moving the channel loop outside the kernel. No atomics for now so only one execuation can be live at a time.
template<EncodingMatrix READI> __global__ void
forces_beamform_new(const cuComplex* __restrict__ rf_data, cuComplex* volume, u64 decode_row = 0)
{
	// TODO: Check if inlining this to the vox_loc calculation drops the register count
	uint3 voxel_idx = { threadIdx.x + blockIdx.x * blockDim.x,
						threadIdx.y + blockIdx.y * blockDim.y,
						threadIdx.z + blockIdx.z * blockDim.z };

	const float3 vox_loc = {
		Beamformer_Constants.volume_mins.x + voxel_idx.x * Beamformer_Constants.resolutions.x,
		Beamformer_Constants.volume_mins.y + voxel_idx.y * Beamformer_Constants.resolutions.y,
		Beamformer_Constants.volume_mins.z + voxel_idx.z * Beamformer_Constants.resolutions.z,
	};

	if(!COMPARE_LT_V3(voxel_idx, Beamformer_Constants.voxel_dims))
	{
		return;
	}

	float3 tx_vec = initial_tx_vec<FORCES, FocalDirection::XZ_FOCUS>(Beamformer_Constants.xdc_mins,
													Beamformer_Constants.pitches,
													vox_loc, {0.0f, 0.0f, 0.0f});

	float3 rx_vec = initial_rx_vec<FORCES>(Beamformer_Constants.xdc_mins,
											Beamformer_Constants.pitches,
											vox_loc);
							
	float incoherent_sum = 0.0f;
	cuComplex total = {0.0f, 0.0f};
	int total_transmits = Beamformer_Constants.tx_count * Beamformer_Constants.readi_group_count;
	uint decode_bit = 0;
	for (int c = 0; c < Beamformer_Constants.channel_count; c++)
	{
		static constexpr float APO_MIN = 0.1f;
		float rx_dist = NORM_F3(rx_vec);
		float apo = utils::f_num_apodization(abs(rx_vec.x), vox_loc.z, Beamformer_Constants.f_number);
		if(apo > APO_MIN)
		{
			for (int readi_sub_signal = 0; readi_sub_signal < Beamformer_Constants.readi_group_count; readi_sub_signal++)
			{
				if constexpr (READI != EncodingMatrix::NONE) { decode_bit = ((decode_row >> readi_sub_signal) & 1u) << 31; }
				for( int t_signal = 0; t_signal < Beamformer_Constants.tx_count; t_signal++)
				{

					float scan_index = (NORM_F3(tx_vec) + rx_dist)
										* Beamformer_Constants.samples_per_meter
										+ Beamformer_Constants.delay_samples;

					scan_index = utils::clampf(scan_index, 1.0f, (float)Beamformer_Constants.sample_count - 2.0f);
					size_t channel_offset = Beamformer_Constants.channel_count * Beamformer_Constants.sample_count * t_signal + Beamformer_Constants.sample_count * c;
					
					cuComplex value = utils::lerp_read(scan_index, rf_data + channel_offset);	
					//cuComplex value = utils::fast_cubic_spline(scan_index, rf_data + channel_offset);					

					float signed_apo;
					if constexpr (READI != EncodingMatrix::NONE)
					{
						// Set the sign of the apo to the hadamard coefficient -> inverts the sample if needed
						signed_apo = __uint_as_float(__float_as_uint(apo) ^ decode_bit);
					}
					else
					{
						signed_apo = apo;
					}
					
					total.x = fmaf(signed_apo, value.x, total.x);
					total.y = fmaf(signed_apo, value.y, total.y);
					incoherent_sum += NORM_SQUARE_V2(value);

					tx_vec.x -= Beamformer_Constants.pitches.x;
				}
			}
			tx_vec.x += Beamformer_Constants.pitches.x * total_transmits;
		}
		rx_vec.x += Beamformer_Constants.pitches.x;
	}

	float coherency_factor = NORM_SQUARE_V2(total) / incoherent_sum;
	coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
	coherency_factor = utils::clear_nan(coherency_factor);
	total = SCALE_V2(total, coherency_factor);

	size_t volume_offset = voxel_idx.z * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + voxel_idx.y * Beamformer_Constants.voxel_dims.x + voxel_idx.x;
	volume[volume_offset] = total;
	
}

}


