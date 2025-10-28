#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../defs.h"
#include "../beamformer_constants.cuh"
#include "../beamformer_utils.cuh"

namespace bf_kernels
{
	// TODO: If this works remove the other one in beamformer_kernels.cu
	__host__ bool
	copy_kernel_constants1(const BeamformerConstants& constants)
	{
		CUDA_RETURN_IF_ERROR(cudaMemcpyToSymbol(Beamformer_Constants, &constants, sizeof(BeamformerConstants)));
		return true;
	}

	template <SequenceId SEQ>
	concept SupportedDASSequence = (SEQ == SequenceId::FORCES) || (SEQ == SequenceId::HERCULES);

	template<SequenceId SEQ, FocalDirection DIR> __device__ inline float3 
	initial_tx_vec(const float2 xdc_mins, const float2 pitches, const float3 vox_loc, const float3 focal_point)
	{
		if constexpr (SEQ == SequenceId::FORCES)
		{
			return make_float3(vox_loc.x - (Beamformer_Constants.xdc_mins.x + Beamformer_Constants.pitches.x / 2), 0.0f, vox_loc.z);
		}
		else if constexpr (SEQ == SequenceId::HERCULES)
		{
			if constexpr (DIR == FocalDirection::PLANE)
			{
				return make_float3(0.0f, 0.0f, vox_loc.z);
			}
			else if constexpr (DIR == FocalDirection::XZ_PLANE)
			{
				return make_float3(vox_loc.x - focal_point.x, 0.0f, vox_loc.z - focal_point.z);
			}
			else if constexpr (DIR == FocalDirection::YZ_PLANE)
			{
				return make_float3(0.0f, vox_loc.y - focal_point.y, vox_loc.z - focal_point.z);
			}
			else if constexpr (DIR == FocalDirection::SPHERE)
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
			return make_float3(Beamformer_Constants.xdc_mins.x + Beamformer_Constants.pitches.x / 2 -vox_loc.x, 0.0f, -vox_loc.z);
		}
		else if constexpr (SEQ == SequenceId::HERCULES)
		{
			return make_float3(Beamformer_Constants.xdc_mins.x + Beamformer_Constants.pitches.x / 2 - vox_loc.x,
							   Beamformer_Constants.xdc_mins.y + Beamformer_Constants.pitches.y / 2 - vox_loc.y, -vox_loc.z);
		}
		else
		{
			static_assert(false, "Unsupported sequence for DAS beamforming");
		}
	}

	template<SequenceId SEQ> __device__ inline float3
	calc_tx_vector(float3 initial_vec, uint transmit_idx, float2 pitches)
	{
		if constexpr (SEQ == SequenceId::FORCES)
		{
			// tx vector is voxel_loc - focus so as we move left to right on the array we need to subtract the pitch
			initial_vec.x -= transmit_idx * pitches.x;
		}
		else if constexpr (SEQ == SequenceId::HERCULES)
		{}
		else
		{
			static_assert(false, "Unsupported sequence for DAS beamforming");
		}
		return initial_vec;
	}
	
	template<SequenceId SEQ> __device__ inline float3
	calc_rx_vector(float3 initial_vec, uint channel_idx, uint transmit_idx, float2 pitches)
	{
		if constexpr (SEQ == SequenceId::FORCES)
		{
			initial_vec.x += channel_idx * pitches.x;
		}
		else if constexpr (SEQ == SequenceId::HERCULES)
		{
			initial_vec.x += channel_idx * pitches.x;
			initial_vec.y += transmit_idx * pitches.y;
		}
		else
		{
			static_assert(false, "Unsupported sequence for DAS beamforming");
		}
		return initial_vec;
	}

	__device__ inline float calc_total_distance(float3 tx_vec, float3 rx_vec, float focal_depth)
	{
		// Tx vec is from the focus -> the voxel.
		// If its z value is negative then we are between the transducer and the focus.
		return focal_depth + NORM_F3(rx_vec) + copysignf(NORM_F3(tx_vec), tx_vec.z);
		//return focal_depth + NORM_F3(rx_vec) + NORM_F3(tx_vec) * copysignf(1.0f, tx_vec.z);
	}
	
	/* Each dim in thread and block ID corresponds with a voxel dim for now */
    template<SequenceId SEQ, FocalDirection DIR> requires SupportedDASSequence<SEQ> __global__ void
    das_beamform(const cuComplex* rf_data, cuComplex* volume, u64 hadamard_row = 0)
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

		float3 focal_point;
		if constexpr (SEQ == SequenceId::FORCES)
		{
			focal_point = { 0.0f, 0.0f, 0.0f };
		}
		else
		{
			focal_point = { 0.0f, 0.0f, Beamformer_Constants.focal_point.z };
		}

		float3 initial_tx = initial_tx_vec<SEQ, DIR>(Beamformer_Constants.xdc_mins,
													 Beamformer_Constants.pitches,
													 vox_loc,
													 focal_point);

		float3 initial_rx = initial_rx_vec<SEQ>(Beamformer_Constants.xdc_mins,
											   Beamformer_Constants.pitches,
											   vox_loc);

		float incoherent_sum = 0.0f;
		cuComplex total = {0.0f, 0.0f};
		for (int t_position = 0; t_position < Beamformer_Constants.tx_count * Beamformer_Constants.readi_group_count; t_position++)
		{
			int readi_sub_signal = t_position / Beamformer_Constants.tx_count;
			int t_signal = t_position % Beamformer_Constants.tx_count;
			for (int c = 0; c < Beamformer_Constants.channel_count; c++)
			{
				static constexpr float APO_MIN = 0.1f;
				float3 rx_vec = calc_rx_vector<SEQ>(initial_rx, c, t_position, Beamformer_Constants.pitches);
				float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.f_number);

				if(apo > APO_MIN)
				{
					float3 tx_vec = calc_tx_vector<SEQ>(initial_tx, t_position, Beamformer_Constants.pitches);

					float scan_index = calc_total_distance(tx_vec, rx_vec, focal_point.z) 
									   * Beamformer_Constants.samples_per_meter 
									   + Beamformer_Constants.delay_samples;

					scan_index = utils::clampf(scan_index, 1.0f, (float)Beamformer_Constants.sample_count - 2.0f);
					size_t channel_offset = Beamformer_Constants.channel_count * Beamformer_Constants.sample_count * t_signal + Beamformer_Constants.sample_count * c;
					
					cuComplex value = utils::cubic_spline(channel_offset, scan_index, rf_data);					

					// TODO: Compare performance of this vs multiplication
					// The compiler should make this a predicate op with no branching, confirm this
					//float hadamard_sign = ((hadamard_row >> readi_sub_signal) & 1u) ? -1.0f : 1.0f;
					//apo *= hadamard_sign;

					// If the hadamard bit is 1 we need to flip the sign of this sample.
					// XOR the bit with the sign bit of the apodization --> Avoid multiplication
					uint hadamard_bit = (hadamard_row >> readi_sub_signal) & 1u;
					apo = __uint_as_float(__float_as_uint(apo) ^ (hadamard_bit << 31));

					value = SCALE_F2(value, apo);
					total = ADD_V2(total, value);
					incoherent_sum += NORM_SQUARE_F2(value);
				}
			}
		}

		if(COMPARE_LT_V3(voxel_idx, Beamformer_Constants.voxel_dims))
		{
			float coherency_factor = NORM_SQUARE_F2(total) / incoherent_sum;
			coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
			coherency_factor = utils::clear_nan(coherency_factor);
			total = SCALE_F2(total, coherency_factor);

			size_t volume_offset = voxel_idx.z * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + voxel_idx.y * Beamformer_Constants.voxel_dims.x + voxel_idx.x;
			volume[volume_offset] = total;
		}
		
	}


	// Moving the channel loop outside the kernel. No atomics for now so only one execuation can be live at a time.
	template<FocalDirection DIR, EncodeMatrix READI> __global__ void
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

		float3 initial_tx = initial_tx_vec<HERCULES, DIR>(Beamformer_Constants.xdc_mins,
													 Beamformer_Constants.pitches,
													 vox_loc,
													 focal_point);

		float3 initial_rx = initial_rx_vec<HERCULES>(Beamformer_Constants.xdc_mins,
											   Beamformer_Constants.pitches,
											   vox_loc);
							   
		float tx_distance = copysignf(NORM_F3(initial_tx), initial_tx.z);
		float3 rx_vec = initial_rx;
		float incoherent_sum = 0.0f;
		cuComplex total = {0.0f, 0.0f};
		int total_transmits = Beamformer_Constants.tx_count * Beamformer_Constants.readi_group_count;
		int t_signal = 0;
		uint decode_bit = 0;
		
		for (int t_position = 0; t_position < total_transmits; t_position++)
		{
			if constexpr (READI == EncodeMatrix::HADAMARD)
			{
				int readi_sub_signal = t_position / Beamformer_Constants.tx_count;
				t_signal = t_position % Beamformer_Constants.tx_count;
				decode_bit = (decode_row >> readi_sub_signal) & 1u;
			}
			else if constexpr (READI == EncodeMatrix::WALSH)
			{
				t_signal = t_position / Beamformer_Constants.readi_group_count;
				int readi_sub_signal = t_position % Beamformer_Constants.readi_group_count;

				// Todo, make not terrible
				int test = (t_signal % 2 == 0) ? readi_sub_signal : (Beamformer_Constants.readi_group_count - 1 - readi_sub_signal);
				decode_bit = (decode_row >> test) & 1u;
			}
			else
			{
				t_signal = t_position;
			}
			
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
					size_t channel_offset = Beamformer_Constants.channel_count * Beamformer_Constants.sample_count * t_signal + Beamformer_Constants.sample_count * c;
					
					cuComplex value = utils::lerp_read(scan_index, rf_data + channel_offset);	
					// cuComplex value = utils::fast_cubic_spline(scan_index, rf_data + channel_offset);					

					if constexpr (READI != EncodeMatrix::NONE)
					{
						apo = __uint_as_float(__float_as_uint(apo) ^ (decode_bit << 31));
					}
					
					value = SCALE_F2(value, apo);
					total = ADD_V2(total, value);
					incoherent_sum += NORM_SQUARE_F2(value);
				}

				rx_vec.x += Beamformer_Constants.pitches.x;
			}
			rx_vec.x = initial_rx.x;
			rx_vec.y += Beamformer_Constants.pitches.y;
		}
		float coherency_factor = NORM_SQUARE_F2(total) / incoherent_sum;
		coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
		coherency_factor = utils::clear_nan(coherency_factor);
		total = SCALE_F2(total, coherency_factor);

		size_t volume_offset = voxel_idx.z * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + voxel_idx.y * Beamformer_Constants.voxel_dims.x + voxel_idx.x;
		volume[volume_offset] = total;
		
	}

}


