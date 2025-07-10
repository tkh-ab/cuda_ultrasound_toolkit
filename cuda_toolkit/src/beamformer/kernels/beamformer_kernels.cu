
#include "../beamformer_constants.cuh"
#include "../beamformer_utils.cuh"
#include "beamformer_kernels.cuh"


namespace bf_kernels
{

__global__ void
walsh_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard_row)
{
	uint xy_voxel = threadIdx.x + blockIdx.x * blockDim.x;
	if (xy_voxel > Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y)
	{
		return;
	}

	uint3 voxel_idx = { xy_voxel % Beamformer_Constants.voxel_dims.x, xy_voxel / Beamformer_Constants.voxel_dims.x, blockIdx.y };
	size_t volume_offset = voxel_idx.z * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + voxel_idx.y * Beamformer_Constants.voxel_dims.x + voxel_idx.x;

	const float3 vox_loc =
	{
		Beamformer_Constants.volume_mins.x + voxel_idx.x * Beamformer_Constants.resolutions.x,
		Beamformer_Constants.volume_mins.y + voxel_idx.y * Beamformer_Constants.resolutions.y,
		Beamformer_Constants.volume_mins.z + voxel_idx.z * Beamformer_Constants.resolutions.z,
	};

	// If the voxel is out of the f_number defined range for all elements skip it
	// if (!utils::check_ranges(vox_loc, Beamformer_Constants.f_number, Beamformer_Constants.xdc_maxes)) return;
	float3 focal_point = Beamformer_Constants.focal_point;
	if (Beamformer_Constants.sequence == SequenceId::FORCES)
	{
		focal_point.x = Beamformer_Constants.xdc_mins.x + Beamformer_Constants.pitches.x / 2;
		focal_point.z = 0; // Ignoring the elevational focus as it is out of plane
	}

	float3 tx_vec = utils::calc_tx_distance(vox_loc, focal_point, Beamformer_Constants.focal_direction);
	float3 rx_vec = { Beamformer_Constants.xdc_mins.x - vox_loc.x + Beamformer_Constants.pitches.x / 2, Beamformer_Constants.xdc_mins.y - vox_loc.y + Beamformer_Constants.pitches.y / 2, vox_loc.z };

	if (Beamformer_Constants.sequence == SequenceId::FORCES)
	{
		rx_vec.y = 0;
	}

	uint readi_group_count = Beamformer_Constants.readi_group_count;
	int delay_samples = Beamformer_Constants.delay_samples;

	cuComplex total = { 0.0f, 0.0f }, value;
	float incoherent_sum = 0.0f;

	float starting_x = rx_vec.x;
	
	uint sample_count = Beamformer_Constants.sample_count;
	uint channel_count = Beamformer_Constants.channel_count;
	float samples_per_meter = Beamformer_Constants.samples_per_meter;
	float focal_distance_sign = copysignf(1.0f, vox_loc.z - focal_point.z);
	for (int t = 0; t < Beamformer_Constants.tx_count; t++)
	{
		for(int g = 0; g < readi_group_count; g++)
		{
		// With walsh matricies every other tx needs to be flipped for some reason
		uint hadamard_index = (t%2 == 0) ? g : readi_group_count - 1 - g;
		float hadamard_value = hadamard_row[hadamard_index];
			for (int e = 0; e < channel_count; e++)
			{
				size_t channel_offset = channel_count * sample_count * t + sample_count * e;
				float total_distance = utils::total_path_length(tx_vec, rx_vec, focal_point.z, focal_distance_sign);
				float scan_index = total_distance * samples_per_meter + delay_samples;
				scan_index = utils::clampf(scan_index, 0.0f, (float)sample_count - 2.0f);

				value = utils::cubic_spline(channel_offset, scan_index, rfData);
				float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.f_number);
				value = SCALE_F2(value, apo);

				value = SCALE_F2(value, hadamard_value);

				total = ADD_F2(total, value);
				incoherent_sum += NORM_SQUARE_F2(value);

				rx_vec.x += Beamformer_Constants.pitches.x;
			}
		
			rx_vec.x = starting_x;

			if (Beamformer_Constants.sequence == SequenceId::HERCULES)
			{
				rx_vec.y += Beamformer_Constants.pitches.x;
			}
			else if (Beamformer_Constants.sequence == SequenceId::FORCES)
			{
				tx_vec.x += Beamformer_Constants.pitches.x;
			}
		}
	}

//            float coherent_sum = NORM_SQUARE_F2(total);

	//float coherency_factor = coherent_sum / (incoherent_sum * total_used_channels);
	//total = SCALE_F2(total, coherency_factor);

	volume[volume_offset] = total;
}

__global__ void
per_voxel_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard_row)
{
	uint xy_voxel = threadIdx.x + blockIdx.x * blockDim.x;
	if (xy_voxel > Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y)
	{
		return;
	}

	uint3 voxel_idx = { xy_voxel % Beamformer_Constants.voxel_dims.x, xy_voxel / Beamformer_Constants.voxel_dims.x, blockIdx.y };
	size_t volume_offset = voxel_idx.z * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + voxel_idx.y * Beamformer_Constants.voxel_dims.x + voxel_idx.x;

	const float3 vox_loc =
	{
		Beamformer_Constants.volume_mins.x + voxel_idx.x * Beamformer_Constants.resolutions.x,
		Beamformer_Constants.volume_mins.y + voxel_idx.y * Beamformer_Constants.resolutions.y,
		Beamformer_Constants.volume_mins.z + voxel_idx.z * Beamformer_Constants.resolutions.z,
	};

	// If the voxel is out of the f_number defined range for all elements skip it
	// if (!utils::check_ranges(vox_loc, Beamformer_Constants.f_number, Beamformer_Constants.xdc_maxes)) return;
	float3 focal_point = Beamformer_Constants.focal_point;
	if (Beamformer_Constants.sequence == SequenceId::FORCES)
	{
		focal_point.x = Beamformer_Constants.xdc_mins.x + Beamformer_Constants.pitches.x / 2;
		focal_point.z = 0; // Ignoring the elevational focus as it is out of plane
	}

	float3 tx_vec = utils::calc_tx_distance(vox_loc, focal_point, Beamformer_Constants.focal_direction);
	float3 rx_vec = { Beamformer_Constants.xdc_mins.x - vox_loc.x + Beamformer_Constants.pitches.x / 2, Beamformer_Constants.xdc_mins.y - vox_loc.y + Beamformer_Constants.pitches.y / 2, vox_loc.z };

	if (Beamformer_Constants.sequence == SequenceId::FORCES)
	{
		rx_vec.y = 0;
	}

	uint readi_group_count = Beamformer_Constants.readi_group_count;
	int delay_samples = Beamformer_Constants.delay_samples;

	cuComplex total = { 0.0f, 0.0f }, value;
	float incoherent_sum = 0.0f;

	float starting_x = rx_vec.x;
	
	uint sample_count = Beamformer_Constants.sample_count;
	uint channel_count = Beamformer_Constants.channel_count;
	float samples_per_meter = Beamformer_Constants.samples_per_meter;
	float focal_distance_sign = copysignf(1.0f, vox_loc.z - focal_point.z);
	for (int g = 0; g < readi_group_count; g++)
	{
		float hadamard_value = hadamard_row[g];
		for (int t = 0; t < Beamformer_Constants.tx_count; t++)
		{
			for (int e = 0; e < channel_count; e++)
			{
				size_t channel_offset = channel_count * sample_count * t + sample_count * e;
				float total_distance = utils::total_path_length(tx_vec, rx_vec, focal_point.z, focal_distance_sign);
				float scan_index = total_distance * samples_per_meter + delay_samples;
				scan_index = utils::clampf(scan_index, 0.0f, (float)sample_count - 2.0f);

				value = utils::cubic_spline(channel_offset, scan_index, rfData);

				// if (t == 0)
				// {
				//     value = SCALE_F2(value, I_SQRT_128);
				// }

				float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.f_number);
				value = SCALE_F2(value, apo);

				// This acts as the final decoding step for the data within the readi group
				// If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
				value = SCALE_F2(value, hadamard_value);

				total = ADD_F2(total, value);
				incoherent_sum += NORM_SQUARE_F2(value);

				rx_vec.x += Beamformer_Constants.pitches.x;
			}
		
			rx_vec.x = starting_x;

			if (Beamformer_Constants.sequence == SequenceId::HERCULES)
			{
				rx_vec.y += Beamformer_Constants.pitches.x;
			}
			else if (Beamformer_Constants.sequence == SequenceId::FORCES)
			{
				tx_vec.x += Beamformer_Constants.pitches.x;
			}
		}
	}

//            float coherent_sum = NORM_SQUARE_F2(total);

	//float coherency_factor = coherent_sum / (incoherent_sum * total_used_channels);5
	//total = SCALE_F2(total, coherency_factor);

	volume[volume_offset] = total;
}


__global__ void
forces_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard_row)
{
	uint xy_voxel = threadIdx.x + blockIdx.x * blockDim.x;
	if (xy_voxel > Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y)
	{
		return;
	}

	uint3 voxel_idx = { xy_voxel % Beamformer_Constants.voxel_dims.x, xy_voxel / Beamformer_Constants.voxel_dims.x, blockIdx.y };
	size_t volume_offset = voxel_idx.z * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + voxel_idx.y * Beamformer_Constants.voxel_dims.x + voxel_idx.x;

	const float3 vox_loc =
	{
		Beamformer_Constants.volume_mins.x + voxel_idx.x * Beamformer_Constants.resolutions.x,
		Beamformer_Constants.volume_mins.y + voxel_idx.y * Beamformer_Constants.resolutions.y,
		Beamformer_Constants.volume_mins.z + voxel_idx.z * Beamformer_Constants.resolutions.z,
	};

	// If the voxel is out of the f_number defined range for all elements skip it
	// if (!utils::check_ranges(vox_loc, Beamformer_Constants.f_number, Beamformer_Constants.xdc_maxes)) return;
	float3 focal_point = {Beamformer_Constants.xdc_mins.x + Beamformer_Constants.pitches.x / 2, vox_loc.y, 0.0f};

	float3 tx_vec = utils::calc_tx_distance(vox_loc, focal_point, Beamformer_Constants.focal_direction);
	float3 rx_vec = { Beamformer_Constants.xdc_mins.x - vox_loc.x + Beamformer_Constants.pitches.x / 2, 0, vox_loc.z };

	uint readi_group_count = Beamformer_Constants.readi_group_count;
	int delay_samples = Beamformer_Constants.delay_samples;

	cuComplex total = { 0.0f, 0.0f }, value;
	float incoherent_sum = 0.0f;

	float starting_x = rx_vec.x;
	
	uint sample_count = Beamformer_Constants.sample_count;
	uint channel_count = Beamformer_Constants.channel_count;
	float samples_per_meter = Beamformer_Constants.samples_per_meter;
	float focal_distance_sign = copysignf(1.0f, vox_loc.z - focal_point.z);
	for (int g = 0; g < readi_group_count; g++)
	{
		float hadamard_value = hadamard_row[g];
		for (int t = 0; t < Beamformer_Constants.tx_count; t++)
		{
			for (int e = 0; e < channel_count; e++)
			{
				size_t channel_offset = channel_count * sample_count * t + sample_count * e;
				float total_distance = utils::total_path_length(tx_vec, rx_vec, focal_point.z, focal_distance_sign);
				float scan_index = total_distance * samples_per_meter + delay_samples;
				scan_index = utils::clampf(scan_index, 0.0f, (float)sample_count - 2.0f);

				value = utils::cubic_spline(channel_offset, scan_index, rfData);

				if (t == 0)
				{
				    value = SCALE_F2(value, I_SQRT_128);
				}

				float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.f_number);
				value = SCALE_F2(value, apo);

				// This acts as the final decoding step for the data within the readi group
				// If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
				value = SCALE_F2(value, hadamard_value);

				total = ADD_F2(total, value);
				incoherent_sum += NORM_SQUARE_F2(value);

				rx_vec.x += Beamformer_Constants.pitches.x;
			}
		
			rx_vec.x = starting_x;
			tx_vec.x += Beamformer_Constants.pitches.x;
		}
	}

    float coherent_sum = NORM_SQUARE_F2(total);

	float coherency_factor = coherent_sum / incoherent_sum;

	coherency_factor = powf(coherency_factor, 1/10.f);

	total = SCALE_F2(total, coherency_factor);

	volume[volume_offset] = total;
}



__host__ bool
copy_kernel_constants(const BeamformerConstants& constants)
{
	CUDA_RETURN_IF_ERROR(cudaMemcpyToSymbol(Beamformer_Constants, &constants, sizeof(constants)));
	return true;
}

}

