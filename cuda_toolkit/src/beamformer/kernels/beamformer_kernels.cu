
#include "../beamformer_constants.cuh"
#include "../beamformer_utils.cuh"
#include "beamformer_kernels.cuh"


namespace bf_kernels
{

__device__ __forceinline__ inline
cuComplex rotate_iq(cuComplex sample, int index, float demod_freq, float sample_freq)
{
	float t = TWO_PI_F * index * demod_freq / sample_freq;
	cuComplex demod = {cosf(t), -sinf(t)};
	return cuCmulf(sample, demod);
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

	float3 tx_vec = utils::calc_tx_distance(vox_loc, focal_point, Beamformer_Constants.focus_type);
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

				value = utils::lerp_read(scan_index, rfData + channel_offset);

				float apo = utils::f_num_apodization(abs(rx_vec.x), vox_loc.z, Beamformer_Constants.fn_rx);
				value = SCALE_V2(value, apo);

				// This acts as the final decoding step for the data within the readi group
				// If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
				value = SCALE_V2(value, hadamard_value);

				total = ADD_V2(total, value);
				incoherent_sum += NORM_SQUARE_V2(value);

				rx_vec.x += Beamformer_Constants.pitches.x;
			}
		
			rx_vec.x = starting_x;
			tx_vec.x += Beamformer_Constants.pitches.x;
		}
	}

    float coherent_sum = NORM_SQUARE_V2(total);

	float coherency_factor = coherent_sum / incoherent_sum;
	coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
	coherency_factor = utils::clear_nan(coherency_factor);

	total = SCALE_V2(total, coherency_factor);

	volume[volume_offset] = total;
}

__global__ void
uforces_beamform(const cuComplex* rfData, cuComplex* volume, const short* uforces_elements)
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

	float3 tx_vec = utils::calc_tx_distance(vox_loc, focal_point, Beamformer_Constants.focus_type);
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
	short last_element = uforces_elements[0];
	for (short t = 0; t < Beamformer_Constants.tx_count - 1; t++)
	{
		int element_diff = uforces_elements[t] - last_element;
		tx_vec.x += element_diff * Beamformer_Constants.pitches.x;
		last_element = uforces_elements[t];
		for (int e = 0; e < channel_count; e++)
		{
			size_t channel_offset = channel_count * sample_count * (t+1) + sample_count * e; // Add 1 to T because the first transmit is the discarded flash
			float total_distance = utils::total_path_length(tx_vec, rx_vec, focal_point.z, focal_distance_sign);
			float scan_index = total_distance * samples_per_meter + delay_samples;
			scan_index = utils::clampf(scan_index, 0.0f, (float)sample_count - 2.0f);

			value = utils::cubic_spline(channel_offset, scan_index, rfData);

			if (t == 0)
			{
				value = SCALE_V2(value, I_SQRT_128);
			}

			float apo = utils::f_num_apodization(abs(rx_vec.x), vox_loc.z, Beamformer_Constants.fn_rx);
			value = SCALE_V2(value, apo);

			total = ADD_V2(total, value);
			incoherent_sum += NORM_SQUARE_V2(value);

			rx_vec.x += Beamformer_Constants.pitches.x;
		}
	
		rx_vec.x = starting_x;
	}

    float coherent_sum = NORM_SQUARE_V2(total);

	float coherency_factor = coherent_sum / incoherent_sum;
	coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
	coherency_factor = utils::clear_nan(coherency_factor);

	total = SCALE_V2(total, coherency_factor);

	volume[volume_offset] = total;
}


__global__ void
hercules_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard_row)
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
	float3 focal_point = {0.0f, 0.0f, Beamformer_Constants.focal_point.z};

	float3 tx_vec = utils::calc_tx_distance(vox_loc, focal_point, Beamformer_Constants.focus_type);
	float3 rx_vec = {	Beamformer_Constants.xdc_mins.x - vox_loc.x + Beamformer_Constants.pitches.x / 2, 
						Beamformer_Constants.xdc_mins.y - vox_loc.y + Beamformer_Constants.pitches.y / 2, vox_loc.z };

	uint readi_group_count = Beamformer_Constants.readi_group_count;
	int delay_samples = Beamformer_Constants.delay_samples;

	cuComplex total = { 0.0f, 0.0f }, value;
	float incoherent_sum = 0.0f;

	float starting_x = rx_vec.x;
	//starting_x = -Beamformer_Constants.pitches.x/2;
	
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
				float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.fn_rx);
				static constexpr float APO_MIN = 0.1f;
				if(apo > APO_MIN)
				{
					size_t channel_offset = channel_count * sample_count * t + sample_count * e;
					float total_distance = utils::total_path_length(tx_vec, rx_vec, focal_point.z, focal_distance_sign);
					float scan_index = total_distance * samples_per_meter + delay_samples;
					scan_index = utils::clampf(scan_index, 0.0f, (float)sample_count - 2.0f);

					value = utils::cubic_spline(channel_offset, scan_index, rfData);

					
					value = SCALE_V2(value, apo);

					// This acts as the final decoding step for the data within the readi group
					// If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
					value = SCALE_V2(value, hadamard_value);

					total = ADD_V2(total, value);
					incoherent_sum += NORM_SQUARE_V2(value);
				}

				rx_vec.x += Beamformer_Constants.pitches.x;
			}
		
			rx_vec.x = starting_x;
			rx_vec.y += Beamformer_Constants.pitches.y;
		}
	}

    float coherent_sum = NORM_SQUARE_V2(total);

	float coherency_factor = coherent_sum / incoherent_sum;
	coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
	coherency_factor = utils::clear_nan(coherency_factor);

	total = SCALE_V2(total, coherency_factor);

	volume[volume_offset] = total;
}


__global__ void
walsh_hercules_beamform(const cuComplex* rfData, cuComplex* volume, const float* hadamard_row)
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
	float3 focal_point = {0.0f, 0.0f, Beamformer_Constants.focal_point.z};

	float3 tx_vec = utils::calc_tx_distance(vox_loc, focal_point, Beamformer_Constants.focus_type);
	float3 rx_vec = {	Beamformer_Constants.xdc_mins.x - vox_loc.x + Beamformer_Constants.pitches.x / 2, 
						Beamformer_Constants.xdc_mins.y - vox_loc.y + Beamformer_Constants.pitches.y / 2, vox_loc.z };

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
			// TODO: Figure this out 
			uint hadamard_index = (t%2 == 0) ? g : readi_group_count - 1 - g;
			float hadamard_value = hadamard_row[hadamard_index];
			for (int e = 0; e < channel_count; e++)
			{
				size_t channel_offset = channel_count * sample_count * t + sample_count * e;
				float total_distance = utils::total_path_length(tx_vec, rx_vec, focal_point.z, focal_distance_sign);
				float scan_index = total_distance * samples_per_meter + delay_samples;
				scan_index = utils::clampf(scan_index, 0.0f, (float)sample_count - 2.0f);

				value = utils::cubic_spline(channel_offset, scan_index, rfData);

				//  if (t == 0)
				//  {
				//      value = SCALE_V2(value, I_SQRT_128);
				//  }
				float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.fn_rx);
				value = SCALE_V2(value, apo);

				// This acts as the final decoding step for the data within the readi group
				// If readi is turned off this will just scan the first row of the hadamard matrix (all 1s)
				value = SCALE_V2(value, hadamard_value);

				total = ADD_V2(total, value);
				incoherent_sum += NORM_SQUARE_V2(value);

				rx_vec.x += Beamformer_Constants.pitches.x;
			}
		
			rx_vec.x = starting_x;
			rx_vec.y += Beamformer_Constants.pitches.y;
		}
	}

    float coherent_sum = NORM_SQUARE_V2(total);

	float coherency_factor = coherent_sum / incoherent_sum;
	coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
	coherency_factor = utils::clear_nan(coherency_factor);

	total = SCALE_V2(total, coherency_factor);

	volume[volume_offset] = total;
}


/*
* Dispatch blocks for each xz 16x16 patch, block wide calculate the delay for the center of the patch,
* load 64 samples either side of that and store the center idx
* each thread/pixel calculates its delay relative to the center idx.
*
*
* Repeat for each transmit, repeat for each channel
*
* Block Dims: 16x1x16
* Grid Dims: (vox_dims.x/16, voxel_dims.y/y_block_size, vox_dims.z/16)
*
* Getting this to work with basic hercules before anything else.
*/
__global__ void
block_beamform(const cuComplex* rfData, cuComplex* volume)
{

	static constexpr uint smem_padding = 2;
	__shared__ cuComplex shared_rf_data[128 + smem_padding * 2];

	cuComplex value_store[Y_BLOCK_SIZE] = {0.0f, 0.0f};

	// uint x_block = blockIdx.x;
	// uint y_block = blockIdx.y;
	// uint z_block = blockIdx.z;

	uint3 voxel_idx = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * Y_BLOCK_SIZE + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z };
	uint linear_thread_idx = threadIdx.z * blockDim.x + threadIdx.x;

	if(linear_thread_idx < 128 + smem_padding * 2)
	{
		shared_rf_data[linear_thread_idx] = {0.0f, 0.0f};
	}
	__syncthreads();

	float3 block_loc = {
		Beamformer_Constants.volume_mins.x + (blockIdx.x * blockDim.x + blockDim.x/2) * Beamformer_Constants.resolutions.x,
		Beamformer_Constants.volume_mins.y + (blockIdx.y * Y_BLOCK_SIZE + Y_BLOCK_SIZE/2) * Beamformer_Constants.resolutions.y,
		Beamformer_Constants.volume_mins.z + (blockIdx.z * blockDim.z + blockDim.z/2) * Beamformer_Constants.resolutions.z,
	};

	float3 vox_loc =
	{
		Beamformer_Constants.volume_mins.x + voxel_idx.x * Beamformer_Constants.resolutions.x,
		Beamformer_Constants.volume_mins.y + voxel_idx.y * Beamformer_Constants.resolutions.y,
		Beamformer_Constants.volume_mins.z + voxel_idx.z * Beamformer_Constants.resolutions.z,
	};

	//float3 tx_loc = { 0.0f, 0.0f, 0.0f};
	float3 rx_loc = {
		Beamformer_Constants.xdc_mins.x + Beamformer_Constants.pitches.x / 2,
		Beamformer_Constants.xdc_mins.y + Beamformer_Constants.pitches.y / 2,
		0.0f
	};

	float2 starting_rx_loc = {rx_loc.x, rx_loc.y};

	float3 focal_point = {0.0f, 0.0f, Beamformer_Constants.focal_point.z};

	// uint sample_count = Beamformer_Constants.sample_count;
	// uint channel_count = Beamformer_Constants.channel_count;
	// float samples_per_meter = Beamformer_Constants.samples_per_meter;
	// int delay_samples = Beamformer_Constants.delay_samples;

	cuComplex value;
	for (int t = 0; t < Beamformer_Constants.tx_count; t++)
	{
		rx_loc.y = starting_rx_loc.y + t * Beamformer_Constants.pitches.y;
		for (int e = 0; e < Beamformer_Constants.channel_count; e++)
		{
			rx_loc.x = starting_rx_loc.x + e * Beamformer_Constants.pitches.x;

			float3 tx_vec = utils::calc_tx_distance(block_loc, focal_point, Beamformer_Constants.focus_type);
			float3 rx_vec = SUB_V3(rx_loc, block_loc);

			//float total_block_distance = utils::total_path_length(tx_vec, rx_vec, focal_point.z, 1.0f);
			float block_scan_index = floorf(utils::total_path_length(tx_vec, rx_vec, focal_point.z, 1.0f) * Beamformer_Constants.samples_per_meter + Beamformer_Constants.delay_samples - 64.0f);

			block_scan_index = CLAMP(block_scan_index, 0.0f, (float)(Beamformer_Constants.sample_count - 128));

			if (linear_thread_idx < 128)
			{
				size_t channel_offset = Beamformer_Constants.channel_count * Beamformer_Constants.sample_count * t + Beamformer_Constants.sample_count * e;
				shared_rf_data[linear_thread_idx + smem_padding] = rfData[channel_offset + (int)(block_scan_index)+linear_thread_idx];
			}
			__syncthreads();

			for (int y = 0; y < Y_BLOCK_SIZE; y++)
			{
				float3 current_vox_loc = vox_loc;
				current_vox_loc.y += y * Beamformer_Constants.resolutions.y;

				tx_vec = utils::calc_tx_distance(current_vox_loc, focal_point, Beamformer_Constants.focus_type);
				rx_vec = SUB_V3(rx_loc, current_vox_loc);

				//float total_vox_distance = utils::total_path_length(tx_vec, rx_vec, focal_point.z, 1.0f);
				float vox_scan_index = utils::total_path_length(tx_vec, rx_vec, focal_point.z, 1.0f) * Beamformer_Constants.samples_per_meter + Beamformer_Constants.delay_samples;

				// The cubic spline needs 2 integer samples either side of the decimal index
				vox_scan_index = vox_scan_index - block_scan_index;
				
				vox_scan_index = CLAMP(vox_scan_index, 0.0f, 127.0f);
				value = utils::cubic_spline(smem_padding, vox_scan_index, shared_rf_data);

				float apo = utils::f_num_apodization(NORM_F2(rx_vec), vox_loc.z, Beamformer_Constants.fn_rx);
				value = SCALE_V2(value, apo);

				value_store[y] = ADD_V2(value_store[y], value);
			}

			__syncthreads(); 
		}
	}

	if (voxel_idx.x < Beamformer_Constants.voxel_dims.x && voxel_idx.z < Beamformer_Constants.voxel_dims.z)
	{

		for (uint y = 0; y < Y_BLOCK_SIZE && voxel_idx.y + y < Beamformer_Constants.voxel_dims.y; y++)
		{
			size_t volume_offset = (voxel_idx.z)*Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + (voxel_idx.y + y) * Beamformer_Constants.voxel_dims.x + voxel_idx.x;
			volume[volume_offset] = value_store[y];
		}
	}
	return;
}

__global__ void
tpw_beamform(const cuComplex* rfData, cuComplex* volume, const float* angles)
{
	uint3 voxel_idx = { threadIdx.x + blockIdx.x * blockDim.x,
						threadIdx.y + blockIdx.y * blockDim.y,
						threadIdx.z + blockIdx.z * blockDim.z };

	const float3 vox_loc = {
		Beamformer_Constants.volume_mins.x + voxel_idx.x * Beamformer_Constants.resolutions.x,
		Beamformer_Constants.volume_mins.y + voxel_idx.y * Beamformer_Constants.resolutions.y,
		Beamformer_Constants.volume_mins.z + voxel_idx.z * Beamformer_Constants.resolutions.z,
	};

	if(!COMPARE_LT_V3(voxel_idx, Beamformer_Constants.voxel_dims)) return;

	float3 rx_vec = { Beamformer_Constants.xdc_mins.x - vox_loc.x + Beamformer_Constants.pitches.x / 2, 0, vox_loc.z };

	float tpw_lat_pos = Beamformer_Constants.focus_type == FocusType::YZ_PLANE ? vox_loc.y : vox_loc.x;

	int delay_samples = Beamformer_Constants.delay_samples;
	cuComplex total = { 0.0f, 0.0f };
	float incoherent_sum = 0.0f;

	for (int c = 0; c < Beamformer_Constants.channel_count; c++)
	{
		float rx_dist = NORM_F3(rx_vec);
		float apo = utils::f_num_apodization(abs(rx_vec.x), vox_loc.z, Beamformer_Constants.fn_rx);
		rx_vec.x += Beamformer_Constants.pitches.x;

		if (apo < 0.1f) continue;
		
		for (int acq = 0; acq < Beamformer_Constants.tx_count; acq++)
		{	
			float tx_dist = tpw_lat_pos * sinf(angles[acq]) + vox_loc.z * cosf(angles[acq]);

			// Distance calculations assume t=0 occurs when the center element transmits,
			// but VSX considereds 0 when the first element transmits, so for each acq we need to adjust
			//int angle_delay = (int)roundf(Beamformer_Constants.xdc_maxes.x * tanf(abs(angles[acq])) * Beamformer_Constants.samples_per_meter);

			float scan_index = (rx_dist + tx_dist) * Beamformer_Constants.samples_per_meter + delay_samples;
			scan_index = utils::clampf(scan_index, 1.0f, (float)Beamformer_Constants.sample_count - 2.0f);

			size_t channel_offset = Beamformer_Constants.channel_count * Beamformer_Constants.sample_count * acq + Beamformer_Constants.sample_count * c;

			cuComplex value = utils::fast_cubic_spline(scan_index, rfData + channel_offset);
			value = rotate_iq(value, scan_index, Beamformer_Constants.center_freq, Beamformer_Constants.sample_freq);
			value = SCALE_V2(value, apo);
			total = ADD_V2(total, value);
			incoherent_sum += NORM_SQUARE_V2(value);
		}		
	}

	float coherency_factor = NORM_SQUARE_V2(total) / incoherent_sum;
	coherency_factor = powf(coherency_factor, Beamformer_Constants.coherency_weighting);
	coherency_factor = utils::clear_nan(coherency_factor);
	total = SCALE_V2(total, coherency_factor);

	size_t volume_offset = voxel_idx.z * Beamformer_Constants.voxel_dims.x * Beamformer_Constants.voxel_dims.y + voxel_idx.y * Beamformer_Constants.voxel_dims.x + voxel_idx.x;
	volume[volume_offset] = total;

}

}

