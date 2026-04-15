#include <cstring>

#include "../rf_processing/hadamard/hadamard_decoder.h"
#include "kernels/beamformer_kernels.cuh"
#include "kernels/templated_beamformer.cuh"
#include "beamformer.h"

__device__ __constant__ bf_kernels::BeamformerConstants Beamformer_Constants;

__host__ bool
copy_kernel_constants(const bf_kernels::BeamformerConstants& constants)
{
	CUDA_RETURN_IF_ERROR(cudaMemcpyToSymbol(Beamformer_Constants, &constants, sizeof(bf_kernels::BeamformerConstants)));
	return true;
}

bool
Beamformer::_params_to_constants(const CudaBeamformerParameters& bp)
{
    bf_kernels::BeamformerConstants constants;
    constants.sample_count = bp.dec_data_dim[0];
    constants.channel_count = bp.dec_data_dim[1];
    constants.tx_count = bp.dec_data_dim[2];

    constants.samples_per_meter = bp.sampling_frequency / bp.speed_of_sound;
	constants.lambda_0 = bp.speed_of_sound / bp.center_frequency;

    float2 pitches = {bp.xdc_element_pitch[0], bp.xdc_element_pitch[1]};
	constants.pitches = pitches;

	// Move the bounds in from the edge to the center of the first and last element
	pitches = SCALE_V2(pitches, 0.5f);
	constants.xdc_mins = {-bp.xdc_transform[12] + pitches.x, -bp.xdc_transform[13] + pitches.y};
    constants.xdc_maxes = {bp.xdc_transform[12] - pitches.x, bp.xdc_transform[13] - pitches.y};

    constants.delay_samples = static_cast<int>((bp.time_offset * bp.sampling_frequency));
    constants.sequence = bp.das_shader_id;

    constants.voxel_dims = {bp.output_points[0], bp.output_points[1], bp.output_points[2]};
    constants.volume_mins = {bp.output_min_coordinate[0], bp.output_min_coordinate[1], bp.output_min_coordinate[2]};

    float lateral_resolution = (bp.output_max_coordinate[0] - bp.output_min_coordinate[0])
                                / (bp.output_points[0] - 1);

    float elevation_resolution = lateral_resolution;
    float axial_resolution = (bp.output_max_coordinate[2] - bp.output_min_coordinate[2])
                                / (bp.output_points[2] - 1);

    constants.resolutions = {lateral_resolution, elevation_resolution, axial_resolution};
    constants.fn_rx = bp.fn_rx;
	constants.fn_tx = bp.fn_tx;

    constants.mixes_count = static_cast<u8>(bp.mixes_count);
    constants.mixes_offset = static_cast<u8>(bp.mixes_offset);
	
    constants.readi_group_count = static_cast<u8>(bp.readi_group_count);
    if(constants.readi_group_count == 0)
    {
        constants.readi_group_count = 1; // If no groups, just use one
    }
    constants.readi_group_id = static_cast<u8>(bp.readi_group_id);
    constants.encoded_matrix = bp.decode;

    float3 focal_point = {0.0f, 0.0f, bp.focal_depths[0]};
    constants.focal_point = focal_point;
    if(isinf(constants.focal_point.z))
    {
        constants.focal_direction = bf_kernels::FocalDirection::PLANE_FOCUS;
        constants.focal_point.z = 0.0f;
    }
    else if(bp.das_shader_id == SequenceId::HERCULES 
        || bp.das_shader_id == SequenceId::UHURCULES
        || bp.das_shader_id == SequenceId::EPIC_UHERCULES)
    {
        constants.focal_direction = bf_kernels::FocalDirection::YZ_FOCUS;
    }
    else
    {
        constants.focal_direction = bf_kernels::FocalDirection::XZ_FOCUS;
    }

	constants.coherency_weighting = min(max(bp.coherency_weighting, 0.0f), 1.0f);

	constants.apo_type = bp.apo_type;
	constants.to_power = bp.to_power;

    bool readi_matrix_changed = (_constants.readi_group_count != bp.readi_group_count ||
                                _constants.encoded_matrix != bp.decode);

    std::memcpy(&_constants, &constants, sizeof(bf_kernels::BeamformerConstants));
    return readi_matrix_changed;
}

bool
Beamformer::setup_beamformer(const CudaBeamformerParameters& bp)
{
    bool readi_count_changed = _params_to_constants(bp);

    if(!readi_count_changed && _d_beamformer_hadamard)
    {
        // No change in parameters and already initialized
        return true;
    }

	CUDA_NULL_FREE(_d_beamformer_hadamard);

    size_t hadamard_size = _constants.readi_group_count * _constants.readi_group_count * sizeof(float);
    CUDA_RETURN_IF_ERROR(cudaMalloc(&_d_beamformer_hadamard, hadamard_size));
    if(_constants.readi_group_count == 1)
    {
        float one = 1.0f;
        
        CUDA_RETURN_IF_ERROR(cudaMemcpy(_d_beamformer_hadamard, &one, sizeof(float), cudaMemcpyHostToDevice));
    }
    else
    {
        if(! decoding::HadamardDecoder::generate_hadamard(
            _d_beamformer_hadamard, _constants.readi_group_count, _constants.encoded_matrix))
        {
            std::cerr << "Beamformer: Failed to generate Hadamard matrix." << std::endl;
            return false;
        }
    }

    return true;
}

bool
Beamformer::beamform(cuComplex* d_input, cuComplex* d_output, const CudaBeamformerParameters& bp)
{
	if(!setup_beamformer(bp))
	{
		std::cerr << "Beamformer: Failed to setup beamformer." << std::endl;
		return false;
	}

	if(!d_input || !d_output)
	{
		std::cerr << "Beamformer: Invalid input or output buffer." << std::endl;
		return false;
	}

	if (!copy_kernel_constants(_constants))
	{
		std::cerr << "Beamformer: Failed to copy kernel constants." << std::endl;
		return false;
	}

	bool result = false;
	if(bp.das_shader_id == SequenceId::UFORCES)
	{
		
		if(bp.sparse_elements[0] == -1)
		{
			std::cerr << "Beamformer: UFORCES requires sparse elements to be defined." << std::endl;	
			return false;
		}

		short* d_uforces_elements = nullptr;
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_uforces_elements, sizeof(short) * bp.dec_data_dim[2]));
		CUDA_RETURN_IF_ERROR(cudaMemcpy((void*)d_uforces_elements, bp.sparse_elements, sizeof(short) * bp.dec_data_dim[2], cudaMemcpyHostToDevice));

		std::cout << "UFORCES beamforming." << std::endl;
		result = _uforces_beamform(d_input, d_output, d_uforces_elements);
		CUDA_NULL_FREE(d_uforces_elements);
	}
	else if (bp.das_shader_id == SequenceId::FORCES)
	{
		result = _test_new_forces_beamform(d_input, d_output);
	}
	else if (bp.das_shader_id == SequenceId::HERCULES)
	{
		result = _test_new_herc_beamform(d_input, d_output);
	}
	else
	{
		std::cerr << "Beamformer: Unsupported sequence ID " << static_cast<int>(bp.das_shader_id) << std::endl;
		return false;
	}

	return result;
}

bool
Beamformer::_uforces_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume, const short* uforces_elements)
{
    std::cout << "Starting beamform." << std::endl;
    uint3 vox_counts = _constants.voxel_dims;
    uint xy_count = vox_counts.x * vox_counts.y;
    dim3 grid_dim = { (xy_count + MAX_THREADS_PER_BLOCK -1) / MAX_THREADS_PER_BLOCK, vox_counts.z, 1 };
    dim3 block_dim = { MAX_THREADS_PER_BLOCK, 1, 1 };

    auto start = std::chrono::high_resolution_clock::now();

    bf_kernels::uforces_beamform << < grid_dim, block_dim >> > (d_rf_buffer, d_volume, uforces_elements);

    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;


    return true;
}

bool
Beamformer::_test_new_herc_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume)
{

	std::cout << "Starting HERCULES beamform." << std::endl;
	uint3 vox_counts = _constants.voxel_dims;
	static constexpr dim3 test_block_dims = {8,1,32};
	dim3 block_dims = test_block_dims;
	dim3 grid_dims = { UINT_DIV_CEIL(vox_counts.x, block_dims.x),
					   UINT_DIV_CEIL(vox_counts.y, block_dims.y),
					   UINT_DIV_CEIL(vox_counts.z, block_dims.z) };
	auto start = std::chrono::high_resolution_clock::now();
	u64 compact_hadamard_row = 0;

	// Todo: make a better dispatcher
	if(_constants.readi_group_count > 1)
	{
		uint* hadamard_row = (uint*) malloc(_constants.readi_group_count * sizeof(uint));

		CUDA_RETURN_IF_ERROR(cudaMemcpy((void*)hadamard_row, 
		(void*)(_d_beamformer_hadamard + (_constants.readi_group_id * _constants.readi_group_count)),
		 _constants.readi_group_count * sizeof(uint), cudaMemcpyDeviceToHost));
		// For READI decoding we need the hadamard row corresponding with the current group
		// Packing it up like this lets every thread hold it locally in registers.
        //d_hadamard_row += _constants.readi_group_id * _constants.readi_group_count;
		for(int i = 0; i < _constants.readi_group_count; i++)
		{
			compact_hadamard_row |= (hadamard_row[i] >> 31) << i;
		}
		free(hadamard_row);

		if(_constants.encoded_matrix == EncodingMatrix::WALSH)
		{
			if(_constants.focal_direction == bf_kernels::FocalDirection::YZ_FOCUS)
			{
				bf_kernels::hercules_beamform_new<bf_kernels::FocalDirection::YZ_FOCUS, EncodingMatrix::WALSH><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
			}
			else if(_constants.focal_direction == bf_kernels::FocalDirection::PLANE_FOCUS)
			{
				bf_kernels::hercules_beamform_new<bf_kernels::FocalDirection::PLANE_FOCUS, EncodingMatrix::WALSH><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
			}
			else
			{
				std::cerr << "Invalid focus for HERCULES." << std::endl;
				return false;
			}
		}
		else if (_constants.encoded_matrix == EncodingMatrix::HADAMARD)
		{
			if(_constants.focal_direction == bf_kernels::FocalDirection::YZ_FOCUS)
			{
				bf_kernels::hercules_beamform_new<bf_kernels::FocalDirection::YZ_FOCUS, EncodingMatrix::HADAMARD><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
			}
			else if(_constants.focal_direction == bf_kernels::FocalDirection::PLANE_FOCUS)
			{
				bf_kernels::hercules_beamform_new<bf_kernels::FocalDirection::PLANE_FOCUS, EncodingMatrix::HADAMARD><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
			}
			else
			{
				std::cerr << "Invalid focus for HERCULES." << std::endl;
				return false;
			}
		}
	}
	else
	{
		if(_constants.focal_direction == bf_kernels::FocalDirection::YZ_FOCUS)
		{
			bf_kernels::hercules_beamform_new<bf_kernels::FocalDirection::YZ_FOCUS, EncodingMatrix::NONE><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
		}
		else if(_constants.focal_direction == bf_kernels::FocalDirection::PLANE_FOCUS)
		{
			bf_kernels::hercules_beamform_new<bf_kernels::FocalDirection::PLANE_FOCUS, EncodingMatrix::NONE><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
		}
		else
		{
			std::cerr << "Invalid focus for HERCULES." << std::endl;
			return false;
		}
	}
	
	
	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
	
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;


	return true;
}



bool
Beamformer::_test_new_forces_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume)
{

	std::cout << "Starting FORCES beamform." << std::endl;
	uint3 vox_counts = _constants.voxel_dims;
	static constexpr dim3 test_block_dims = {8,1,32};
	dim3 block_dims = test_block_dims;
	dim3 grid_dims = { UINT_DIV_CEIL(vox_counts.x, block_dims.x),
					   UINT_DIV_CEIL(vox_counts.y, block_dims.y),
					   UINT_DIV_CEIL(vox_counts.z, block_dims.z) };
	auto start = std::chrono::high_resolution_clock::now();
	u64 compact_hadamard_row = 0;

	size_t vol_size = vox_counts.x * vox_counts.y * vox_counts.z * sizeof(cuComplex);
	CUDA_RETURN_IF_ERROR(cudaMemset(d_volume, 0x00, vol_size));

	std::cout << "Using apo: " << _constants.apo_type << std::endl;
	// Todo: make a better dispatcher
	if(_constants.readi_group_count > 1)
	{
		uint* hadamard_row = (uint*) malloc(_constants.readi_group_count * sizeof(uint));

		CUDA_RETURN_IF_ERROR(cudaMemcpy((void*)hadamard_row, 
		(void*)(_d_beamformer_hadamard + (_constants.readi_group_id * _constants.readi_group_count)),
		 _constants.readi_group_count * sizeof(uint), cudaMemcpyDeviceToHost));
		// For READI decoding we need the hadamard row corresponding with the current group
		// Packing it up like this lets every thread hold it locally in registers.
        //d_hadamard_row += _constants.readi_group_id * _constants.readi_group_count;
		for(int i = 0; i < _constants.readi_group_count; i++)
		{
			compact_hadamard_row |= (u64)(hadamard_row[i] >> 31) << i;
		}
		free(hadamard_row);

		
		if (_constants.encoded_matrix == EncodingMatrix::HADAMARD)
		{
			switch (_constants.apo_type)
			{
				case ApoType::RX_HANN:
					bf_kernels::forces_beamform_new<EncodingMatrix::HADAMARD, ApoType::RX_HANN><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
				break;
				case ApoType::TX_TO_SIN:
					bf_kernels::forces_beamform_new<EncodingMatrix::HADAMARD, ApoType::TX_TO_SIN><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
				break;
				case ApoType::RX_TO_SIN:
					bf_kernels::forces_beamform_new<EncodingMatrix::HADAMARD, ApoType::RX_TO_SIN><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
				break;
				case ApoType::BOTH_TO_SIN:
					bf_kernels::forces_beamform_new<EncodingMatrix::HADAMARD, ApoType::BOTH_TO_SIN><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
				break;
				default:
					std::cerr << "Invalid apodization type for FORCES." << std::endl;
					return false;
			}
		}
		else
		{
			std::cerr << "FORCES only supports Hadamard encoding at this time." << std::endl;
			return false;
		}
	}
	else
	{
		switch (_constants.apo_type)
		{
			case ApoType::RX_HANN:
				bf_kernels::forces_beamform_new<EncodingMatrix::NONE, ApoType::RX_HANN><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
			break;
			case ApoType::TX_TO_SIN:
				bf_kernels::forces_beamform_new<EncodingMatrix::NONE, ApoType::TX_TO_SIN><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
			break;
			case ApoType::RX_TO_SIN:
				bf_kernels::forces_beamform_new<EncodingMatrix::NONE, ApoType::RX_TO_SIN><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
			break;
			case ApoType::BOTH_TO_SIN:
				bf_kernels::forces_beamform_new<EncodingMatrix::NONE, ApoType::BOTH_TO_SIN><<<grid_dims, block_dims>>>(d_rf_buffer, d_volume, compact_hadamard_row);
			break;
			default:
				std::cerr << "Invalid apodization type for FORCES." << std::endl;
				return false;
		}
	}
	
	
	CUDA_RETURN_IF_ERROR(cudaGetLastError());
	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;


	return true;
}

