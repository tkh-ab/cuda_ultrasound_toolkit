#include <cstring>

#include "../rf_processing/hadamard/hadamard_decoder.h"
#include "kernels/beamformer_kernels.cuh"
#include "beamformer.h"

bool
Beamformer::_params_to_constants(const CudaBeamformerParameters& bp)
{
    bf_kernels::BeamformerConstants constants;
    constants.sample_count = bp.dec_data_dim[0];
    constants.channel_count = bp.dec_data_dim[1];
    constants.tx_count = bp.dec_data_dim[2];

    constants.xdc_mins = {-bp.xdc_transform[12], -bp.xdc_transform[13]};
    constants.xdc_maxes = {bp.xdc_transform[12], bp.xdc_transform[13]};

    constants.samples_per_meter = bp.sampling_frequency / bp.speed_of_sound;

    constants.pitches = {bp.xdc_element_pitch[0], bp.xdc_element_pitch[1]};
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
    constants.f_number = bp.f_number;

    constants.mixes_count = static_cast<u8>(bp.mixes_count);
    constants.mixes_offset = static_cast<u8>(bp.mixes_offset);
	
    constants.readi_group_count = static_cast<u8>(bp.readi_group_count);
    if(constants.readi_group_count == 0)
    {
        constants.readi_group_count = 1; // If no groups, just use one
    }
    constants.readi_group_id = static_cast<u8>(bp.readi_group_id);
    constants.readi_order = bp.readi_ordering;

    float3 focal_point = {0.0f, 0.0f, bp.focal_depths[0]};
    constants.focal_point = focal_point;
    if(focal_point.z == INFINITY)
    {
        constants.focal_direction = bf_kernels::FocalDirection::PLANE;
    }
    else if(bp.das_shader_id == SequenceId::HERCULES 
        || bp.das_shader_id == SequenceId::UHURCULES
        || bp.das_shader_id == SequenceId::EPIC_UHERCULES)
    {
        constants.focal_direction = bf_kernels::FocalDirection::YZ_PLANE;
    }
    else
    {
        constants.focal_direction = bf_kernels::FocalDirection::XZ_PLANE;
    }
    

    bool readi_count_changed = (_constants.readi_group_count != bp.readi_group_count ||
                                _constants.readi_order != bp.readi_ordering);

    std::memcpy(&_constants, &constants, sizeof(bf_kernels::BeamformerConstants));
    return readi_count_changed;
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
            _d_beamformer_hadamard, _constants.readi_group_count, _constants.readi_order))
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

	if (!bf_kernels::copy_kernel_constants(_constants))
	{
		std::cerr << "Beamformer: Failed to copy kernel constants." << std::endl;
		return false;
	}
	// Perform the beamforming operation
	return _per_voxel_beamform(d_input, d_output);
}

bool
Beamformer::_per_voxel_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume)
{
    std::cout << "Starting beamform." << std::endl;

    float* d_hadamard_row = _d_beamformer_hadamard;
    if(_constants.readi_group_count > 1)
    {
        // We just want the relevant row for this group
        d_hadamard_row += _constants.readi_group_id * _constants.readi_group_count;
    }

    uint3 vox_counts = _constants.voxel_dims;
    uint xy_count = vox_counts.x * vox_counts.y;
    dim3 grid_dim = { (xy_count + MAX_THREADS_PER_BLOCK -1) / MAX_THREADS_PER_BLOCK, vox_counts.z, 1 };
    dim3 block_dim = { MAX_THREADS_PER_BLOCK, 1, 1 };

    auto start = std::chrono::high_resolution_clock::now();

    if (_constants.readi_order == ReadiOrdering::WALSH)
    {
        bf_kernels::walsh_beamform << < grid_dim, block_dim >> > (d_rf_buffer, d_volume, d_hadamard_row);
    }
    else
    {
        bf_kernels::forces_beamform << < grid_dim, block_dim >> > (d_rf_buffer, d_volume, d_hadamard_row);
    }
    

    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;


    return true;
}