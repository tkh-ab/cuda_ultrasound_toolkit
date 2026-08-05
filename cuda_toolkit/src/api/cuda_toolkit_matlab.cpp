#include <vector>
#include <chrono>

#include "../defs.h"
#include "../public/cuda_toolkit.hpp"
#include "../public/cuda_toolkit_matlab.h"


void 
beamform_i16_matlab( const short* data, CudaBeamformerParameters bp, float* output)
{
    size_t data_size = bp.rf_raw_dim[0] * bp.rf_raw_dim[1] * sizeof(short);
	uint output_size = bp.output_points[0] * bp.output_points[1] * bp.output_points[2] * sizeof(float) * 2; 

    std::span<const u8> input_data(reinterpret_cast<const u8*>(data), data_size);
	std::span<u8> output_data(reinterpret_cast<u8*>(output), output_size);

	if (!cuda_toolkit::beamform(input_data, output_data, bp))
	{
		std::cerr << "Error: Beamforming failed." << std::endl;
	}
}

void 
beamform_f32_matlab( const float* data, CudaBeamformerParameters bp, float* output)
{
    size_t data_size = bp.rf_raw_dim[0] * bp.rf_raw_dim[1] * sizeof(float);
	uint output_size = bp.output_points[0] * bp.output_points[1] * bp.output_points[2] * sizeof(float) * 2; 

    std::span<const u8> input_data(reinterpret_cast<const u8*>(data), data_size);
	std::span<u8> output_data(reinterpret_cast<u8*>(output), output_size);

	if (!cuda_toolkit::beamform(input_data, output_data, bp))
	{
		std::cerr << "Error: Beamforming failed." << std::endl;
	}
}

void
motion_detect_f32_matlab(const float* images, NCCMotionParameters params, float* motion_maps)
{
	size_t data_size = params.image_dims[0] * params.image_dims[1] * params.frame_count * sizeof(float);
	size_t motion_map_size = params.motion_grid_dims[0] * params.motion_grid_dims[1] * sizeof(float) * 4 * params.frame_count;

    std::span<const u8> input_data(reinterpret_cast<const u8*>(images), data_size);
	std::span<u8> output_data(reinterpret_cast<u8*>(motion_maps), motion_map_size);

	if (!cuda_toolkit::motion_detection(input_data, output_data, params))
	{
		std::cerr << "Error: Motion detection failed." << std::endl;
	}
}

void
corr_images_f32_matlab(const float* template_image,
				const float* source_image,
				CorrImagesParameters params,
				float* corr_map)
{
	size_t template_count = params.template_dims[0] * params.template_dims[1];
	size_t source_count = params.source_dims[0] * params.source_dims[1];

	// Valid correlation output dims
	size_t corr_width = params.source_dims[0] - params.template_dims[0] + 1;
	size_t corr_height = params.source_dims[1] - params.template_dims[1] + 1;
	size_t corr_map_count = corr_width * corr_height;

    std::span<const float> template_data(template_image, template_count);
    std::span<const float> source_data(source_image, source_count);
    std::span<float> corr_data(corr_map, corr_map_count);

	if (!cuda_toolkit::corr_images(template_data, source_data, corr_data, params.template_dims, params.source_dims))
	{
		std::cerr << "Error: Correlation failed." << std::endl;
	}
}