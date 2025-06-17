

#include "image_processor.h"


namespace image_processing {



NppStreamContext 
ImageProcessor::_create_stream_context() 
{
    NppStreamContext ctx = {};

    int device = -1;
    cudaGetDevice(&device);  // this always returns the device active in this thread

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    ctx.hStream = 0;  // Default stream, can be set to a specific stream if needed

    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = props.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;

    ctx.nSharedMemPerBlock = props.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = props.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = props.minor;

    return ctx;    
}

bool ImageProcessor::ncc_forward_match(std::span<const u8> input, 
										std::span<u8> motion_maps, 
										uint3 image_dims, 
										const NccCompParameters& params)
{

	uint2 single_image_dims = { image_dims.x, image_dims.y };
	uint image_count = image_dims.z;

	size_t image_size = single_image_dims.x * single_image_dims.y * sizeof(float);

	float* d_src_image = nullptr;
	float* d_tpl_image = nullptr;
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_src_image, image_size));
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_tpl_image, image_size));

	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_src_image, input.data(), image_size, cudaMemcpyHostToDevice));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_tpl_image, input.data() + image_size, image_size, cudaMemcpyHostToDevice));


	uint2 motion_grid_dims = { single_image_dims.x / params.motion_grid_spacing, 
							   single_image_dims.y / params.motion_grid_spacing };

	size_t motion_map_size = motion_grid_dims.x * motion_grid_dims.y * sizeof(int2);
	size_t motion_map_count = motion_grid_dims.x * motion_grid_dims.y;

	std::span<int2> motion_map_span(reinterpret_cast<int2*>(motion_maps.data()), motion_map_count);

	bool result = _compare_images(
		std::span<const float>(d_tpl_image, single_image_dims.x * single_image_dims.y),
		std::span<const float>(d_src_image, single_image_dims.x * single_image_dims.y),
		motion_map_span,
		single_image_dims,
		params
	);

	cudaFree(d_src_image);
	cudaFree(d_tpl_image);

	return result;
}

bool 
ImageProcessor::_compare_images(std::span<const float> template_image, 
						std::span<const float> source_image,
						std::span<int2> motion_map, 
						uint2 image_dims, 
						const NccCompParameters& params)
{

	NppiSize tpl_roi = { (int)params.patch_size, (int)params.patch_size };
	NppiSize src_roi = { tpl_roi.width + (int)params.search_margins[0] * 2, 
							tpl_roi.height + (int)params.search_margins[1] * 2 };

	int image_line_step = (int)image_dims.x * sizeof(float);

	uint2 motion_grid_dims = { image_dims.x / params.motion_grid_spacing, 
							   image_dims.y / params.motion_grid_spacing };

	
	size_t scratch_buffer_size;
	
	NppStatus status = nppiValidNormLevelGetBufferHostSize_32f_C1R_Ctx(src_roi, &scratch_buffer_size, _stream_context);

	uint8_t* d_scratch_buffer = nullptr;
	float* d_output_buffer = nullptr;

	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_scratch_buffer, scratch_buffer_size));

	size_t corr_out_size = tpl_roi.width * tpl_roi.height * sizeof(float);
	int out_line_step = tpl_roi.width * sizeof(float);
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_output_buffer, corr_out_size)); 

	int y_margin = params.search_margins[1];
	int x_margin = params.search_margins[0];
	for( uint i = 0; i < motion_grid_dims.y; i++ ) // Rows
	{
		uint row_id = i * params.motion_grid_spacing;
		float* target_row_start = (float*)template_image.data() + row_id * image_dims.x;

		int src_row_id = row_id - y_margin;
		int bot_overflow = row_id + tpl_roi.height + y_margin - image_dims.y;

		NppiSize row_src_roi = src_roi;
		if( src_row_id < 0 )
		{
			// Shrink the ROI and set the new corner
			row_src_roi.height += src_row_id;
			src_row_id = 0;
		}

		if( bot_overflow > 0 )
		{
			// Shrink the ROI to avoid overflow
			row_src_roi.height -= bot_overflow;
			bot_overflow = 0;
		}

		float* src_row_start = (float*)source_image.data() + src_row_id * image_dims.x;
		
		for( uint j = 0; j < motion_grid_dims.x; j++ ) // Columns
		{
			uint col_id = j * params.motion_grid_spacing;
			float* template_corner = target_row_start + col_id;

			int src_col_id = col_id - x_margin;
			int right_overflow = col_id + tpl_roi.width + x_margin - image_dims.x;

			NppiSize current_src_roi = row_src_roi;
			if( src_col_id < 0 )
			{
				// Shrink the ROI and set the new corner
				current_src_roi.width += src_col_id;
				src_col_id = 0;
			}
			if( right_overflow > 0 )
			{
				// Shrink the ROI to avoid overflow
				current_src_roi.width -= right_overflow;
				right_overflow = 0;
			}
			float* source_corner = src_row_start + src_col_id;


			// Perform the NCC comparison
			status = nppiCrossCorrValid_NormLevel_32f_C1R_Ctx(source_corner, image_line_step, current_src_roi, 
													template_corner, image_line_step, tpl_roi, 
													d_output_buffer, out_line_step, d_scratch_buffer, _stream_context);
			if (status != NPP_SUCCESS)
			{
				std::cerr << "NPP error during NCC comparison: " << status << std::endl;
				cudaFree(d_scratch_buffer);
				cudaFree(d_output_buffer);
				return false;
			}

		}
	}
	
	cudaFree(d_scratch_buffer);
	cudaFree(d_output_buffer);
	return true;
	
}

}
