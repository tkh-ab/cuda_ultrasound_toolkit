
#include <format>

#include "kernels/block_match_kernels.cuh"

#include "image_processor.h"




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

bool ImageProcessor::ncc_block_match(std::vector<PitchedArray<float>> &d_input_images, 
										int2* motion_maps, 
										const NccMotionParameters& params)
{
	int2 search_margins = { (int)params.search_margins[0], (int)params.search_margins[1] };
	NppiSize tpl_roi = { (int)params.patch_size, (int)params.patch_size };
	NppiSize src_roi = { tpl_roi.width + (int)search_margins.x * 2 + 1, 
							tpl_roi.height + (int)search_margins.y * 2 };

	if (!_create_buffers(src_roi, tpl_roi))
		return false;

	size_t motion_map_count = params.motion_grid_dims[0] * params.motion_grid_dims[1];
	uint2 image_dims = { params.image_dims[0], params.image_dims[1] };
	uint reference_frame = params.reference_frame;

	

	bool result = false;
	for( uint i = 0; i < d_input_images.size(); ++i)
	{
		std::cout << "Processing frame " << i << std::endl;

		auto start = std::chrono::high_resolution_clock::now();
		if( i == reference_frame ) continue;

		int2 *current_map = motion_maps + i * motion_map_count;

		result &= _compare_images( 	d_input_images[reference_frame],
												d_input_images[i],
												current_map,
												image_dims,
												params
											);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
    	std::cout << "Block match duration: " << elapsed.count() << " seconds" << std::endl << std::endl;
	}

	return result;
}


bool
ImageProcessor::_compare_images(const PitchedArray<float>& template_image,
						const PitchedArray<float>& source_image,
						int2* motion_map, 
						uint2 image_dims, 
						const NccMotionParameters& params)
{
	std::chrono::duration<double> corr_duration = std::chrono::duration<double>::zero();
	std::chrono::duration<double> peak_duration = std::chrono::duration<double>::zero();
	int2 search_margins = { (int)params.search_margins[0], (int)params.search_margins[1] };
	NppiSize tpl_roi = { (int)params.patch_size, (int)params.patch_size };
	NppiSize src_roi = { tpl_roi.width + (int)search_margins.x * 2 + 1, 
							tpl_roi.height + (int)search_margins.y * 2 };

	int template_line_step = template_image.pitch;
	int source_line_step = source_image.pitch;

	int template_row_offset = template_line_step / sizeof(float);
	int source_row_offset = source_line_step / sizeof(float);

	uint2 motion_grid_dims = { params.motion_grid_dims[0], params.motion_grid_dims[1] };

	int y_margin = params.search_margins[1];
	int x_margin = params.search_margins[0];
	for( uint i = 0; i < motion_grid_dims.y; i++ ) // Rows
	{
		uint tpl_row_id = i * params.motion_grid_spacing;
		

		int src_row_id = tpl_row_id - y_margin;

		int s_bot_overflow = src_row_id + src_roi.height - image_dims.y;
		int t_bot_overflow = s_bot_overflow - search_margins.y;

		NppiSize row_src_roi = src_roi;
		NppiSize row_tpl_roi = tpl_roi;
		if( src_row_id < 0 )
		{
			// Shrink the ROI and set the new corner
			row_src_roi.height += src_row_id;
			src_row_id = 0;
		}

		if( t_bot_overflow > 0 )
		{
			// Shrink the ROI to avoid overflow
			row_tpl_roi.height -= t_bot_overflow;
		}

		if( s_bot_overflow > 0 )
		{
			// Shrink the ROI to avoid overflow
			row_src_roi.height -= s_bot_overflow;
		}

		float* template_row_start = template_image.data + tpl_row_id * template_row_offset;
		float* src_row_start = source_image.data + src_row_id * source_row_offset;

		for( uint j = 0; j < motion_grid_dims.x; j++ ) // Columns
		{
			uint tpl_col_id = j * params.motion_grid_spacing;
			int src_col_id = tpl_col_id - x_margin;

			int s_right_overflow = src_col_id + src_roi.width - image_dims.x;
			int t_right_overflow = s_right_overflow - search_margins.x;

			NppiSize current_src_roi = row_src_roi;
			NppiSize current_tpl_roi = row_tpl_roi;
			if( src_col_id < 0 )
			{
				// Shrink the ROI and set the new corner
				current_src_roi.width += src_col_id;
				src_col_id = 0;
			}

			if ( t_right_overflow > 0 )
			{
				// Shrink the ROI to avoid overflow
				current_tpl_roi.width -= t_right_overflow;
			}

			if( s_right_overflow > 0 )
			{
				// Shrink the ROI to avoid overflow
				current_src_roi.width -= s_right_overflow;
			}
			
			float* template_corner = template_row_start + tpl_col_id;
			float* source_corner = src_row_start + src_col_id;

			NppiSize valid_corr_dims = { .width = current_src_roi.width - current_tpl_roi.width + 1, 
								.height = current_src_roi.height - current_tpl_roi.height + 1 };
			int corr_line_step = valid_corr_dims.width * sizeof(float);

			
			// Perform the NCC comparison

			auto corr_start = std::chrono::high_resolution_clock::now();
			NppStatus status = nppiCrossCorrValid_NormLevel_32f_C1R_Ctx(source_corner, source_line_step, current_src_roi, 
													template_corner, template_line_step, current_tpl_roi, 
													_d_corr_map, corr_line_step, _d_scratch_buffer, _stream_context);
			cudaDeviceSynchronize();
			auto corr_end = std::chrono::high_resolution_clock::now();
			corr_duration += (corr_end - corr_start);

			int2 no_shift_index = {(uint)tpl_col_id - src_col_id, (uint)tpl_row_id - src_row_id};
			uint no_shift_offset = no_shift_index.y * valid_corr_dims.width + no_shift_index.x;
		
			//show_corr_map(d_output_buffer, valid_corr_dims, corr_line_step);

			auto peak_start = std::chrono::high_resolution_clock::now();
			int2 motion_vector = block_match::select_peak(_d_corr_map, valid_corr_dims, params, _d_scratch_buffer, _stream_context, no_shift_index, corr_line_step);
			cudaDeviceSynchronize();
			auto peak_end = std::chrono::high_resolution_clock::now();
			peak_duration += (peak_end - peak_start);

			if (motion_vector.x == INT_MIN || motion_vector.y == INT_MIN)
			{
				std::cerr << "Error selecting peak position." << std::endl;
				return false;
			}

			motion_map[i * motion_grid_dims.x + j] = motion_vector;
		}
	}
	
	std::cout << "Cross-correlation duration: " << corr_duration.count() << " seconds" << std::endl;
	std::cout << "Peak selection duration: " << peak_duration.count() << " seconds" << std::endl;
	cudaDeviceSynchronize();
	return true;
	
}

bool
ImageProcessor::_create_buffers(NppiSize src_size, NppiSize tpl_size)
{
	_cleanup_buffers();
	NppiSize valid_corr_dims = { .width = src_size.width - tpl_size.width + 1, 
							 	 .height = src_size.height - tpl_size.height + 1 };

	size_t valid_corr_size = valid_corr_dims.width * valid_corr_dims.height  * sizeof(float);

	size_t scratch_buffer_size = 0;
	NppStatus status = nppiValidNormLevelGetBufferHostSize_32f_C1R_Ctx(valid_corr_dims, &scratch_buffer_size, _stream_context);
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Failed to get buffer size for cross-correlation: " << status << std::endl;
		return false;
	}

	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&_d_scratch_buffer, scratch_buffer_size));
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&_d_corr_map, valid_corr_size)); 

	return true;
}


