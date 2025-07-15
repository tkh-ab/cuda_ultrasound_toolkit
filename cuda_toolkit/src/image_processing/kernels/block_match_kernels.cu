#include "block_match_kernels.cuh"


__host__ static inline void 
show_corr_map(const float* d_corr_map, NppiSize dims)
{
	float* corr_map = new float[dims.width * dims.height];
	cudaMemcpy(corr_map, d_corr_map, dims.width * dims.height * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Correlation Map:" << std::endl;
	for (int y = 0; y < dims.height; ++y)
	{
		for (int x = 0; x < dims.width; ++x)
		{
			std::cout << std::showpos << std::fixed << std::setprecision(3) << corr_map[y * dims.width + x] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	delete[] corr_map;
}


bool
block_match::compare_images(const PitchedArray<float>& template_image,
						const PitchedArray<float>& source_image,
						std::span<int2> motion_map, 
						uint2 image_dims, 
						const NccMotionParameters& params,
						NppStreamContext stream_context)
{

	NppiSize tpl_roi = { (int)params.patch_size, (int)params.patch_size };
	NppiSize src_roi = { tpl_roi.width + (int)params.search_margins[0] * 2, 
							tpl_roi.height + (int)params.search_margins[1] * 2 };

	int image_line_step = (int)image_dims.x * sizeof(float);

	uint2 motion_grid_dims = { image_dims.x / params.motion_grid_spacing, 
							   image_dims.y / params.motion_grid_spacing };

	
	size_t scratch_buffer_size;
	
	NppStatus status = nppiValidNormLevelGetBufferHostSize_32f_C1R_Ctx(src_roi, &scratch_buffer_size, stream_context);

	uint8_t* d_scratch_buffer = nullptr;
	float* d_output_buffer = nullptr;

	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_scratch_buffer, scratch_buffer_size));

	NppiSize valid_corr_dims = { .width = src_roi.width - tpl_roi.width + 1, 
							 	 .height = src_roi.height - tpl_roi.height + 1 };

	size_t valid_corr_size = valid_corr_dims.width * valid_corr_dims.height  * sizeof(float);
	int valid_corr_line_step = valid_corr_dims.width * sizeof(float);

	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_output_buffer, valid_corr_size)); 

	int y_margin = params.search_margins[1];
	int x_margin = params.search_margins[0];
	for( uint i = 0; i < motion_grid_dims.y; i++ ) // Rows
	{
		uint tpl_row_id = i * params.motion_grid_spacing;
		float* target_row_start = template_image.data + tpl_row_id * image_dims.x;

		int src_row_id = tpl_row_id - y_margin;
		int bot_overflow = tpl_row_id + tpl_roi.height + y_margin - image_dims.y;

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

		float* src_row_start = source_image.data + src_row_id * image_dims.x;

		for( uint j = 0; j < motion_grid_dims.x; j++ ) // Columns
		{
			uint tpl_col_id = j * params.motion_grid_spacing;
			float* template_corner = target_row_start + tpl_col_id;

			int src_col_id = tpl_col_id - x_margin;
			int right_overflow = tpl_col_id + tpl_roi.width + x_margin - image_dims.x;

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
													d_output_buffer, valid_corr_line_step, d_scratch_buffer, stream_context);

			//show_corr_map(d_output_buffer, valid_corr_dims);

			uint2 max_index = find_peak(d_output_buffer, valid_corr_dims);
			uint max_offset = max_index.y * valid_corr_dims.width + max_index.x;

			uint2 no_shift_index = {tpl_col_id - src_col_id, tpl_row_id - src_row_id};
			uint no_shift_offset = no_shift_index.y * valid_corr_dims.width + no_shift_index.x;
		
			std::cout << "Motion grid position: X " << j << ", Y " << i << "." << std::endl;
			std::cout << "Template position: X " << tpl_col_id << ", Y " << tpl_row_id << "." << std::endl;
			std::cout << "Source position: X " << src_col_id << ", Y " << src_row_id << "." << std::endl << std::endl;

			std::cout << "No shift index: X " << no_shift_index.x << ", Y " << no_shift_index.y << "." << std::endl;
			std::cout << "No shift value: " << sample_value<float>(d_output_buffer + no_shift_offset) << std::endl << std::endl;
			std::cout << "Peak index: X " << max_index.x << ", Y " << max_index.y << "." << std::endl;
			std::cout << "Peak value: " << sample_value<float>(d_output_buffer + max_offset) << std::endl << std::endl << std::endl;
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