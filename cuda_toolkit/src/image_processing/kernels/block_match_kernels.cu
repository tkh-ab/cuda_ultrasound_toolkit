#include "block_match_kernels.cuh"


__host__ static inline void 
show_corr_map(const float* d_corr_map, NppiSize dims, int line_step = 0)
{
	int row_offset;
	if (line_step == 0) {row_offset = dims.width;}
	else 				{row_offset = line_step / sizeof(float);}

	float* corr_map = new float[dims.width];
	
	for (int y = 0; y < dims.height; ++y)
	{
		cudaMemcpy(corr_map, d_corr_map + y * row_offset, dims.width * sizeof(float), cudaMemcpyDeviceToHost);
		for (int x = 0; x < dims.width; ++x)
		{
			std::cout << std::showpos << std::scientific << std::setprecision(2) << corr_map[x] << ' ';
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

	
	int template_line_step = template_image.pitch;
	int source_line_step = source_image.pitch;

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
			status = nppiCrossCorrValid_NormLevel_32f_C1R_Ctx(source_corner, source_line_step, current_src_roi, 
													template_corner, template_line_step, tpl_roi, 
													d_output_buffer, valid_corr_line_step, d_scratch_buffer, stream_context);
			// if (i == 0 && j == 0)
			// {
			// 	std::cout << "Template: " << std::endl;
			// 	show_corr_map(template_corner, tpl_roi, template_line_step);
			// 	std::cout << std::endl << "Source: " << std::endl;
			// 	show_corr_map(source_corner, current_src_roi, source_line_step);

			// 	std::cout << std::endl << "NCC comparison output: " << status << std::endl;
			// 	show_corr_map(d_output_buffer, valid_corr_dims);
			// 	std::cout << std::endl;
			// }
			

			int2 no_shift_index = {(uint)tpl_col_id - src_col_id, (uint)tpl_row_id - src_row_id};
			uint no_shift_offset = no_shift_index.y * valid_corr_dims.width + no_shift_index.x;
		
			int2 peak_pos = select_peak(d_output_buffer, valid_corr_dims, params, d_scratch_buffer, stream_context, no_shift_index);

			int2 other_peak = find_peak(d_output_buffer, valid_corr_dims);

			other_peak = SUB_V2(other_peak, no_shift_index);

			if (peak_pos.x == INT_MIN || peak_pos.y == INT_MIN)
			{
				std::cerr << "Error selecting peak position." << std::endl;
				cudaFree(d_scratch_buffer);
				cudaFree(d_output_buffer);
				return false;
			}

			motion_map[i * motion_grid_dims.x + j] = peak_pos;

			std::cout << "Motion at (" << j << ", " << i << "): "
				<< "Peak Position: (" << peak_pos.x << ", " << peak_pos.y << ") " << std::endl;

			// std::cout << "Other Peak Position: (" << other_peak.x << ", " << other_peak.y << ") " << std::endl;

			std::cout << std::endl;
		}
	}
	
	cudaFree(d_scratch_buffer);
	cudaFree(d_output_buffer);
	return true;
	
}

int2
block_match::select_peak(const float* d_corr_map, NppiSize dims, const NccMotionParameters& params, Npp8u* d_scratch_buffer, NppStreamContext stream_context, int2 no_shift_pos)
{
	int *d_peak_x, *d_peak_y;
	float *d_peak_value;
	double *d_corr_mean, *d_corr_variance;
	cudaMalloc((void**)&d_peak_x, sizeof(int));
	cudaMalloc((void**)&d_peak_y, sizeof(int));
	cudaMalloc((void**)&d_peak_value, sizeof(float));
	cudaMalloc((void**)&d_corr_mean, sizeof(double));
	cudaMalloc((void**)&d_corr_variance, sizeof(double));

	int image_line_step = (int)dims.width * sizeof(float);
	NppStatus status = nppiMaxIndx_32f_C1R_Ctx(d_corr_map, image_line_step, dims, d_scratch_buffer, d_peak_value, d_peak_x, d_peak_y, stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '"<< status <<"' during peak detection." << std::endl;
		return { INT_MIN, INT_MIN };
	}

	status = nppiMean_StdDev_32f_C1R_Ctx(d_corr_map, image_line_step, dims, d_scratch_buffer, d_corr_mean, d_corr_variance, stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '"<< status <<"' during mean/stddev calculation." << std::endl;
		return { INT_MIN, INT_MIN };
	}

	float no_shift_value = sample_value<float>(d_corr_map + no_shift_pos.y * dims.width + no_shift_pos.x);

	int2 peak_pos = { sample_value<int>(d_peak_x), sample_value<int>(d_peak_y) };
	float peak_value = sample_value<float>(d_peak_value);

	double corr_mean = sample_value<double>(d_corr_mean);
	double corr_variance = sample_value<double>(d_corr_variance);
	corr_variance *= corr_variance; // Convert stddev to variance

	// Log the peak and no shift values, positions and variance
	// std::cout << "Peak Value: " << peak_value << " at (" << peak_pos.x << ", " << peak_pos.y << ")" << std::endl;
	// std::cout << "No Shift Value: " << no_shift_value << " at (" << no_shift_pos.x << ", " << no_shift_pos.y << ")" << std::endl;
	// std::cout << "Correlation Mean: " << corr_mean <<  " Correlation Variance: " << corr_variance << std::endl;

	// If the peak isn't much higher than the no-shift value, we reject motion
	if (peak_value < no_shift_value * params.correlation_threshold)
	{
		peak_pos = { 0, 0 };
		return peak_pos;
	}

	// If the correlation map is too uniform we can't trust the peak
	if( corr_variance < params.min_patch_variance )
	{
		std::cerr << "Patch variance too low: " << corr_variance << std::endl;
		return { 0, 0 };
	}
	peak_pos = SUB_V2(peak_pos, no_shift_pos);

	cudaFree(d_peak_x);
	cudaFree(d_peak_y);
	cudaFree(d_peak_value);
	cudaFree(d_corr_mean);
	cudaFree(d_corr_variance);
	return peak_pos;
}