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
						int2* motion_map, 
						uint2 image_dims, 
						const NccMotionParameters& params,
						NppStreamContext stream_context)
{

	int2 search_margins = { (int)params.search_margins[0], (int)params.search_margins[1] };
	NppiSize tpl_roi = { (int)params.patch_size, (int)params.patch_size };
	NppiSize src_roi = { tpl_roi.width + (int)search_margins.x * 2, 
							tpl_roi.height + (int)search_margins.y * 2 };

	
	int template_line_step = template_image.pitch;
	int source_line_step = source_image.pitch;

	int template_row_offset = template_line_step / sizeof(float);
	int source_row_offset = source_line_step / sizeof(float);

	uint2 motion_grid_dims = { params.motion_grid_dims[0], params.motion_grid_dims[1] };

	size_t scratch_buffer_size;
	
	NppStatus status = nppiValidNormLevelGetBufferHostSize_32f_C1R_Ctx(src_roi, &scratch_buffer_size, stream_context);

	uint8_t* d_scratch_buffer = nullptr;
	float* d_output_buffer = nullptr;

	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_scratch_buffer, scratch_buffer_size));

	NppiSize valid_corr_dims = { .width = src_roi.width - tpl_roi.width + 1, 
							 	 .height = src_roi.height - tpl_roi.height + 1 };

	size_t valid_corr_size = valid_corr_dims.width * valid_corr_dims.height  * sizeof(float);

	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_output_buffer, valid_corr_size)); 

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

			valid_corr_dims = { .width = current_src_roi.width - current_tpl_roi.width + 1, 
								.height = current_src_roi.height - current_tpl_roi.height + 1 };
			int corr_line_step = valid_corr_dims.width * sizeof(float);

			
			// Perform the NCC comparison
			status = nppiCrossCorrValid_NormLevel_32f_C1R_Ctx(source_corner, source_line_step, current_src_roi, 
													template_corner, template_line_step, current_tpl_roi, 
													d_output_buffer, corr_line_step, d_scratch_buffer, stream_context);
			if (i == 7 && j == 2)
			{
				std::cout << "Template: " << std::endl;
				show_corr_map(template_corner, current_tpl_roi, template_line_step);
				std::cout << std::endl << "Source: " << std::endl;
				show_corr_map(source_corner, current_src_roi, source_line_step);

				std::cout << std::endl << "NCC comparison output: " << status << std::endl;
				show_corr_map(d_output_buffer, valid_corr_dims, corr_line_step);
				std::cout << std::endl;
			}
			

			int2 no_shift_index = {(uint)tpl_col_id - src_col_id, (uint)tpl_row_id - src_row_id};
			uint no_shift_offset = no_shift_index.y * valid_corr_dims.width + no_shift_index.x;
		
			int2 motion_vector = select_peak(d_output_buffer, valid_corr_dims, params, d_scratch_buffer, stream_context, no_shift_index);

			if (motion_vector.x == INT_MIN || motion_vector.y == INT_MIN)
			{
				std::cerr << "Error selecting peak position." << std::endl;
				cudaFree(d_scratch_buffer);
				cudaFree(d_output_buffer);
				return false;
			}

			int2 peak_pos = ADD_V2(motion_vector, no_shift_index);

			motion_map[i * motion_grid_dims.x + j] = motion_vector;

			// std::cout << "Motion at (" << j << ", " << i << "): "
			// 	<< "Peak Position: (" << peak_pos.x << ", " << peak_pos.y << ") "
			// 	<< "Motion Vector: (" << motion_vector.x << ", " << motion_vector.y << ")" << std::endl;
			// std::cout << std::endl;
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

	int *d_min_peak_x, *d_min_peak_y;
	float *d_min_peak_value;
	cudaMalloc((void**)&d_min_peak_x, sizeof(int));
	cudaMalloc((void**)&d_min_peak_y, sizeof(int));
	cudaMalloc((void**)&d_min_peak_value, sizeof(float));

	int image_line_step = (int)dims.width * sizeof(float);
	NppStatus status = nppiMaxIndx_32f_C1R_Ctx(d_corr_map, image_line_step, dims, d_scratch_buffer, d_peak_value, d_peak_x, d_peak_y, stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '"<< status <<"' during peak detection." << std::endl;
		return { INT_MIN, INT_MIN };
	}

	status = nppiMinIndx_32f_C1R_Ctx(d_corr_map, image_line_step, dims, d_scratch_buffer, d_min_peak_value, d_min_peak_x, d_min_peak_y, stream_context);

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

	int2 min_peak_pos = { sample_value<int>(d_min_peak_x), sample_value<int>(d_min_peak_y) };
	float min_peak_value = sample_value<float>(d_min_peak_value);

	if(abs(min_peak_value) > abs(peak_value))
	{
		peak_value = abs(min_peak_value);
		peak_pos = min_peak_pos;
	}

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

	cudaFree(d_min_peak_x);
	cudaFree(d_min_peak_y);
	cudaFree(d_min_peak_value);
	return peak_pos;
}