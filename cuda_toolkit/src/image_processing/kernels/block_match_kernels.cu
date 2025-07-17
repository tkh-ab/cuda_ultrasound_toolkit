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


int2
block_match::select_peak(const float* d_corr_map, NppiSize dims, const NccMotionParameters& params, Npp8u* d_scratch_buffer, NppStreamContext stream_context, int2 no_shift_pos)
{

	struct Stats
	{
		int2 peak_pos;
		float peak_value;
		int2 min_peak_pos;
		float min_peak_value;
		double corr_mean;
		double corr_std;
	};

	// The scratch buffer is sized for the NCC so we have extra space.
	u8 *scratch_stack = d_scratch_buffer;
	Stats *d_stats = (Stats *)scratch_stack;
	scratch_stack += sizeof(Stats);

	int image_line_step = (int)dims.width * sizeof(float);
	NppStatus status = nppiMaxIndx_32f_C1R_Ctx(d_corr_map, image_line_step, dims, scratch_stack, &d_stats->peak_value, &d_stats->peak_pos.x, &d_stats->peak_pos.y, stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '"<< status <<"' during peak detection." << std::endl;
		return { INT_MIN, INT_MIN };
	}

	status = nppiMinIndx_32f_C1R_Ctx(d_corr_map, image_line_step, dims, scratch_stack, &d_stats->min_peak_value, &d_stats->min_peak_pos.x, &d_stats->min_peak_pos.y, stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '"<< status <<"' during peak detection." << std::endl;
		return { INT_MIN, INT_MIN };
	}

	// status = nppiMean_StdDev_32f_C1R_Ctx(d_corr_map, image_line_step, dims, scratch_stack, &d_stats->corr_mean, &d_stats->corr_std, stream_context);

	// if (status != NPP_SUCCESS)
	// {
	// 	std::cerr << "NPP error '"<< status <<"' during mean/stddev calculation." << std::endl;
	// 	return { INT_MIN, INT_MIN };
	// }

	Stats c_stats = sample_value<Stats>(d_stats);
	float no_shift_value = sample_value<float>(d_corr_map + no_shift_pos.y * dims.width + no_shift_pos.x);

	int2 true_peak_pos = c_stats.peak_pos;
	float true_peak_value = c_stats.peak_value;

	// if(abs(c_stats.min_peak_value) > abs(c_stats.peak_value))
	// {
	// 	true_peak_value = abs(c_stats.min_peak_value);
	// 	true_peak_pos = c_stats.min_peak_pos;
	// }

	double corr_variance = c_stats.corr_std * c_stats.corr_std;

	// If the peak isn't much higher than the no-shift value, we reject motion
	if (true_peak_value < (no_shift_value * params.correlation_threshold))
	{
		true_peak_pos = { 0, 0 };
		return true_peak_pos;
	}

	// If the correlation map is too uniform we can't trust the peak
	if( corr_variance < params.min_patch_variance )
	{
		return { 0, 0 };
	}
	true_peak_pos = SUB_V2(true_peak_pos, no_shift_pos);
	return true_peak_pos;
}