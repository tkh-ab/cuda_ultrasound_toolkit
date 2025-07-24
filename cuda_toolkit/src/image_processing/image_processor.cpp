
#include <format>
#include <chrono>
#include <algorithm>

#include "kernels/block_match.h"

#include "image_processor.h"


static inline void 
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
		std::cout << "Processing frame " << i + 1<< std::endl;

		auto start = std::chrono::high_resolution_clock::now();
		if( i == reference_frame ) continue;

		int2 *current_map = motion_maps + i * motion_map_count;

		// PitchedArray<float>* template_image = d_input_images.data() + reference_frame;
		// PitchedArray<float>* source_image = d_input_images.data() + i;

		PitchedArray<float>* template_image = d_input_images.data() + i;
		PitchedArray<float>* source_image = d_input_images.data() + reference_frame;

		result &= _compare_images( *template_image, *source_image, current_map, image_dims, params);

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
	int patch_size = params.patch_size;

	int template_line_step = template_image.pitch;
	int source_line_step = source_image.pitch;

	int template_pitch = template_line_step / sizeof(float);
	int source_pitch = source_line_step / sizeof(float);

	uint2 motion_grid_dims = { params.motion_grid_dims[0], params.motion_grid_dims[1] };
	int grid_spacing = params.motion_grid_spacing;

	uint2 template_center = { patch_size / 2, patch_size / 2 };

	int count = 0;
	for( uint i = 0; i < motion_grid_dims.y; i++ ) // Rows
	{
		int tpl_center_y = i * grid_spacing;

		int tpl_top_y = tpl_center_y - template_center.y;
		int tpl_bottom_y = tpl_top_y + patch_size - 1;

		int src_top_y = tpl_top_y - search_margins.y;
		int src_bottom_y = tpl_bottom_y + search_margins.y;

		tpl_top_y = max(tpl_top_y, 0);
		tpl_bottom_y = min(tpl_bottom_y, image_dims.y - 1);
		src_top_y = max(src_top_y, 0);
		src_bottom_y = min(src_bottom_y, image_dims.y - 1);

		float* template_row_start = template_image.get_row(tpl_top_y);
		float* src_row_start = source_image.get_row(src_top_y);

		for( uint j = 0; j < motion_grid_dims.x; j++ ) // Columns
		{

			int tpl_center_x = j * grid_spacing;
			int tpl_left_x = tpl_center_x - template_center.x;
			int tpl_right_x = tpl_left_x + patch_size - 1;

			// +1 Ensures the valid correlation output has even width for best NCC performance.
			// There's still a performance hit on the edges but its an improvement.
			int src_left_x = tpl_left_x - search_margins.x; 
			int src_right_x = tpl_right_x + search_margins.x + 1;

			tpl_left_x = max(tpl_left_x, 0);
			tpl_right_x = min(tpl_right_x, image_dims.x - 1);
			src_left_x = max(src_left_x, 0);
			src_right_x = min(src_right_x, image_dims.x - 1);

			NppiSize tpl_roi = { tpl_right_x - tpl_left_x + 1, 
								 tpl_bottom_y - tpl_top_y + 1 };
			NppiSize src_roi = { src_right_x - src_left_x + 1, 
								 src_bottom_y - src_top_y + 1 };

			float* template_corner = template_row_start + tpl_left_x;
			float* source_corner = src_row_start + src_left_x;

			NppiSize valid_corr_dims = { .width = src_roi.width - tpl_roi.width + 1, 
										.height = src_roi.height - tpl_roi.height + 1 };

			int corr_line_step = valid_corr_dims.width * sizeof(float);

			
			// Perform the NCC comparison

			auto corr_start = std::chrono::high_resolution_clock::now();
			volatile NppStatus status = nppiCrossCorrValid_NormLevel_32f_C1R_Ctx(source_corner, source_line_step, src_roi, 
													template_corner, template_line_step, tpl_roi, 
													_d_corr_map, corr_line_step, _d_scratch_buffer, _stream_context);
			cudaDeviceSynchronize();
			if (status != NPP_SUCCESS)
			{
				std::cerr << "Cross-correlation failed with status: " << status << std::endl;
				return false;
			}

			auto corr_end = std::chrono::high_resolution_clock::now();
			corr_duration += (corr_end - corr_start);

			// Which value in the correlation map represents no motion.
			int2 no_shift_index = {tpl_left_x - src_left_x, tpl_top_y - src_top_y};
			uint no_shift_offset = no_shift_index.y * valid_corr_dims.width + no_shift_index.x;
	

			auto peak_start = std::chrono::high_resolution_clock::now();
			//int2 motion_vector = block_match::select_peak(_d_corr_map, valid_corr_dims, params, _d_scratch_buffer, _stream_context, no_shift_index, corr_line_step);

			u8* scratch_buffer;
			cudaMalloc((void**)&scratch_buffer, _scratch_buffer_size);
			int2 motion_vector = block_match::find_peaks(_d_corr_map, valid_corr_dims, params, corr_line_step, no_shift_index, scratch_buffer);
			cudaDeviceSynchronize();

			cudaFree(scratch_buffer);

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

	scratch_buffer_size = scratch_buffer_size < Min_Scratch_Buffer_Size ? Min_Scratch_Buffer_Size : scratch_buffer_size;

	std::cout << "Scratch buffer size: " << scratch_buffer_size << " bytes" << std::endl;
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&_d_scratch_buffer, scratch_buffer_size));
	_scratch_buffer_size = scratch_buffer_size;
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&_d_corr_map, valid_corr_size));

	return true;
}


