
#include <format>
#include <chrono>
#include <algorithm>

#include "kernels/block_match.h"

#include "image_processor.h"

bool ImageProcessor::ncc_block_match(std::vector<PitchedArray<float>> &d_input_images, 
										float4* motion_maps, 
										const NccMotionParameters& params)
{
	constexpr uint stream_count = 8; // Number of streams to use for processing

	size_t motion_map_count = params.motion_grid_dims[0] * params.motion_grid_dims[1];
	uint2 image_dims = { params.image_dims[0], params.image_dims[1] };
	
	float4* d_motion_map;
	size_t motion_map_size = motion_map_count * sizeof(float4) * d_input_images.size();
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_motion_map, motion_map_size));
	CUDA_RETURN_IF_ERROR(cudaMemset(d_motion_map, 0, motion_map_size));

	if (!_create_pipeline_ctxs(params, stream_count)) return false;

	bool result = false;
	for( uint i = 0; i < d_input_images.size(); ++i)
	{
		std::cout << "Processing frame " << i + 1<< std::endl;

		auto start = std::chrono::high_resolution_clock::now();
		uint reference_frame = params.reference_frame;
		if( i == reference_frame ) continue;

		if (params.neighbour_compare)
		{
			if (reference_frame > i) reference_frame = i + 1;
			else reference_frame = i - 1;
		}

		PitchedArray<float>* template_image = d_input_images.data() + reference_frame;
		PitchedArray<float>* source_image = d_input_images.data() + i;

		result &= _compare_images( *template_image, *source_image, d_motion_map + i * motion_map_count, image_dims, params);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
    	std::cout << "Block match duration: " << elapsed.count() << " seconds" << std::endl << std::endl;
	}

	// Copy the motion maps to the output
	CUDA_RETURN_IF_ERROR(cudaMemcpy(motion_maps, d_motion_map, motion_map_size, cudaMemcpyDeviceToHost));
	cudaFree(d_motion_map);

	return result;
}

bool
ImageProcessor::corr_images(const PitchedArray<float>& d_template_image,
							const PitchedArray<float>& d_source_image,
							float* corr_map,
							uint2 template_dims,
							uint2 source_dims)
{
	if (template_dims.x > source_dims.x || template_dims.y > source_dims.y)
	{
		std::cerr << "Template dimensions must be <= source dimensions." << std::endl;
		return false;
	}

	NppiSize tpl_roi = { (int)template_dims.x, (int)template_dims.y };
	NppiSize src_roi = { (int)source_dims.x, (int)source_dims.y };
	NppiSize valid_corr_dims = {
		.width = src_roi.width - tpl_roi.width + 1,
		.height = src_roi.height - tpl_roi.height + 1
	};

	int tpl_line_step = (int)d_template_image.pitch;
	int src_line_step = (int)d_source_image.pitch;
	int corr_line_step = valid_corr_dims.width * (int)sizeof(float);

	size_t corr_bytes = (size_t)valid_corr_dims.width * valid_corr_dims.height * sizeof(float);
	size_t scratch_buffer_size = 0;
	NPP_RETURN_IF_ERR(nppiValidNormLevelGetBufferHostSize_32f_C1R_Ctx(
		valid_corr_dims, &scratch_buffer_size, _default_stream_context));
	scratch_buffer_size = scratch_buffer_size < Min_Scratch_Buffer_Size ? Min_Scratch_Buffer_Size : scratch_buffer_size;

	float* d_corr_map = nullptr;
	u8* d_scratch_buffer = nullptr;
	cudaError_t err = cudaMalloc((void**)&d_corr_map, corr_bytes);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate correlation map buffer: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	err = cudaMalloc((void**)&d_scratch_buffer, scratch_buffer_size);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate NPP scratch buffer: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_corr_map);
		return false;
	}

	NppStatus status = nppiCrossCorrValid_NormLevel_32f_C1R_Ctx(
		d_source_image.data, src_line_step, src_roi,
		d_template_image.data, tpl_line_step, tpl_roi,
		d_corr_map, corr_line_step,
		d_scratch_buffer, _default_stream_context);

	if (status != NPP_SUCCESS)
	{
		std::cerr << "NPP error '" << status << "' during full-image cross-correlation." << std::endl;
		cudaFree(d_scratch_buffer);
		cudaFree(d_corr_map);
		return false;
	}

	err = cudaMemcpy(corr_map, d_corr_map, corr_bytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy correlation map to host: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_scratch_buffer);
		cudaFree(d_corr_map);
		return false;
	}

	cudaFree(d_scratch_buffer);
	cudaFree(d_corr_map);
	return true;
}

bool
ImageProcessor::_compare_images(const PitchedArray<float>& template_image,
						const PitchedArray<float>& source_image,
						float4* d_motion_map, 
						uint2 image_dims, 
						const NccMotionParameters& params)
{
	std::chrono::duration<double> corr_duration = std::chrono::duration<double>::zero();
	int2 search_margins = { (int)params.search_margins[0], (int)params.search_margins[1] };

	uint patch_size = params.patch_size;

	int template_line_step = (int)template_image.pitch;
	int source_line_step = (int)source_image.pitch;

	uint2 motion_grid_dims = { params.motion_grid_dims[0], params.motion_grid_dims[1] };
	int grid_spacing = params.motion_grid_spacing;

	uint2 template_center = { patch_size / 2, patch_size / 2 };

	uint stream_index = 0;
	for( uint i = 0; i < motion_grid_dims.y; i++ ) // Rows
	{
		int tpl_center_y = i * grid_spacing;

		int tpl_top_y = tpl_center_y - template_center.y;
		int tpl_bottom_y = tpl_top_y + patch_size - 1;

		int src_top_y = tpl_top_y - search_margins.y;
		int src_bottom_y = tpl_bottom_y + search_margins.y;

		tpl_top_y = max(tpl_top_y, 0);
		tpl_bottom_y = min(tpl_bottom_y, (int)image_dims.y - 1);
		src_top_y = max(src_top_y, 0);
		src_bottom_y = min(src_bottom_y, (int)image_dims.y - 1);

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
			tpl_right_x = min(tpl_right_x, (int)image_dims.x - 1);
			src_left_x = max(src_left_x, 0);
			src_right_x = min(src_right_x, (int)image_dims.x - 1);

			NppiSize tpl_roi = { tpl_right_x - tpl_left_x + 1, 
								 tpl_bottom_y - tpl_top_y + 1 };
			NppiSize src_roi = { src_right_x - src_left_x + 1, 
								 src_bottom_y - src_top_y + 1 };

			float* template_corner = template_row_start + tpl_left_x;
			float* source_corner = src_row_start + src_left_x;

			NppiSize valid_corr_dims = { .width = src_roi.width - tpl_roi.width + 1, 
										.height = src_roi.height - tpl_roi.height + 1 };

			//int corr_line_step = valid_corr_dims.width * sizeof(float);
			int2 no_shift_index = {tpl_left_x - src_left_x, tpl_top_y - src_top_y};

			float4* d_motion_point = d_motion_map + i * motion_grid_dims.x + j;

			block_match::PipelineCtx& ctx = _pipeline_contexts[stream_index % _pipeline_contexts.size()];
			uint2 vector_id = { j, i };
			block_match::block_match_pipeline(source_corner, template_corner, d_motion_point,
									src_roi, tpl_roi, source_line_step, template_line_step, ctx,
									no_shift_index, params,vector_id);
			

		}
	}

	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error during synchronization: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error after block match pipeline: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	
	std::cout << "Cross-correlation duration: " << corr_duration.count() << " seconds" << std::endl;

	return true;
	
}

bool
ImageProcessor::_create_pipeline_ctxs(const NccMotionParameters& params, uint stream_count)
{
	_clear_pipeline_contexts();

	int2 search_margins = { (int)params.search_margins[0], (int)params.search_margins[1] };
	NppiSize tpl_roi = { (int)params.patch_size, (int)params.patch_size };
	NppiSize src_roi = { tpl_roi.width + (int)search_margins.x * 2 + 1, 
							tpl_roi.height + (int)search_margins.y * 2 };
	NppiSize valid_corr_dims = { .width = src_roi.width - tpl_roi.width + 1, 
							 	 .height = src_roi.height - tpl_roi.height + 1 };

	size_t valid_corr_size = valid_corr_dims.width * valid_corr_dims.height  * sizeof(float);
	size_t scratch_buffer_size = 0;
	NppStatus status = nppiValidNormLevelGetBufferHostSize_32f_C1R_Ctx(valid_corr_dims, &scratch_buffer_size, _default_stream_context);
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Failed to get buffer size for cross-correlation: " << status << std::endl;
		return false;
	}

	scratch_buffer_size = scratch_buffer_size < Min_Scratch_Buffer_Size ? Min_Scratch_Buffer_Size : scratch_buffer_size;

	_pipeline_contexts.resize(stream_count);
	for (auto& ctx : _pipeline_contexts)
	{
		CUDA_RETURN_IF_ERROR(cudaStreamCreate(&ctx.stream));
		ctx.stream_context = _create_stream_context(ctx.stream);
		ctx.scratch_buffer_size = scratch_buffer_size;
		ctx.corr_map_size = valid_corr_size;

		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&ctx.d_scratch_buffer, scratch_buffer_size));
		CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&ctx.d_corr_map, valid_corr_size));
	}


	return true;
}

NppStreamContext 
ImageProcessor::_create_stream_context(cudaStream_t stream) 
{
    NppStreamContext ctx = {};

    int device = -1;
    cudaGetDevice(&device);  // this always returns the device active in this thread

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    ctx.hStream = stream;

    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = props.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;

    ctx.nSharedMemPerBlock = props.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = props.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = props.minor;

    return ctx;    
}
