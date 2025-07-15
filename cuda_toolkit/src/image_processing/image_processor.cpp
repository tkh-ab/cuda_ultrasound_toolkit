
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

bool ImageProcessor::ncc_forward_match(std::span<const u8> input, 
										std::span<u8> motion_maps, 
										const NccMotionParameters& params)
{
	uint2 image_dims = { params.image_dims[0], params.image_dims[1] };
	uint image_count = params.frame_count;

	size_t pixel_count = image_dims.x * image_dims.y;
	size_t image_size = pixel_count * sizeof(float);


	std::vector<PitchedArray<float>> d_input_images;
	d_input_images.reserve(image_count);

	for (uint i = 0; i < image_count; i++)
	{
		float* d_image = nullptr;
		size_t image_pitch = 0;
		CUDA_RETURN_IF_ERROR(cudaMallocPitch((void**)&d_image, &image_pitch, image_dims.x * sizeof(float), image_dims.y));
		CUDA_RETURN_IF_ERROR(cudaMemcpy2D(d_image, image_pitch, 
										   input.data() + i * image_size, 
										   image_dims.x * sizeof(float), 
										   image_dims.x * sizeof(float), 
										   image_dims.y, 
										   cudaMemcpyHostToDevice));
		d_input_images.emplace_back(d_image, image_pitch, uint3(image_dims.x, image_dims.y, 1));
	}
	

	float* d_src_image = nullptr;
	float* d_tpl_image = nullptr;
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_src_image, image_size));
	CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&d_tpl_image, image_size));

	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_src_image, input.data(), image_size, cudaMemcpyHostToDevice));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_tpl_image, input.data(), image_size, cudaMemcpyHostToDevice));


	uint2 motion_grid_dims = { image_dims.x / params.motion_grid_spacing, 
							   image_dims.y / params.motion_grid_spacing };

	size_t motion_map_size = motion_grid_dims.x * motion_grid_dims.y * sizeof(int2);
	size_t motion_map_count = motion_grid_dims.x * motion_grid_dims.y;

	std::span<int2> motion_map_span(reinterpret_cast<int2*>(motion_maps.data()), motion_map_count);

	bool result = block_match::compare_images(
		d_input_images[0],
		d_input_images[1],
		motion_map_span,
		image_dims,
		params,
		_stream_context
	);

	cudaFree(d_src_image);
	cudaFree(d_tpl_image);
	return result;
}


