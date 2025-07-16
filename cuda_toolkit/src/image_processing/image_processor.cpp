
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

bool ImageProcessor::ncc_forward_match(std::vector<PitchedArray<float>> &d_input_images, 
										int2* motion_maps, 
										const NccMotionParameters& params)
{
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

		result &= block_match::compare_images( 	d_input_images[i],
												d_input_images[reference_frame],
												current_map,
												image_dims,
												params,
												_stream_context
											);


		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
    	std::cout << "Block match duration: " << elapsed.count() << " seconds" << std::endl << std::endl;
	}

	

	return result;
}


