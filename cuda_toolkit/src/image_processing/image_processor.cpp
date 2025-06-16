

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

bool 
ImageProcessor::_compare_images(std::span<const float> image_target, 
						std::span<const float> reference,
						std::span<int2> motion_map, 
						uint2 image_dims, 
						const NccCompParameters& params)
{

	NppiSize template_roi = { params.patch_size, params.patch_size };
	NppiSize source_roi = { template_roi.width + params.search_margins[0] * 2, 
							template_roi.height + params.search_margins[1] * 2 };

	size_t image_line_step = image_dims.x * sizeof(float);

	uint2 motion_grid_dims = { image_dims.x / params.motion_grid_spacing, 
							   image_dims.y / params.motion_grid_spacing };
}

}
