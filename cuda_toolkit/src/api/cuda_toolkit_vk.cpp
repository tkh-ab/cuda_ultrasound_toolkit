#include "../rf_processing/rf_processor.h"
#include "../public/cuda_beamformer_parameters.h"
#include "../public/cuda_toolkit_vk.h"

#define MAX_BUFFER_COUNT 16

bool unregister_vk_buffers_();

struct GraphicsSession 
{
    RfProcessor rf_processor;
    std::pair<HANDLE, size_t> ping_pong_memory{nullptr, 0};
	uint ping_pong_count = 0;
	uint ping_pong_buffer_size = 0;
	std::array<void*, MAX_BUFFER_COUNT> mapped_ping_pong_buffers{nullptr};
    bool buffers_init = false;

    ~GraphicsSession()
    {
        unregister_vk_buffers_();
    }
}; 

static GraphicsSession& get_session_()
{
    static GraphicsSession session;
    return session;
}

bool
map_ogl_buffer_(void** d_ptr, cudaGraphicsResource_t ogl_resource)
{
    if (!ogl_resource)
    {
        std::cerr << "OpenGL resource is null." << std::endl;
        return false;
    }

    size_t num_bytes;
    CUDA_RETURN_IF_ERROR(cudaGraphicsMapResources(1, &ogl_resource));
    CUDA_RETURN_IF_ERROR(cudaGraphicsResourceGetMappedPointer(d_ptr, &num_bytes, ogl_resource));

    if (*d_ptr == nullptr)
    {
        std::cerr << "Failed to map OpenGL buffer." << std::endl;
        return false;
    }

    return true;
}

bool
unmap_ogl_buffer_(cudaGraphicsResource_t ogl_resource)
{
    if (ogl_resource)
    {
        CUDA_RETURN_IF_ERROR(cudaGraphicsUnmapResources(1, &ogl_resource));
    }
    return true;
}

bool
unregister_vk_buffers_()
{
    auto& graphics_session = get_session_();
    

    graphics_session.buffers_init = false;
    return true;    
}


bool
init_cuda_configuration(const uint* input_dims, const uint* decoded_dims)
{
	std::cerr << "\n[CUDA]Initializing CUDA configuration with input_dims=[" << input_dims[0] << ", " << input_dims[1] << "] and decoded_dims=[" 
			  << decoded_dims[0] << ", " << decoded_dims[1] << ", " << decoded_dims[2] << "]" << std::endl << std::endl;
    RfProcessor& rf_processor = get_session_().rf_processor;

    if (!rf_processor.init({input_dims[0], input_dims[1]}, {decoded_dims[0], decoded_dims[1], decoded_dims[2]}))
    {
        std::cerr << "Failed to initialize CUDA session." << std::endl;
        return false;
    }

    return true;
}

void
deinit_cuda_configuration()
{
    RfProcessor& rf_processor = get_session_().rf_processor;
    rf_processor.deinit();
    unregister_vk_buffers_();
}

bool
register_cuda_buffers(const uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo)
{
	return true;
}

bool
cuda_hilbert(uint input_buffer_idx, uint output_buffer_idx)
{


	std::cerr << "cuda_hilbert called with input_buffer_idx=" << input_buffer_idx << " and output_buffer_idx=" << output_buffer_idx << std::endl;
    GraphicsSession& graphics_session = get_session_();

    if (!graphics_session.buffers_init)
    {
        std::cerr << "OGL buffers not registered." << std::endl;
        return false;
    }

    RfProcessor& rf_processor = graphics_session.rf_processor;
    

	float* d_input = (float*)graphics_session.mapped_ping_pong_buffers[input_buffer_idx]; 
	cuComplex* d_output = (cuComplex*)graphics_session.mapped_ping_pong_buffers[output_buffer_idx];

    bool result = rf_processor.hilbert_transform_strided(d_input, d_output);

    return result;
}

bool
register_ping_pong_buffers(void* memory_handle, size_t memory_size, uint buffer_count, uint buffer_size)
{
	std::cerr << "register_ping_pong_buffer called with handle=" << memory_handle << ", size=" << memory_size << ", buffer_count=" << buffer_count << ", buffer_size=" << buffer_size << std::endl;

	auto& graphics_session = get_session_();

	cudaExternalMemoryHandleDesc mem_desc = {};
	mem_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	mem_desc.handle.win32.handle = memory_handle;
	mem_desc.size = memory_size;

	cudaExternalMemory_t cuda_ext_memory = nullptr;
	cudaError_t err = cudaImportExternalMemory(&cuda_ext_memory, &mem_desc);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to import external memory: " << cudaGetErrorString(err) << std::endl;
		return false;
	}

	graphics_session.ping_pong_memory = { memory_handle, memory_size };
	graphics_session.ping_pong_count = buffer_count;
	graphics_session.ping_pong_buffer_size = buffer_size;

	cudaExternalMemoryBufferDesc buffer_desc = {};
	buffer_desc.size = buffer_size;
	buffer_desc.flags = 0;

	for (uint i = 0; i < buffer_count; i++)
	{
		buffer_desc.offset = i * buffer_size;
		void* cuda_ptr = nullptr;
		err = cudaExternalMemoryGetMappedBuffer( &cuda_ptr, cuda_ext_memory, &buffer_desc);

		if (err != cudaSuccess)
		{
			std::cerr << "Failed to get mapped buffer for ping-pong buffer " << i << ": " << cudaGetErrorString(err) << std::endl;
			unregister_vk_buffers_();
			return false;
		}
		graphics_session.mapped_ping_pong_buffers[i] = cuda_ptr;
	}

	graphics_session.buffers_init = true;

	return true;
}

