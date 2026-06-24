#include "../rf_processing/rf_processor.h"
#include "../public/cuda_beamformer_parameters.h"
#include "../public/cuda_toolkit_vk.h"

#define MAX_BUFFER_COUNT 16

bool unregister_vk_buffers_();
bool unregister_vk_semaphores_();

struct GraphicsSession 
{
    RfProcessor rf_processor;
    std::pair<HANDLE, size_t> ping_pong_memory{nullptr, 0};
    cudaExternalMemory_t ping_pong_ext_memory = nullptr;
	uint ping_pong_count = 0;
	uint ping_pong_buffer_size = 0;
	std::array<void*, MAX_BUFFER_COUNT> mapped_ping_pong_buffers{nullptr};
    bool buffers_init = false;
    cudaExternalSemaphore_t vulkan_signal_semaphore = nullptr;
    cudaExternalSemaphore_t cuda_signal_semaphore = nullptr;
    bool semaphores_init = false;

    ~GraphicsSession()
    {
        unregister_vk_buffers_();
        unregister_vk_semaphores_();
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

    for (void* ptr : graphics_session.mapped_ping_pong_buffers)
    {
        if (ptr)
        {
            CUDA_RETURN_IF_ERROR(cudaFree(ptr));
        }
    }

    if (graphics_session.ping_pong_ext_memory)
    {
        CUDA_RETURN_IF_ERROR(cudaDestroyExternalMemory(graphics_session.ping_pong_ext_memory));
        graphics_session.ping_pong_ext_memory = nullptr;
    }

    graphics_session.buffers_init = false;
    graphics_session.ping_pong_memory = { nullptr, 0 };
    graphics_session.ping_pong_count = 0;
    graphics_session.ping_pong_buffer_size = 0;
    graphics_session.mapped_ping_pong_buffers.fill(nullptr);

    return true;    
}

bool
unregister_vk_semaphores_()
{
    auto& graphics_session = get_session_();

    if (graphics_session.vulkan_signal_semaphore)
    {
        CUDA_RETURN_IF_ERROR(cudaDestroyExternalSemaphore(graphics_session.vulkan_signal_semaphore));
        graphics_session.vulkan_signal_semaphore = nullptr;
    }

    if (graphics_session.cuda_signal_semaphore)
    {
        CUDA_RETURN_IF_ERROR(cudaDestroyExternalSemaphore(graphics_session.cuda_signal_semaphore));
        graphics_session.cuda_signal_semaphore = nullptr;
    }

    graphics_session.semaphores_init = false;
    return true;
}


bool
init_cuda_configuration(const uint* input_dims, const uint* decoded_dims, uint chunk_channel_count)
{
	std::cerr << "\n[CUDA]Initializing CUDA configuration with input_dims=[" << input_dims[0] << ", " << input_dims[1] << "] and decoded_dims=[" 
			  << decoded_dims[0] << ", " << decoded_dims[1] << ", " << decoded_dims[2] << "]" << std::endl << std::endl;
    RfProcessor& rf_processor = get_session_().rf_processor;

    if (!rf_processor.init({input_dims[0], input_dims[1]}, {decoded_dims[0], chunk_channel_count, decoded_dims[2]}))
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
    unregister_vk_semaphores_();
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

    if (input_buffer_idx >= graphics_session.ping_pong_count ||
        output_buffer_idx >= graphics_session.ping_pong_count)
    {
        std::cerr << "Invalid ping-pong buffer index." << std::endl;
        return false;
    }

    RfProcessor& rf_processor = graphics_session.rf_processor;
    

	float* d_input = (float*)graphics_session.mapped_ping_pong_buffers[input_buffer_idx]; 
	cuComplex* d_output = (cuComplex*)graphics_session.mapped_ping_pong_buffers[output_buffer_idx];

    if (graphics_session.semaphores_init)
    {
        cudaExternalSemaphoreWaitParams wait_params = {};
        CUDA_RETURN_IF_ERROR(cudaWaitExternalSemaphoresAsync(&graphics_session.vulkan_signal_semaphore, &wait_params, 1, 0));
    }



    bool result = rf_processor.hilbert_transform_strided(d_input, d_output);

	cuComplex input_sample;
	cuComplex output_sample;
	CUDA_RETURN_IF_ERROR(cudaMemcpy(&input_sample, d_input, sizeof(cuComplex), cudaMemcpyDeviceToHost));
	CUDA_RETURN_IF_ERROR(cudaMemcpy(&output_sample, d_output, sizeof(cuComplex), cudaMemcpyDeviceToHost));
	std::cerr << "Input sample: (" << input_sample.x << ", " << input_sample.y << "), Output sample: (" << output_sample.x << ", " << output_sample.y << ")" << std::endl;

    if (result && graphics_session.semaphores_init)
    {
        cudaExternalSemaphoreSignalParams signal_params = {};
        CUDA_RETURN_IF_ERROR(cudaSignalExternalSemaphoresAsync(&graphics_session.cuda_signal_semaphore, &signal_params, 1, 0));
    }

    return result;
}

bool
register_ping_pong_buffers(void* memory_handle, size_t memory_size, uint buffer_count, uint buffer_size)
{
	std::cerr << "register_ping_pong_buffer called with handle=" << memory_handle << ", size=" << memory_size << ", buffer_count=" << buffer_count << ", buffer_size=" << buffer_size << std::endl;

	auto& graphics_session = get_session_();
    if (buffer_count > MAX_BUFFER_COUNT)
    {
        std::cerr << "Too many ping-pong buffers requested." << std::endl;
        return false;
    }

    unregister_vk_buffers_();

	cudaExternalMemoryHandleDesc mem_desc = {};
#ifdef _WIN32
	mem_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	mem_desc.handle.win32.handle = memory_handle;
#else
	mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	mem_desc.handle.fd = static_cast<int>(reinterpret_cast<intptr_t>(memory_handle));
#endif
	mem_desc.size = memory_size;

	cudaError_t err = cudaImportExternalMemory(&graphics_session.ping_pong_ext_memory, &mem_desc);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to import external memory with cuda error: "<< err << " - " << cudaGetErrorString(err) << std::endl;
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
		err = cudaExternalMemoryGetMappedBuffer( &cuda_ptr, graphics_session.ping_pong_ext_memory, &buffer_desc);

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

bool
register_cuda_vk_semaphores(void* vulkan_signal_handle, void* cuda_signal_handle)
{
	std::cerr << "register_cuda_vk_semaphores called with vulkan_signal_handle="
              << vulkan_signal_handle << ", cuda_signal_handle=" << cuda_signal_handle << std::endl;

    auto& graphics_session = get_session_();
    unregister_vk_semaphores_();

    cudaExternalSemaphoreHandleDesc wait_desc = {};
    cudaExternalSemaphoreHandleDesc signal_desc = {};
#ifdef _WIN32
    wait_desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    wait_desc.handle.win32.handle = vulkan_signal_handle;
    signal_desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    signal_desc.handle.win32.handle = cuda_signal_handle;
#else
    wait_desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    wait_desc.handle.fd = static_cast<int>(reinterpret_cast<intptr_t>(vulkan_signal_handle));
    signal_desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    signal_desc.handle.fd = static_cast<int>(reinterpret_cast<intptr_t>(cuda_signal_handle));
#endif

    cudaError_t err = cudaImportExternalSemaphore(&graphics_session.vulkan_signal_semaphore, &wait_desc);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to import Vulkan-to-CUDA semaphore: " << cudaGetErrorString(err) << std::endl;
        unregister_vk_semaphores_();
        return false;
    }

    err = cudaImportExternalSemaphore(&graphics_session.cuda_signal_semaphore, &signal_desc);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to import CUDA-to-Vulkan semaphore: " << cudaGetErrorString(err) << std::endl;
        unregister_vk_semaphores_();
        return false;
    }

    graphics_session.semaphores_init = true;
    return true;
}
