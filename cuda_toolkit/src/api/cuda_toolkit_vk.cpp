#include "../rf_processing/rf_processor.h"
#include "../public/cuda_beamformer_parameters.h"
#include "../public/cuda_toolkit_vk.h"

#define MAX_BUFFER_COUNT 16

bool unregister_ogl_buffers_();

struct GraphicsSession 
{
    RfProcessor rf_processor;
    std::pair<uint, cudaGraphicsResource_t> ogl_raw_buffer;
    std::array<std::pair<uint, cudaGraphicsResource_t>, MAX_BUFFER_COUNT> ogl_rf_buffers;
    bool buffers_init = false;

    GraphicsSession() : ogl_raw_buffer({0, nullptr}), buffers_init(false)
    {
        ogl_rf_buffers.fill({0, nullptr});
    }
    ~GraphicsSession()
    {
        unregister_ogl_buffers_();
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
unregister_ogl_buffers_()
{
    auto& graphics_session = get_session_();
    auto& ogl_raw_buffer = graphics_session.ogl_raw_buffer;
    cudaError_t err;
    if (ogl_raw_buffer.second)
    {
        err = cudaGraphicsUnregisterResource(ogl_raw_buffer.second);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to unregister OpenGL raw buffer: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "Shared buffers may be in an invalid state." << std::endl;
            return false;
        }
        ogl_raw_buffer.second = nullptr;
        ogl_raw_buffer.first = 0;
    }

    for (auto& pair : graphics_session.ogl_rf_buffers)
    {
        if (pair.second)
        {
            err = cudaGraphicsUnregisterResource(pair.second);
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to unregister OpenGL RF buffer: " << cudaGetErrorString(err) << std::endl;
                std::cerr << "Shared buffers may be in an invalid state." << std::endl;
                return false;
            }

            pair.second = nullptr;
            pair.first = 0;
        }
    }

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
    unregister_ogl_buffers_();
}

bool
register_cuda_buffers(const uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo)
{
    unregister_ogl_buffers_();

    GraphicsSession& graphics_session = get_session_();
    
    if(rf_buffer_count > MAX_BUFFER_COUNT)
    {
        std::cerr << "Too many RF data buffers. Maximum is " << MAX_BUFFER_COUNT << "." << std::endl;
        return false;
    }

    auto& ogl_raw_buffer = graphics_session.ogl_raw_buffer;
    ogl_raw_buffer.first = raw_data_ssbo;

    CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&(ogl_raw_buffer.second), raw_data_ssbo, cudaGraphicsRegisterFlagsNone));

    auto it = graphics_session.ogl_rf_buffers.begin();
    for (uint i = 0; i < rf_buffer_count; ++i, ++it)
    {
        uint ssbo = rf_data_ssbos[i];
        if (ssbo == 0)
        {
            std::cerr << "Invalid SSBO at index " << i << "." << std::endl;
            return false;
        }
        cudaGraphicsResource_t rf_resource;
        CUDA_RETURN_IF_ERROR(cudaGraphicsGLRegisterBuffer(&rf_resource, ssbo, cudaGraphicsRegisterFlagsNone));
        *it = { ssbo, rf_resource };
    }
    graphics_session.buffers_init = true;
    return true;
}

bool
cuda_set_channel_mapping(const i16 channel_mapping[MAX_CHANNEL_COUNT])
{
    RfProcessor& rf_processor = get_session_().rf_processor;
    
    std::span<const int16_t> mapping_span(channel_mapping, MAX_CHANNEL_COUNT);
    if (!rf_processor.set_channel_mapping(mapping_span))
    {
        std::cerr << "Failed to set channel mapping." << std::endl;
        return false;
    }
    
    return true;
}

bool
cuda_set_match_filter(const float* match_filter, uint length)
{
    RfProcessor& rf_processor = get_session_().rf_processor;

    std::span<const float> filter_span(match_filter, length);
    if (!rf_processor.set_match_filter(filter_span))
    {
        std::cerr << "Failed to set match filter." << std::endl;
        return false;
    }
    
    return true;
}

bool
cuda_decode(size_t input_offset, uint output_buffer_idx)
{
    GraphicsSession& graphics_session = get_session_();

    if (!graphics_session.buffers_init)
    {
        std::cerr << "OGL buffers not registered." << std::endl;
        return false;
    }

    RfProcessor& rf_processor = graphics_session.rf_processor;
    auto& ogl_raw_buffer = graphics_session.ogl_raw_buffer;
    auto& ogl_rf_buffers = graphics_session.ogl_rf_buffers;

    auto input_resource = ogl_raw_buffer.second;
    auto output_resource = ogl_rf_buffers[output_buffer_idx].second;

    int16_t* d_input = nullptr;
    cuComplex* d_output = nullptr;

    if (!map_ogl_buffer_((void**)&d_input, input_resource))
    {
        std::cerr << "Failed to map input OpenGL buffer." << std::endl;
        return false;
    }
    if (!map_ogl_buffer_((void**)&d_output, output_resource))
    {
        std::cerr << "Failed to map output OpenGL buffer." << std::endl;
        unmap_ogl_buffer_(input_resource);
        return false;
    }

    size_t input_offset_count = input_offset / sizeof(int16_t);

    bool result = rf_processor.convert_decode_strided(d_input + input_offset_count, d_output, InputDataTypes::TYPE_I16);
    if (!result)
    {
        std::cerr << "Failed to decode data." << std::endl;
    }
    unmap_ogl_buffer_(input_resource);
    unmap_ogl_buffer_(output_resource);
    return result;
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
    auto& ogl_rf_buffers = graphics_session.ogl_rf_buffers;

    auto input_resource = ogl_rf_buffers[input_buffer_idx].second;
    auto output_resource = ogl_rf_buffers[output_buffer_idx].second;

    float* d_input = nullptr;
    cuComplex* d_output = nullptr;

    if (!map_ogl_buffer_((void**)&d_input, input_resource))
    {
        std::cerr << "Failed to map input OpenGL buffer." << std::endl;
        return false;
    }
    if (!map_ogl_buffer_((void**)&d_output, output_resource))
    {
        std::cerr << "Failed to map output OpenGL buffer." << std::endl;
        unmap_ogl_buffer_(input_resource);
        return false;
    }
    
    bool result = rf_processor.hilbert_transform_strided(d_input, d_output);
    unmap_ogl_buffer_(input_resource);
    unmap_ogl_buffer_(output_resource);
    return result;
}

