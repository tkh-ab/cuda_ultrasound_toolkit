#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <npp.h>
#include <cuda_runtime.h>
#include <span>

#include "../defs.h"

template <typename T>
concept SupportedNccType = std::is_same_v<T, float> || std::is_same_v<T, uint8_t>;

class ImageProcessor
{

public:
    // Constructor
    ImageProcessor() {
        _stream_context = _create_stream_context();
    }

    bool ncc_forward_match(std::span<const u8> input, 
                            std::span<u8> motion_maps, 
                            const NccMotionParameters& params);

    bool svd_filter(std::span<const cuComplex> input, 
                    std::span<cuComplex> output, 
                    uint2 image_dims, 
                    std::span<const uint > mask);

private:

        // Creates the context for the default cuda stream
        NppStreamContext _create_stream_context();

        NppStreamContext _stream_context;
    
};





#endif // IMAGE_PROCESSOR_H