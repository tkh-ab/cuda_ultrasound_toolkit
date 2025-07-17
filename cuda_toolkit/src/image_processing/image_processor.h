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

    bool ncc_forward_match( std::vector<PitchedArray<float>>& d_input_images, 
                            int2* motion_maps, 
                            const NccMotionParameters& params);

    bool svd_filter(std::span<const cuComplex> input, 
                    std::span<cuComplex> output, 
                    uint2 image_dims, 
                    std::span<const uint > mask);

private:

        // Creates the context for the default cuda stream
        NppStreamContext _create_stream_context();

		bool 
		_compare_images(const PitchedArray<float>& template_image,
							const PitchedArray<float>& source_image,
							int2* motion_map,
							uint2 image_dims,
							const NccMotionParameters& params);

		bool
		_create_buffers(NppiSize src_size, NppiSize tpl_size);

		void
		_cleanup_buffers()
		{
			if (_d_scratch_buffer)
			{
				cudaFree(_d_scratch_buffer);
				_d_scratch_buffer = nullptr;
			}
			if (_d_corr_map)
			{
				cudaFree(_d_corr_map);
				_d_corr_map = nullptr;
			}
			_scratch_buffer_size = 0;
			_corr_map_size = 0;
		}

        NppStreamContext _stream_context;

		u8* _d_scratch_buffer = nullptr;
		size_t _scratch_buffer_size = 0;

		float* _d_corr_map = nullptr;
		size_t _corr_map_size = 0;
    
};





#endif // IMAGE_PROCESSOR_H