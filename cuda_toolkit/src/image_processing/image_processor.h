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

	bool _compare_images(const PitchedArray<float>& template_image,
						const PitchedArray<float>& source_image,
						std::span<int2> motion_map, 
						uint2 image_dims, 
						const NccMotionParameters& params);

        // Creates the context for the default cuda stream
        NppStreamContext _create_stream_context();

        NppStreamContext _stream_context;



    
};



namespace process_helpers
{

	inline uint2
	find_peak(const float* d_corr_map, int width, int height)
	{
		float max_val = -FLT_MAX;
		uint2 max_pos = { 0, 0 };

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				float val = sample_value<float>(d_corr_map + y * width + x);
				if (val > max_val)
				{
					max_val = val;
					max_pos = { (uint)x, (uint)y };
				}
			}
		}

		return max_pos;
	}

};






#endif // IMAGE_PROCESSOR_H