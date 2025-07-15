
#include <span>
#include "../../defs.h"


namespace block_match
{

	__host__ bool 
	compare_images(const PitchedArray<float>& template_image,
						const PitchedArray<float>& source_image,
						std::span<int2> motion_map, 
						uint2 image_dims, 
						const NccMotionParameters& params,
						NppStreamContext stream_context);



	__host__ inline uint2
	find_peak(const float* d_corr_map, NppiSize dims)
	{
		float max_val = -1.0f;
		uint2 max_pos = { 0, 0 };

		for (int y = 0; y < dims.height; ++y)
		{
			for (int x = 0; x < dims.width; ++x)
			{
				float val = abs(sample_value<float>(d_corr_map + y * dims.width + x));
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