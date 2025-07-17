
#include <span>
#include "../../defs.h"


namespace block_match
{

	__host__ inline int2
	find_peak(const float* d_corr_map, NppiSize dims)
	{
		float max_val = -1.0f;
		int2 max_pos = { 0, 0 };

		for (int y = 0; y < dims.height; ++y)
		{
			for (int x = 0; x < dims.width; ++x)
			{
				float val = abs(sample_value<float>(d_corr_map + y * dims.width + x));
				if (val > max_val)
				{
					max_val = val;
					max_pos = { x, y };
				}
			}
		}

		return max_pos;
	}

	__host__ int2
	select_peak(const float* d_corr_map, NppiSize dims, const NccMotionParameters& params, Npp8u* d_scratch_buffer, NppStreamContext stream_context, int2 no_shift_pos);

};