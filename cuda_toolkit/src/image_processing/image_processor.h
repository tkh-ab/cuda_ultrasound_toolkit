#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <npp.h>
#include <cuda_runtime.h>
#include <span>

#include "kernels/block_match.h"

#include "../defs.h"

template <typename T>
concept SupportedNccType = std::is_same_v<T, float> || std::is_same_v<T, uint8_t>;

class ImageProcessor
{

public:

    // Constructor
    ImageProcessor() 
	{
		_default_stream_context = _create_stream_context(0); // Default stream for syncronous operations
	}
	~ImageProcessor()
	{
		_clear_pipeline_contexts();
	}

    bool ncc_block_match( std::vector<PitchedArray<float>>& d_input_images, 
                            float4* motion_maps, 
                            const NccMotionParameters& params);

	bool corr_images(const PitchedArray<float>& d_template_image,
					 const PitchedArray<float>& d_source_image,
					 float* corr_map,
					 uint2 template_dims,
					 uint2 source_dims);


	uint get_pipeline_count() const
	{
		return static_cast<uint>(_pipeline_contexts.size());
	}


private:

		static constexpr size_t Min_Scratch_Buffer_Size = 1024 * 4; // 4 KB

        // Creates the context for the default cuda stream
        NppStreamContext _create_stream_context(cudaStream_t stream);

		bool 
		_compare_images(const PitchedArray<float>& template_image,
							const PitchedArray<float>& source_image,
							float4* motion_map,
							uint2 image_dims,
							const NccMotionParameters& params);

		bool
		_create_pipeline_ctxs(const NccMotionParameters& params, uint stream_count);

		void
		_clear_pipeline_contexts()
		{
			for (auto& ctx : _pipeline_contexts)
			{
				if (ctx.d_scratch_buffer) {
					cudaFree(ctx.d_scratch_buffer);
					ctx.d_scratch_buffer = nullptr;
				}
				if (ctx.d_corr_map) {
					cudaFree(ctx.d_corr_map);
					ctx.d_corr_map = nullptr;
				}
				if (ctx.stream) {
					cudaStreamDestroy(ctx.stream);
					ctx.stream = nullptr;
				}
			}
			_pipeline_contexts.clear();
		}

		std::vector<block_match::PipelineCtx> _pipeline_contexts;

		NppStreamContext _default_stream_context;

    
};





#endif // IMAGE_PROCESSOR_H
