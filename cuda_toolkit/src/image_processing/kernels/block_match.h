#pragma once
#include <npp.h>
#include <span>
#include "../../defs.h"


namespace block_match
{
	static constexpr dim3 Peak_Detect_Block_Dims = { 8, 4, 1 };

	struct PipelineCtx {
		NppStreamContext stream_context;
		cudaStream_t stream;

		u8* d_scratch_buffer = nullptr;
		size_t scratch_buffer_size = 0;

		float* d_corr_map = nullptr;
		size_t corr_map_size = 0;
	};

	__host__ bool
	block_match_pipeline(const float* d_source, const float* d_template, float4* d_motion_map,
						 NppiSize src_roi, NppiSize tpl_roi,
						 int src_line_step, int tpl_line_step, PipelineCtx& ctx,
						 int2 no_shift_index, const NccMotionParameters& params);

};
