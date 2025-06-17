#ifndef CUDA_TOOLKIT_HPP
#define CUDA_TOOLKIT_HPP

#ifdef __cplusplus

#ifdef _WIN32
	#define EXPORT_FN __declspec(dllexport)
#else
	#define EXPORT_FN
#endif

#include <span>
#include "cuda_beamformer_parameters.h"

namespace cuda_toolkit
{
    EXPORT_FN bool beamform(std::span<const uint8_t> input_data, 
                  std::span<uint8_t> output_data, 
                  const CudaBeamformerParameters& bp);

	EXPORT_FN bool motion_detection(std::span<const uint8_t> images, 
				  std::span<uint8_t> motion_maps,
				  const NccMotionParameters& params);
}

#endif // __cplusplus
#endif // !CUDA_TOOLKIT_HPP