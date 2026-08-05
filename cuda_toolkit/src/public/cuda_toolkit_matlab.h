#ifndef CUDA_TOOLKIT_MATLAB_H
#define CUDA_TOOLKIT_MATLAB_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_beamformer_parameters.h"

#if defined(_WIN32)
    #define LIB_FN __declspec(dllexport)
#else
    #define LIB_FN
#endif

LIB_FN void beamform_i16_matlab(const short* data, CudaBeamformerParameters bp, float* output);

LIB_FN void beamform_f32_matlab(const float* data, CudaBeamformerParameters bp, float* output);

LIB_FN void motion_detect_f32_matlab(const float* images, NCCMotionParameters params, float* motion_maps);

LIB_FN void corr_images_f32_matlab(const float* template_image, const float* source_image, 
									CorrImagesParameters params, float* corr_map);

#ifdef __cplusplus
}   // extern "C"  
#endif
#endif // !CUDA_TOOLKIT_MATLAB_H
