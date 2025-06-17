#ifndef CUDA_TRANSFER_HH
#define CUDA_TRANSFER_HH

#ifdef __cplusplus
extern "C" {
#endif

#include "communication_params.h"

#if defined(_WIN32)
    #define LIB_FN __declspec(dllexport)
#else
    #define LIB_FN
#endif

LIB_FN void beamform_i16( const short* data, CudaBeamformerParameters bp, float* output);

LIB_FN void beamform_f32(const float* data, CudaBeamformerParameters bp, float* output);

LIB_FN void motion_detect_f32(const float* images, NCCMotionParameters params, int* motion_maps);

LIB_FN void deinit();

#ifdef __cplusplus
}   // extern "C"  
#endif
#endif // !CUDA_TRANSFER_HH