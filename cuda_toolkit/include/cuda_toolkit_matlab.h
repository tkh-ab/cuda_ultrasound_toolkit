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

LIB_FN void beamform_i16( const short* data, CudaBeamformerParameters bp, float* output);

#ifdef __cplusplus
}   // extern "C"  
#endif
#endif // !CUDA_TOOLKIT_MATLAB_H
