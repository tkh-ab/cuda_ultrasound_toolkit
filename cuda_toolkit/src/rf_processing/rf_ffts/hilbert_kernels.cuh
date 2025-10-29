#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "../../defs.h"

namespace rf_fft::kernels
{
    __device__ inline float2 
    complex_multiply_f2(float2 a, float2 b) {
        return make_float2(a.x * b.x - a.y * b.y,
                           a.x * b.y + a.y * b.x);
    }

    // This is templated so that we can remove the branch applying the filter kernel
    // Block dims are <128, 1, 1>, each thread is a sample.
    // Grid dims are <channel_count, sample_count / 128, 1>
    template <bool UseFilter> __global__ void
    scale_and_filter(cuComplex* spectrums, const cuComplex* filter_kernel, uint sample_count, uint cutoff)
    {
        uint sample_idx = threadIdx.x + blockIdx.y * blockDim.x;
        if (sample_idx > cutoff) return; // We only need to process the first half of the spectrum.

        // This should be 2/N, 2 for the hilbert transform and 1/N for the ifft normalization.
        // but atm 1.5/N is what matches the power of the input signal for some reason.
        float scale_factor = 1.5f / ((float)sample_count);

        // Scale the DC and Nyquist components by 0.5 compared to the rest of the spectrum (analytic signal)
        if (sample_idx == 0 || sample_idx == cutoff) scale_factor *= 0.5f;

        uint channel_offset = blockIdx.x * sample_count;
        cuComplex scaled_output = SCALE_V2(spectrums[channel_offset + sample_idx], scale_factor);

        // Templating away this branch
        if constexpr (UseFilter)
        {
            scaled_output = complex_multiply_f2(scaled_output, filter_kernel[sample_idx]);
        }

        spectrums[channel_offset + sample_idx] = scaled_output;
    }
}