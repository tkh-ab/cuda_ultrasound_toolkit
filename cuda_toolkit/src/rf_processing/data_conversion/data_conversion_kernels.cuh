#pragma once

#include <cstddef>
#include <concepts>
#include <type_traits>
#include <cuda_runtime.h>

#include "../../defs.h"


namespace data_conversion::kernels 
{    

	__device__ __forceinline__ inline
	cuComplex iq_demod(cuComplex sample, int index, float demod_freq, float sample_freq)
	{
		float t = TWO_PI_F * index * demod_freq / sample_freq;
		cuComplex demod = {cosf(t), -sinf(t)};
		return cuCmulf(sample, demod);
	}

    template <typename T>
    concept SupportedConversionType = std::is_same_v<T, int16_t> || std::is_same_v<T, float> || std::is_same_v<T, cuComplex>;
    
    template<SupportedConversionType T> __global__ void
    convert_to_f32(const T* input, float* output, uint2 input_dims, uint3 output_dims, const short* d_channel_mapping)
    {
        uint raw_sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
        uint output_channel_idx = blockIdx.y;
        uint tx_idx = raw_sample_idx / output_dims.x;
        uint output_sample_idx = raw_sample_idx % output_dims.x;

        if (tx_idx >= output_dims.z) return;

        uint raw_channel_idx = d_channel_mapping[output_channel_idx];
        
        uint input_idx = (raw_channel_idx * input_dims.x) + raw_sample_idx;
        uint output_idx = (tx_idx * output_dims.y * output_dims.x) + (output_channel_idx * output_dims.x) + output_sample_idx;

        output[output_idx] = static_cast<float>(input[input_idx]);
    }


	// Demod for bandpass data where every two samples are an IQ pair
	// We still need to demod with beating
	template<SupportedConversionType T> __global__ void
    convert_demod_cf32(const T* input, cuComplex* output, uint2 input_dims, uint3 output_dims, 
		const short* d_channel_mapping, float demod_freq, float sample_freq)
    {
        uint raw_sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
        uint output_channel_idx = blockIdx.y;
        uint tx_idx = raw_sample_idx / output_dims.x;
        uint output_sample_idx = raw_sample_idx % output_dims.x;

        if (raw_sample_idx * 2 >= input_dims.x) return;

        uint raw_channel_idx = d_channel_mapping[output_channel_idx]; 
        
        uint input_idx = (raw_channel_idx * input_dims.x) + raw_sample_idx * 2;
        uint output_idx = (tx_idx * output_dims.y * output_dims.x) + (output_channel_idx * output_dims.x) + output_sample_idx;

        cuComplex sample = {static_cast<float>(input[input_idx]), -1 * static_cast<float>(input[input_idx + 1])};
		//float test = abs(demod_freq - sample_freq);
		sample = iq_demod(sample, output_sample_idx, demod_freq, sample_freq);
		output[output_idx] = sample;
    }

	__global__ void
	td_filter_fc32(cuComplex* input, cuComplex* output, uint3 data_dims, const float* d_filter, uint filter_length)
	{
		uint sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
		uint channel_idx = blockIdx.y;
		uint tx_idx = blockIdx.z;

		if (sample_idx >= data_dims.x) return;

		uint base_offset =
			tx_idx * data_dims.y * data_dims.x +
			channel_idx * data_dims.x;

		cuComplex result = make_cuComplex(0.0f, 0.0f);

		int half_len = static_cast<int>(filter_length / 2);
		int sample_count = static_cast<int>(data_dims.x);
		int total_valid = 0;
		for (uint i = 0; i < filter_length; ++i)
		{
			int input_idx = static_cast<int>(sample_idx) +
							static_cast<int>(i) -
							half_len;

			if (input_idx < 0 || input_idx >= sample_count)
				continue;

			cuComplex x = input[base_offset + input_idx];
			float h = d_filter[filter_length - 1 - i];

			result = cuCaddf(result, SCALE_V2(x, h));
			++total_valid;
		}

		output[base_offset + sample_idx] = SCALE_V2(result, 1.0f / static_cast<float>(total_valid));
	}


	// template<SupportedConversionType T> __global__ void
    // convert_demod_cf32(const T* input, cuComplex* output, uint2 input_dims, uint3 output_dims, const short* d_channel_mapping, float demod_freq, float sample_freq)
    // {
    //     uint output_channel_idx = threadIdx.x;
	// 	uint acq_idx = blockIdx.x;

	// 	uint input_channel_idx = d_channel_mapping[output_channel_idx];
	// 	uint raw_sample_idx = acq_idx * output_dims.x * 2;

	// 	uint input_offset = (input_channel_idx * input_dims.x) + raw_sample_idx;
	// 	uint output_offset = (acq_idx * output_dims.y * output_dims.x) + (output_channel_idx * output_dims.x);

	// 	for (uint i = 0; i < output_dims.x; ++i)
	// 	{
	// 		uint input_idx = i * 2 + input_offset;
	// 		uint output_idx = i + output_offset;
	// 		cuComplex sample = {static_cast<float>(input[input_idx]), -1 * static_cast<float>(input[input_idx + 1])};
	// 		output[output_idx] = iq_demod(sample, i, demod_freq, sample_freq);
	// 	}
    // }
}