#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <span>

#include "../../defs.h"

namespace data_conversion
{
    class DataConverter
    {
    public:
        DataConverter() : _d_channel_mapping(nullptr) {};
        ~DataConverter()
        {
            CUDA_NULL_FREE(_d_channel_mapping);
        }

        bool copy_channel_mapping(std::span<const int16_t> channel_mapping);

        bool convert(const void* d_input, void* d_output, InputDataTypes input_type, uint2 input_dims, uint3 output_dims);

		bool convert_and_demod(const void* d_input, cuComplex* d_mid, cuComplex* d_output, InputDataTypes input_type, 
			uint2 input_dims, uint3 output_dims, float demod_freq, float sample_freq, const float* filter_coeffs, int filter_length);
        

    private:
        short* _d_channel_mapping;
    };
};