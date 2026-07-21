
#include "data_conversion_kernels.cuh"
#include "data_converter.h"



namespace data_conversion
{
    bool
    DataConverter::copy_channel_mapping(std::span<const int16_t> channel_mapping)
    {
        CUDA_NULL_FREE(_d_channel_mapping);

        size_t size = channel_mapping.size() * sizeof(short);
        CUDA_RETURN_IF_ERROR(cudaMalloc(&_d_channel_mapping, size));
        CUDA_RETURN_IF_ERROR(cudaMemcpy(_d_channel_mapping, channel_mapping.data(), size, cudaMemcpyHostToDevice));
        return true;
    }

    bool
    DataConverter::convert(const void* d_input, void* d_output, InputDataTypes input_type, uint2 input_dims, uint3 output_dims)
    {
        if (!_d_channel_mapping)
        {
            std::cerr << "Channel mapping not set." << std::endl;
            return false; // Channel mapping not set
        }

        dim3 block_dim(MAX_THREADS_PER_BLOCK, 1, 1);
        uint grid_length = (uint)ceil((double)input_dims.x / MAX_THREADS_PER_BLOCK);
        dim3 grid_dim(grid_length, output_dims.y, 1);


        switch(input_type)
        {
            case InputDataTypes::TYPE_I16:
            {
                using T = types::type_for_t<InputDataTypes::TYPE_I16>;
                static_assert(std::is_same_v<T, int16_t>, "Type mismatch for I16 conversion");
                kernels::convert_to_f32<T><<<grid_dim, block_dim>>>(static_cast<const T*>(d_input), static_cast<float*>(d_output), input_dims, output_dims, _d_channel_mapping);
                break;
            }
            case InputDataTypes::TYPE_F32:
            {
                using T = types::type_for_t<InputDataTypes::TYPE_F32>;
                static_assert(std::is_same_v<T, float>, "Type mismatch for F32 conversion");
                kernels::convert_to_f32<T><<<grid_dim, block_dim>>>(static_cast<const T*>(d_input), static_cast<float*>(d_output), input_dims, output_dims, _d_channel_mapping);
                break;
            }
            default:
            {
                std::cerr << "Data converter: Unsupported input data type." << std::endl;
                return false;
            } 
        }

        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

        return true;
    }

	bool
	DataConverter::convert_and_demod(const void* d_input, cuComplex* d_mid, cuComplex* d_output, 
		InputDataTypes input_type, uint2 input_dims, uint3 output_dims, float demod_freq, float sample_freq,
		const float* filter_coeffs, int filter_length)
	{
		if (!_d_channel_mapping)
        {
            std::cerr << "Channel mapping not set." << std::endl;
            return false; // Channel mapping not set
        }

        dim3 block_dim(MAX_THREADS_PER_BLOCK, 1, 1);
        uint grid_length = (uint)ceil((double)input_dims.x / MAX_THREADS_PER_BLOCK / 2); // Divide by 2 for complex samples
        dim3 grid_dim(grid_length, output_dims.y, 1);

		CUDA_RETURN_IF_ERROR(cudaMemset(d_mid, 0x00, output_dims.x * output_dims.y * output_dims.z * sizeof(cuComplex)));
		CUDA_RETURN_IF_ERROR(cudaMemset(d_output, 0x00, output_dims.x * output_dims.y * output_dims.z * sizeof(cuComplex)));

        switch(input_type)
        {
            case InputDataTypes::TYPE_I16:
            {
                using T = types::type_for_t<InputDataTypes::TYPE_I16>;
                static_assert(std::is_same_v<T, int16_t>, "Type mismatch for I16 conversion");
                kernels::convert_demod_cf32<T><<<grid_dim, block_dim>>>(static_cast<const T*>(d_input), d_mid, input_dims, output_dims, _d_channel_mapping, demod_freq, sample_freq);
                break;
            }
            case InputDataTypes::TYPE_F32:
            {
                using T = types::type_for_t<InputDataTypes::TYPE_F32>;
                static_assert(std::is_same_v<T, float>, "Type mismatch for F32 conversion");
                kernels::convert_demod_cf32<T><<<grid_dim, block_dim>>>(static_cast<const T*>(d_input), d_mid, input_dims, output_dims, _d_channel_mapping, demod_freq, sample_freq);
                break;
            }
            default:
            {
                std::cerr << "Data converter: Unsupported input data type." << std::endl;
                return false;
            } 
        }

        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

		float* d_filter;
		if(filter_length > 0)
		{
			std::cout << "Applying filter of length: " << filter_length << std::endl;
			CUDA_RETURN_IF_ERROR(cudaMalloc(&d_filter, filter_length * sizeof(float)));
			CUDA_RETURN_IF_ERROR(cudaMemcpy(d_filter, filter_coeffs, filter_length * sizeof(float), cudaMemcpyHostToDevice));

			grid_length = (uint)ceil((double)output_dims.x / MAX_THREADS_PER_BLOCK);
			grid_dim = dim3(grid_length, output_dims.y, output_dims.z);
			kernels::td_filter_fc32<<<grid_dim, block_dim>>>(d_mid, d_output, output_dims, d_filter, filter_length);
			CUDA_RETURN_IF_ERROR(cudaGetLastError());
        	CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

			cudaFree(d_filter);
		}
		else
		{
			CUDA_RETURN_IF_ERROR(cudaMemcpy(d_output, d_mid, output_dims.x * output_dims.y * output_dims.z * sizeof(cuComplex), cudaMemcpyDeviceToDevice));	
		}

		print_buffer<cuComplex>(d_output, 16, "Output after demod:");


        return true;
	}
}