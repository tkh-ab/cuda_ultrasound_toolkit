#include "hilbert_kernels.cuh"
#include "hilbert_handler.h"

namespace rf_fft 
{
    void
    HilbertHandler::_cleanup_plans()
    {
        if (_forward_packed_plan != 0) {
            cufftDestroy(_forward_packed_plan);
            _forward_packed_plan = 0;
        }
        if (_forward_strided_plan != 0) {
            cufftDestroy(_forward_strided_plan);
            _forward_strided_plan = 0;
        }
        if (_inverse_plan != 0) {
            cufftDestroy(_inverse_plan);
            _inverse_plan = 0;
        }
        _fft_dims = { 0, 0 };
    }

    void
    HilbertHandler::_cleanup_filter()
    {
		CUDA_NULL_FREE(_d_filter);
    }

    bool
    HilbertHandler::plan_ffts(int2 fft_dims)
    {
        if (_fft_dims.x == fft_dims.x && _fft_dims.y == fft_dims.y) {
            return true; // Already planned
        }

        _cleanup_plans();
        _fft_dims = fft_dims;

        int signal_length = static_cast<int>(fft_dims.x);
        int double_signal_length = signal_length * 2;

        int rank = 1;
        //CUFFT_RETURN_IF_ERR(cufftPlanMany(&_forward_packed_plan, rank, &signal_length, &signal_length, 1, signal_length, &signal_length, 1, signal_length, CUFFT_R2C, fft_dims.y));

        CUFFT_RETURN_IF_ERR(cufftPlanMany(&_inverse_plan, rank, &signal_length, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, fft_dims.y));

        // This tells cufft to only grab every other float in the input (skipping the blank imaginary part)

        CUFFT_RETURN_IF_ERR(cufftPlanMany(&_forward_strided_plan, rank, &signal_length, &double_signal_length, 2, double_signal_length, &signal_length, 1, signal_length, CUFFT_R2C, fft_dims.y));

        return true;
    }
    
    bool
    HilbertHandler::load_filter(std::span<const float> match_filter)
    {
        _cleanup_filter();
        if (match_filter.empty()) {
            return true; // No filter to load
        }

        int signal_length = _fft_dims.x;
        size_t final_filter_size = signal_length * sizeof(cuComplex);
        int filter_length = static_cast<int>(match_filter.size());

        CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&_d_filter, final_filter_size));
        CUDA_RETURN_IF_ERROR(cudaMemset(_d_filter, 0x00, final_filter_size)); // Padded with zeros to signal_length
        CUDA_FLOAT_TO_COMPLEX_COPY(match_filter.data(), _d_filter, filter_length);

		// Filter is provided in time domain, take FFT 
        cufftHandle plan;
        CUFFT_RETURN_IF_ERR(cufftPlan1d(&plan, signal_length, CUFFT_C2C, 1));
        CUFFT_RETURN_IF_ERR(cufftExecC2C(plan, _d_filter, _d_filter, CUFFT_FORWARD));
        cufftDestroy(plan);

        return true;
    }

    bool
    HilbertHandler::_filter_and_scale(cuComplex* d_data)
    {
        uint sample_count = _fft_dims.x;
        uint cutoff = sample_count / 2 + 1;
		uint grid_length = (cutoff + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK; // Divide and round up
        uint channel_count = _fft_dims.y;

        // Grid dim y and z cant be bigger than 2^16 - 1, so we put channel count in X as it could pass that
        dim3 grid_dim(channel_count, grid_length, 1);
        dim3 block_dim(MAX_THREADS_PER_BLOCK, 1, 1);

        if(_d_filter)
        {
            kernels::scale_and_filter<true><<<grid_dim, block_dim>>>(d_data, _d_filter, sample_count, cutoff);
        }
        else
        {
            kernels::scale_and_filter<false><<<grid_dim, block_dim>>>(d_data, nullptr, sample_count, cutoff);
        }

        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

        return true;
    }

    bool
    HilbertHandler::packed_hilbert_and_filter(float* d_input, cuComplex* d_output)
    {
        if (!_forward_packed_plan || !_inverse_plan) {
            return false; // FFTs not planned
        }

        size_t output_size = _fft_dims.x * _fft_dims.y * sizeof(cuComplex);
        CUDA_RETURN_IF_ERROR(cudaMemset(d_output, 0x00, output_size)); // Clear output buffer

        CUFFT_RETURN_IF_ERR(cufftExecR2C(_forward_packed_plan, d_input, d_output));
        CUDA_RETURN_IF_ERROR(cudaGetLastError());

        if (!_filter_and_scale(d_output)) 
        {
            std::cerr << "Error filtering RF data" << std::endl;
            return false;
        }

        CUFFT_RETURN_IF_ERR(cufftExecC2C(_inverse_plan, d_output, d_output, CUFFT_INVERSE));
        CUDA_RETURN_IF_ERROR(cudaGetLastError());

        return true;
    }

    bool
    HilbertHandler::strided_hilbert_and_filter(float* d_input, cuComplex* d_output)
    {
        if (!_forward_strided_plan || !_inverse_plan) {
            return false; // FFTs not planned
        }

        size_t output_size = _fft_dims.x * _fft_dims.y * sizeof(cuComplex);
        CUDA_RETURN_IF_ERROR(cudaMemset(d_output, 0x00, output_size)); // Clear output buffer

        CUFFT_RETURN_IF_ERR(cufftExecR2C(_forward_strided_plan, d_input, d_output));
        CUDA_RETURN_IF_ERROR(cudaGetLastError());

        if (!_filter_and_scale(d_output)) 
        {
            std::cerr << "Error filtering RF data" << std::endl;
            return false;
        }

        CUFFT_RETURN_IF_ERR(cufftExecC2C(_inverse_plan, d_output, d_output, CUFFT_INVERSE));
        CUDA_RETURN_IF_ERROR(cudaGetLastError());

        return true;
    }
}