#pragma once
#include "beamformer_constants.cuh"

#define PATH_LENGTH_SIGN(FOCUS, VOXAL) round((VOXAL.z - FOCUS.z) / abs(VOXAL.z - FOCUS.z))

namespace bf_kernels::utils
{
    __device__ inline float3
    calc_tx_distance(float3 vox_loc, float3 source_pos, FocalDirection direction)
    {
        float3 tx_distance;
        if (direction == FocalDirection::XZ_PLANE)
        {
            tx_distance = { source_pos.x - vox_loc.x, 0.0f, source_pos.z - vox_loc.z};
        }
        else if (direction == FocalDirection::YZ_PLANE)
        {
            tx_distance = { 0.0f, source_pos.y - vox_loc.y, source_pos.z - vox_loc.z };
        }
        else
        {
            tx_distance = {0.0f, 0.0f, vox_loc.z };
        }

        return tx_distance;
    }

    __device__ inline bool
    check_ranges(float3 vox_loc, float f_number, float2 array_edges)
    {
        // Get the max aperture size for this depth
        float lateral_extent = vox_loc.z / f_number;

        // Model 2 1D apertures to maintain square planes
        float x_extent = lateral_extent + array_edges.x;
        float y_extent = lateral_extent + array_edges.y;

        return (abs(vox_loc.x) < x_extent && abs(vox_loc.y) < y_extent);
    }

    __device__ inline float
	clampf(float value, float min_value, float max_value)
	{
		return fmaxf(min_value, fminf(value, max_value));
	}

    __device__ inline cuComplex 
    cubic_spline(int channel_offset, float x, const cuComplex* rf_data)
    {
        static constexpr float C_SPLINE = 0.5f;

        // Hermite basis matrix (transposed for row vector * matrix order)
        float h[4][4] = {
            { 2.0f, -2.0f,  1.0f,  1.0f},
            {-3.0f,  3.0f, -2.0f, -1.0f},
            { 0.0f,  0.0f,  1.0f,  0.0f},
            { 1.0f,  0.0f,  0.0f,  0.0f}
        };

        int x_whole = static_cast<int>(floorf(x));
        float xr = x - static_cast<float>(x_whole);
        float S[4] = { xr * xr * xr, xr * xr, xr, 1.0f };

        cuComplex P1 = rf_data[channel_offset + x_whole];
        cuComplex P2 = rf_data[channel_offset + x_whole + 1];
        cuComplex T1 = {
            C_SPLINE * (P2.x - rf_data[channel_offset + x_whole - 1].x),
            C_SPLINE * (P2.y - rf_data[channel_offset + x_whole - 1].y)
        };
        float2 T2 = {
            C_SPLINE * (rf_data[channel_offset + x_whole + 2].x - P1.x),
            C_SPLINE * (rf_data[channel_offset + x_whole + 2].y - P1.y)
        };

        float Cx[4] = { P1.x, P2.x, T1.x, T2.x };
        float Cy[4] = { P1.y, P2.y, T1.y, T2.y };

        float result_x = 0.0f;
        float result_y = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float h_sum = 0.0f;
            for (int j = 0; j < 4; ++j) {
                h_sum += S[j] * h[j][i];
            }
            result_x += h_sum * Cx[i];
            result_y += h_sum * Cy[i];
        }

        return { result_x, result_y };
    }

    __device__ inline float
    f_num_apodization(float lateral_dist_ratio, float depth, float f_num)
    {
        // When lateral_dist > depth / f_num clamp the argument to pi/2 so that the cos is 0
        // Otherwise the ratio will map between 0 and pi/2 forming a hann window
        float apo = f_num * (lateral_dist_ratio / depth) /2;
        apo = fminf(apo, 0.5);
        apo = cosf(CUDART_PI_F * apo);
        return apo * apo; // cos^2
    }

    __device__ __inline__ bool
    offset_mixes(int transmit, int element, int mixes_spacing, int offset, int pivot)
    {
        int transmit_offset = 0;
        int element_offset = 0;

        if (transmit >= pivot) element_offset = offset;
        if (element >= pivot) transmit_offset = offset;
        
        if (element % mixes_spacing != element_offset && transmit % mixes_spacing != transmit_offset)
        {
            return false;
        }

        return true;
    }
    
    __device__ inline cuComplex
    reduce_shared_sum(const cuComplex* sharedVals, const uint channel_count)
    {
        // Each thread will compute a partial sum.
        int tid = threadIdx.x;

        // Compute partial sum over a stripe of the shared memory.
        cuComplex partial_sum = { 0.0f, 0.0f };
        // Each thread processes elements starting at its index and strides by the block size.
        for (int i = tid; i < channel_count; i += blockDim.x) {
            partial_sum = ADD_V2( sharedVals[i], partial_sum);
        }

        __shared__ cuComplex aux[MAX_TX_COUNT];
        aux[tid] = partial_sum;
        __syncthreads();

        // Perform iterative tree-based reduction.
        // Since blockDim.x is 128, we reduce until we have one value.
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                aux[tid] = ADD_V2(aux[tid], aux[tid + stride]);
            }
            __syncthreads();
        }
        return aux[0];
    }

    __device__ inline float
    total_path_length(float3 tx_vec, float3 rx_vec, float focal_depth, float sign)
    {
        return focal_depth + NORM_F3(rx_vec) + NORM_F3(tx_vec) * sign;
    }

}