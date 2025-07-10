#ifndef DEFS_H
#define DEFS_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include <assert.h>

#include <string>
#include <chrono>
#include <vector>
#include <complex>

#include <iostream>

#include <npp.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_gl_interop.h>

#include "public/cuda_beamformer_parameters.h"

constexpr float NaN = (float)0xFFFFFFFF; 

constexpr float I_SQRT_64 = 0.125f;
constexpr float I_SQRT_128 = 0.088388347648318f;
constexpr float PI_F = 3.141592654f;
constexpr double PI_D = 3.141592653589793;

constexpr int MAX_THREADS_PER_BLOCK = 128;
constexpr int MAX_2D_BLOCK_DIM = 32;
constexpr int WARP_SIZE = 32;
constexpr int TOTAL_TOBE_CHANNELS = 256;

typedef unsigned int uint;

typedef char      c8;
typedef uint8_t   u8;
typedef int16_t   i16;
typedef uint16_t  u16;
typedef int32_t   i32;
typedef uint32_t  u32;
typedef int64_t   i64;
typedef uint64_t  u64;
typedef uint32_t  b32;
typedef float     f32;
typedef double    f64;
typedef ptrdiff_t size;
typedef ptrdiff_t iptr;

#define ISPOWEROF2(a)  (((a) & ((a) - 1)) == 0)

#define SCALAR_ABS(x)         ((x) < 0 ? -(x) : (x))
#define SCALE_F2(v, a) {(v).x * (a), (v).y * (a)}

#define NORM_F2(v) (sqrtf( (v).x * (v).x + (v).y * (v).y))
#define NORM_F3(v) (sqrtf( (v).x * (v).x + (v).y * (v).y + (v).z * (v).z))

#define NORM_SQUARE_F2(v) ((v).x * (v).x + (v).y * (v).y)
#define NORM_SQUARE_F3(v) ((v).x * (v).x + (v).y * (v).y + (v).z * (v).z)
#define ADD_F2(v,u) {(v).x + (u).x, (v).y + (u).y}


inline double hamming_coef(int n, int N)
{
	return 0.54 - 0.46 * cos(2 * PI_D * (double)n / (N - 1));
}



inline std::string format_cplx(const cuComplex& value)
{
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "Re: %e, Im: %e", value.x, value.y);
    return { buffer };
}


#define PRINT_CPLX(value) format_cplx(value).c_str()




#ifdef _DEBUG
	#include <assert.h>
	#define ASSERT(x) assert(x)
#else
	#define ASSERT(x)
#endif // _DEBUG


#define CUDA_NULL_FREE(PTR)                 \
    do {                                    \
        if(PTR)                             \
        {                                   \
            cudaFree(PTR); (PTR) = nullptr; \
        }                                   \
    } while (0)                             \

// CUDA API error checking
#define CUDA_RETURN_IF_ERROR(err)                                                            \
    do {                                                                                    \
        cudaError_t err_ = (err);                                                           \
        if (err_ != cudaSuccess) {                                                          \
            std::printf("CUDA error %s (%d) '%s'\n At %s:%d\n",                             \
                cudaGetErrorName(err_), err_, cudaGetErrorString(err_), __FILE__, __LINE__);\
            ASSERT(false);																	\
			return false;																	\
        }                                                                                   \
    } while (0)

// cublas API error checking
#define CUBLAS_RETURN_IF_ERR(err)                                                            \
    do {                                                                                    \
        cublasStatus_t err_ = (err);                                                        \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);            \
            ASSERT(false);                                                                  \
			return false;																	\
        }                                                                                   \
    } while (0)

// cufft API error checking
#define CUFFT_RETURN_IF_ERR(err)                                                             \
    do {                                                                                    \
        cufftResult_t err_ = (err);                                                         \
        if (err_ != CUFFT_SUCCESS) {                                                        \
            std::printf("cufft error %d at %s:%d\n", err_, __FILE__, __LINE__);             \
            ASSERT(false);                                                                  \
			return false;																	\
		}                                                                                   \
    } while (0)


#define NPP_RETURN_IF_ERR(err)																\
    do {																					\
        NppStatus err_ = (err);																\
        if(err < 0)																			\
        {																					\
            std::printf("NPP error %d at %s:%d\n", err_, __FILE__, __LINE__);				\
            ASSERT(false);																	\
			return false;																	\
        }																					\
        else if(err > 0)																	\
        {																					\
            std::printf("NPP warning %d at %s:%d\n", err_, __FILE__, __LINE__);				\
        }																					\
    } while(0)																				\


#define TIME_FUNCTION(CALL, MESSAGE)                                                        \
{                                                                                           \
    auto start = std::chrono::high_resolution_clock::now();                                 \
    CALL;                                                                                   \
    CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());                                          \
    auto elapsed = std::chrono::high_resolution_clock::now() - start;                       \
    std::cout << (MESSAGE) << " " <<                                                        \
      std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count() <<         \
        " seconds." << std::endl;                                                           \
}  

#define CUDA_FLOAT_TO_COMPLEX_COPY(SOURCE, DEST, COUNT) \
{                                                                                           \
    CUDA_RETURN_IF_ERROR(cudaMemcpy2D(DEST, 2 * sizeof(float), SOURCE, sizeof(float),       \
                                      sizeof(float), COUNT, cudaMemcpyDefault));            \
}   while (0)                                                                               \


template <typename T> void
sample_data(const T* d_value, T* h_value, uint count = 1)
{
    cudaMemcpy(h_value, d_value, count * sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T> T
sample_value(const T* d_value)
{
	T value;
	cudaMemcpy(&value, d_value, sizeof(T), cudaMemcpyDeviceToHost);
	return value;
}


template <typename T> 
struct PitchedArray
{
	T* data;
	size_t pitch;
	uint3 dims;

	PitchedArray() : data(nullptr), pitch(0), dims({0,0,0}) {} 

	PitchedArray(T* data_in, size_t pitch_in, uint3 dims_in)
		: data(data_in), pitch(pitch_in), dims(dims_in) {}

	~PitchedArray()
	{	
		if (data) {
			cudaFree(data);
			data = nullptr;
		}
	}

	PitchedArray(const PitchedArray&) = delete;
	PitchedArray& operator=(const PitchedArray&) = delete;


	PitchedArray(PitchedArray&& from) noexcept
	{
		data = from.data;
		pitch = from.pitch;
		dims = from.dims;

		from.data = nullptr;
		from.pitch = 0;
		from.dims = {0,0,0};
	}

	
	PitchedArray& operator=(PitchedArray&& from) noexcept
	{
		if (this != &from) {
			if(data) cudaFree(data);

			data = from.data;
			pitch = from.pitch;
			dims = from.dims;

			from.data = nullptr;
			from.pitch = 0;
			from.dims = {0,0,0};
		}
		return *this;
	}


	T* get_row(uint row) const
	{
		assert(row < dims.y);
		return (T*)((uint8_t*)data + row * pitch);
	}

	bool allocate(uint width, uint height, uint depth = 1)
	{
		if (data) {
			std::cerr << "PitchedArray already allocated." << std::endl;
			return false;
		}

		dims = { width, height, depth };
		size_t size = width * height * depth * sizeof(T);
		if (cudaMallocPitch((void**)&data, &pitch, width * sizeof(T), height) != cudaSuccess) {
			std::cerr << "Failed to allocate pitched array of size: " << size << std::endl;
			data = nullptr;
			pitch = 0;
			dims = {0, 0, 0};
			return false;
		}

		return true;
	}
};


namespace types
{
    template <InputDataTypes> struct type_for;

    template <> struct type_for<InputDataTypes::TYPE_I16> { using type = int16_t; };
    template <> struct type_for<InputDataTypes::TYPE_F32> { using type = float; };
    template <> struct type_for<InputDataTypes::TYPE_F32C> { using type = cuComplex; };
    template <> struct type_for<InputDataTypes::TYPE_U8> { using type = uint8_t; };

    template <InputDataTypes T>
    using type_for_t = typename type_for<T>::type;
};

#endif // !DEFS_H

