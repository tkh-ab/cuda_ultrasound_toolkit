#include <numeric>
#include <algorithm>

#include "hadamard_decoder.h"

#define HADAMARD_MAX_SIZE 1024 // Maximum size for Hadamard matrix

constexpr float hadamard_12_transpose[] = {
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1,
 1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,
 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1,
 1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1,
 1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1,
 1, 1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1,
 1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1,-1,
 1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1,
 1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1,
 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1,
 1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1,
};


constexpr float hadamard_20_transpose[] = {
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,
 1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,
 1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1,
 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,
 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,
 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,
 1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,
 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,
 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,
 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,
 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,
 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1,
 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,
 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,
 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1,
 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,
 1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,
 1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,
};



bool 
decoding::HadamardDecoder::decode(float* d_input, float* d_output, uint3 decoded_dims)
{
    if (!_d_hadamard || _hadamard_size == 0)
    {
        std::cerr << "Hadamard matrix not generated. Call generate_hadamard() first." << std::endl;
        return false; // Hadamard matrix not generated
    }

    if( decoded_dims.z != _hadamard_size)
    {
        std::cerr << "Transmit count (" << decoded_dims.z << ") does not match Hadamard size (" << _hadamard_size << ")." << std::endl;
        return false; // Mismatch in dimensions
    }
	uint tx_size = decoded_dims.x * decoded_dims.y;

	float alpha = 1/((float)decoded_dims.z); // Counters the N scaling of hadamard decoding 
	float beta = 0.0f;

	CUBLAS_RETURN_IF_ERR(cublasSgemm(
		_cublas_handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		tx_size, decoded_dims.z, decoded_dims.z,
		&alpha, d_input, tx_size,
		_d_hadamard, decoded_dims.z,
		&beta, d_output, tx_size));

	return true;
}

bool 
decoding::HadamardDecoder::set_hadamard(uint row_count, EncodingMatrix readi_ordering)
{
	if( !_cublas_handle )
	{
		if( !_create_cublas_handle() )
		{
			std::cerr << "Failed to create cuBLAS handle." << std::endl;
			return false; // Failed to create cuBLAS handle
		}
	}

	if (row_count == 0 || row_count > HADAMARD_MAX_SIZE) // Arbitrary limit for size
    {
        std::cerr << "Maximum hadamard size (" << HADAMARD_MAX_SIZE << ") exceeded. " << row_count << " requested." << std::endl;
        return false; 
    }
    if( _d_hadamard )
    {
        _cleanup_hadamard(); // Clean up existing Hadamard matrix if it exists
    }

	_readi_ordering = readi_ordering;
    _hadamard_size = row_count;

	size_t hadamard_mem_size = (size_t)row_count * row_count * sizeof(float);
	CUDA_RETURN_IF_ERROR(cudaMalloc(&_d_hadamard, hadamard_mem_size));
	return generate_hadamard(_d_hadamard, row_count, readi_ordering);
}

bool 
decoding::HadamardDecoder::generate_hadamard(float* d_hadamard, uint row_count, EncodingMatrix readi_ordering)
{

    // Create a requested_length x requested_length array on the CPU
	size_t element_count = (size_t)row_count * row_count;
	float* cpu_hadamard = (float*)std::calloc(element_count, sizeof(float));
	if (!cpu_hadamard)
	{
		std::cerr << "Failed to init hadamard matrix of size " << row_count << std::endl;
	}

	uint base_length; // Base case length for Hadamard matrix
	uint iterations;

	// Check if the requested length is valid and set up the base case.
    // The base case will be placed in the top left corner of the matrix.
	if (ISPOWEROF2(row_count))
	{
		base_length = 1;
		iterations = (uint)log2(row_count);
		cpu_hadamard[0] = 1;
	}
	else if( row_count % 12 == 0 && ISPOWEROF2(row_count / 12))
	{
		base_length = 12;
		iterations = (uint)log2(row_count / 12);
		for (uint i = 0; i < base_length; i++)
		{
			int source_offset = i * base_length;
			int dest_offset = i * row_count;
			std::memcpy(cpu_hadamard + dest_offset, hadamard_12_transpose + source_offset, base_length * sizeof(float));
		}
	}
	else if (row_count % 20 == 0 && ISPOWEROF2(row_count / 20))
	{
		base_length = 20;
		iterations = (uint)log2(row_count / 20);
		for (uint i = 0; i < base_length; i++)
		{
			int source_offset = i * base_length;
			int dest_offset = i * row_count;
			std::memcpy(cpu_hadamard + dest_offset, hadamard_20_transpose + source_offset, base_length * sizeof(float));
		}
	}
	else
	{
		std::cerr << "Hadamard: Size '" << row_count << "' not supported" << std::endl;
		std::cerr << "System only supports sizes of 2^n, 12 * 2^n or 20 * 2^n" << std::endl;
		return false;
	}
		
	// Recursively take the kroneker product of the matrix with H2 until we reach the desired size
	for (uint current_length = base_length; current_length < row_count; current_length *= 2)
	{
		int right_half_offset = current_length; // Offset to top right half
		int bottom_half_offset = current_length * row_count; // Offset to bottom left half
		for (uint row_idx = 0; row_idx < current_length; row_idx++)
		{
			int row_offset = row_idx * row_count;
			// Quadrants 1, 2, and 3 are copies of the base case, quadrant 4 is the negative of the base case
			for (uint col_idx = 0; col_idx < current_length; col_idx++)
			{
				cpu_hadamard[col_idx + row_offset + right_half_offset] = cpu_hadamard[col_idx + row_offset];
				cpu_hadamard[col_idx + row_offset + bottom_half_offset] = cpu_hadamard[col_idx + row_offset];
				cpu_hadamard[col_idx + row_offset + bottom_half_offset + right_half_offset] = -1 * cpu_hadamard[col_idx + row_offset];
			}
		}
	}

	if( readi_ordering == EncodingMatrix::WALSH )
	{
		_sort_walsh(cpu_hadamard, row_count);
	}

	CUDA_RETURN_IF_ERROR(cudaMemcpy(d_hadamard, cpu_hadamard, element_count * sizeof(float), cudaMemcpyHostToDevice));
	free(cpu_hadamard);

	return true;
}

void
decoding::HadamardDecoder::_sort_walsh(float* hadamard, uint row_count)
{

	std::vector<uint> zero_crossings(row_count);
	std::vector<float> sorted_hadamard(row_count * row_count);

	for(uint i = 0; i < row_count; i++)
	{
		for (uint j = 0; j < row_count - 1; j++)
		{
			zero_crossings[i] += (hadamard[i * row_count + j] * hadamard[i * row_count + j + 1]) < 0 ? 1 : 0;
		}
	}

	// Sort the rows based on the number of zero crossings
	std::vector<uint> sorted_indices(row_count);
	std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
	std::sort(sorted_indices.begin(), sorted_indices.end(), [&](uint a, uint b) {
		return zero_crossings[a] < zero_crossings[b];
	});

	for (uint i = 0; i < row_count; i++)
	{
		for (uint j = 0; j < row_count; j++)
		{
			sorted_hadamard[i * row_count + j] = hadamard[sorted_indices[i] * row_count + j];
		}
	}

	std::memcpy(hadamard, sorted_hadamard.data(), row_count * row_count * sizeof(float));
}