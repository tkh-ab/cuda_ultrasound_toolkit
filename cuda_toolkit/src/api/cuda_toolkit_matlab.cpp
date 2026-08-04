#include <vector>
#include <chrono>

#include "../defs.h"
#include "../public/cuda_toolkit.hpp"
#include "../public/cuda_toolkit_matlab.h"


void 
beamform_i16( const short* data, CudaBeamformerParameters bp, float* output)
{
    size_t data_size = bp.rf_raw_dim[0] * bp.rf_raw_dim[1] * sizeof(short);
	uint output_size = bp.output_points[0] * bp.output_points[1] * bp.output_points[2] * sizeof(float) * 2; 

    std::span<const u8> input_data(reinterpret_cast<const u8*>(data), data_size);
	std::span<u8> output_data(reinterpret_cast<u8*>(output), output_size);

	if (!cuda_toolkit::beamform(input_data, output_data, bp))
	{
		std::cerr << "Error: Beamforming failed." << std::endl;
	}
}