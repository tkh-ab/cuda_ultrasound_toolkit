#pragma once

#include <cuda_runtime.h>

#include "beamformer_constants.cuh"
#include "kernels/beamformer_kernels.cuh"
#include "../defs.h"

class Beamformer
{
public:
    Beamformer() = default;
    Beamformer(const Beamformer&) = delete;
    Beamformer& operator=(const Beamformer&) = delete;
    Beamformer(Beamformer&&) = delete;
    Beamformer& operator=(Beamformer&&) = delete;
    ~Beamformer() 
    { 
        CUDA_NULL_FREE(_d_beamformer_hadamard);
    }

    bool setup_beamformer(const CudaBeamformerParameters& bp);
    bool beamform(cuComplex* d_input, cuComplex* d_output, const CudaBeamformerParameters& bp);


private:

    float* _d_beamformer_hadamard = nullptr;

    bool _params_to_constants(const CudaBeamformerParameters& bp);
    bool _readi_forces_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume);

	bool _readi_hercules_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume);

	bool _uforces_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume, const short* uforces_elements);

	bool _test_generic_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume);

	bool _test_new_herc_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume);

	bool _test_new_forces_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume);

    bf_kernels::BeamformerConstants _constants;      // Current beamformer constants
};