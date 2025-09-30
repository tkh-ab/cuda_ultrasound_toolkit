#pragma once
#include <map>
#include <cstdlib>
#include <span>
#include <memory>
#include <array>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
 
#include "data_conversion/data_converter.h"
#include "hadamard/hadamard_decoder.h"
#include "rf_ffts/hilbert_handler.h"

#include "../defs.h"


class RfProcessor {
public:

	RfProcessor();
    RfProcessor(const RfProcessor&) = delete;
    RfProcessor& operator=(const RfProcessor&) = delete;
    RfProcessor(RfProcessor&&) = delete;
    RfProcessor& operator=(RfProcessor&&) = delete;
	~RfProcessor() { deinit(); }

    bool init(uint2 rf_raw_dim, uint3 dec_data_dim, ReadiOrdering readi_ordering = ReadiOrdering::HADAMARD);
    bool deinit();

    bool set_channel_mapping(std::span<const int16_t> channel_mapping);
    bool set_match_filter(std::span<const float> match_filter);

	bool convert_decode_strided(void* d_input, cuComplex* d_output, InputDataTypes type);
	bool convert_strided(void* d_input, cuComplex* d_output, InputDataTypes type);
    
    bool hilbert_transform_strided(float* d_input, cuComplex* d_output);

private:

    bool _setup_decode_buffers();
    bool _cleanup_decode_buffers();

    bool _dims_changed(uint2 rf_raw_dim, uint3 dec_data_dim, ReadiOrdering readi_order) const
    {
        return _rf_raw_dim.x != rf_raw_dim.x || 
               _rf_raw_dim.y != rf_raw_dim.y ||
               _dec_data_dim.x != dec_data_dim.x ||
               _dec_data_dim.y != dec_data_dim.y ||
               _dec_data_dim.z != dec_data_dim.z ||
			   _readi_ordering != readi_order;
    };

    size_t _decoded_data_count() { return (size_t)(_dec_data_dim.x) * _dec_data_dim.y * _dec_data_dim.z;}

    bool _init; 

    std::unique_ptr<data_conversion::DataConverter> _data_converter;
    std::unique_ptr<rf_fft::HilbertHandler> _hilbert_handler;
    std::unique_ptr<decoding::HadamardDecoder> _hadamard_decoder;
    uint2 _rf_raw_dim;
    uint3 _dec_data_dim;
    ReadiOrdering _readi_ordering;

    struct 
    {
        float* d_converted;
        float* d_decoded;
    } _decode_buffers;
};