#ifndef CUDA_TOOLKIT_H
#define CUDA_TOOLKIT_H

#include <stdint.h>

#define MAX_CHANNEL_COUNT 256

#ifdef _WIN32
	#define EXPORT_FN __declspec(dllexport)
#else
	#define EXPORT_FN
#endif

#ifdef __cplusplus
	extern "C" {
#endif
		typedef unsigned int uint;
		typedef uint16_t u16;
		typedef int16_t i16;

		/**
		* API Usage:
		* In any order:
		*	a. When data dimensions are known call init_cuda_configuration to configure the kernel sizes
		*	b. When OGL input and decoded data buffers are created register with register_cuda_buffers
		* Both functions can be independently called if configuration changes to handle updates
		*	
		* After init call decode_and_hilbert with the byte offset of the input data in the raw buffer
		* and the index of the output buffer
		*/

		/**
		* input_dims: (samples * transmissions + padding) x raw channels
		* output_dims: samples x channels x transmissions
		* channel_mapping: VSX channels to row-column conversion
		*/
		EXPORT_FN bool init_cuda_configuration(const uint* input_dims, const uint* decoded_dims);
		
		/**
		* rf_data_ssbos: List of opengl buffer ids for decoded rf data
		* rf_buffer_count: List length
		* raw_data_ssbo: Opengl buffer id for the raw data buffer
		*/
		EXPORT_FN bool register_cuda_buffers(const uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo);

		/**
		* channel_mapping: Channel mapping array
		*/
		EXPORT_FN bool cuda_set_channel_mapping(const i16 channel_mapping[MAX_CHANNEL_COUNT]);

		/**
		* Generates the spectrum of the time reversed filter waveform and stores it on the GPU
		* If length is 0 filtering is disabled
		* 
		* match_filter: Match filter waveform
		* length: Filter length
		*/
		EXPORT_FN bool cuda_set_match_filter(const float* match_filter, uint length);

		/**
		* input_buffer_idx: Index into rf_data_ssbos for the input buffer
		* output_buffer_idx: Index into rf_data_ssbos for the output buffer
		*/
		EXPORT_FN bool cuda_hilbert(uint input_buffer_idx, uint output_buffer_idx);

		/**
		* input_offset: Offset into raw_data_ssbo 
		* output_buffer_idx: Index into rf_data_ssbos for the output buffer
		*/
		EXPORT_FN bool cuda_decode(size_t input_offset, uint output_buffer_idx);

		/**
		 * Closes all contexts and frees all cuda GPU buffers.
		 */
		EXPORT_FN void deinit_cuda_configuration();

#ifdef __cplusplus
	}
#endif

#endif // !CUDA_TOOLKIT_H

