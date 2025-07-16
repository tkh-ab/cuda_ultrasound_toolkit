#ifndef CUDA_BEAMFORMER_PARAMETERS_H
#define CUDA_BEAMFORMER_PARAMETERS_H

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CHANNEL_COUNT 256

typedef unsigned int uint;

typedef enum TxRxDirection
{
	TX_ROW_RX_ROW = 0,
	TX_ROW_RX_COL = 1,
	TX_COL_RX_ROW = 2,
	TX_COL_RX_COL = 3,
	INVALID = -1,
} TxRxDirection;

typedef enum BeamformPlane
{
	PLANE_XZ = 0,
	PLANE_YZ = 1,
	PLANE_XY = 2,
	PLANE_ARBITRARY = 3,
} BeamformPlane;

typedef enum SequenceId
{
	FORCES = 0,
	UFORCES = 1,
	HERCULES = 2,
	RCA_VLS = 3,
	RCA_TPW = 4,
	UHURCULES = 5,
	RACES = 6,
	EPIC_FORCES = 7,
	EPIC_UFORCES = 8,
	EPIC_UHERCULES = 9,
	FLASH = 10,
	MIXES_S = 100,
} SequenceId;

typedef enum InputDataTypes
{
	INVALID_TYPE = 0,
    TYPE_I16 = 1,
    TYPE_F32 = 2,
    TYPE_F32C = 3,
	TYPE_U8 = 4,
} InputDataTypes;


typedef enum ReadiOrdering
{
    HADAMARD = 0,
    WALSH = 1,
} ReadiOrdering;

typedef struct NccMotionParameters
{
	uint patch_size; 			// Patch size in pixels (assumed square)
	uint motion_grid_spacing;	// [pixels] Spacing between sample patches
	uint motion_grid_dims[2];	    // [rows, cols] Dimensions of the motion grid
	uint search_margins[2];		// [rows, cols] how far outside the patch to search for motion (symmetric)
	float correlation_threshold;// Threshold for the peak to be considered valid relative to the value for no motion.
	float min_patch_variance; 	// Minimum variance of the search patch, if its too flat we won't get a good result
	uint reference_frame;		// Frame to use as the reference for the computation
	uint frame_count;			// Number of frames in the input data
	uint image_dims[2];			// [rows, cols] Dimensions of the input images
	InputDataTypes data_type;   // Image data type, f32 or u8
} NCCMotionParameters;

typedef struct CudaBeamformerParameters
{
	/*
	*	BP Head (Transducer and sequence information)
	*/
	float xdc_transform[16];		// 4x4 Orientation Matrix for the transducer, (column major order)
	float xdc_element_pitch[2];		// [m] Transducer Element Pitch {row, col}

	uint rf_raw_dim[2];		// Raw Data Dimensions [samples * transmits + padding, total_channels (rows + cols)]
	uint dec_data_dim[4];	// Expected dimensions after decoding [samples, rx_channels, transmits]; last element ignored

	bool decode;					// Decode or just reshape data
	TxRxDirection transmit_mode;	// TX and RX directions
	SequenceId das_shader_id;		// Sequence type
	float time_offset;				// pulse length correction time [s]

	/*
	*	BP UI (Beamforming settings)
	*/
	uint output_points[4];	// [X, Y, Z, Frames]
	float output_min_coordinate[4];	// [m] Min XYZ positions, 4th value ignored
	float output_max_coordinate[4];	// [m] Max XYZ positions, 4th value ignored

	float sampling_frequency;		// [Hz] Sampling frequency
	float center_frequency;			// [Hz]  Frequency of the transmit, not the transducer reasonance
	float speed_of_sound;			// [m/s] In the imaged volume

	float off_axis_pos;				// Unused
	BeamformPlane beamform_plane;	// Unused

	float f_number;					// Dynamic receive apodization F# 
	bool interpolate;				// Interpolate between samples during beamforming 
	bool coherency_weighting;		// Apply coherency factor weighting to output data

	/*
	*	Large arrays seperate from the main BP
	*/
	short channel_mapping[256];		// Maps the ordering of the raw channel data to the physical channels
	short sparse_elements[256];		// Channels used for virtual UFORCES elements
	float focal_depths[256];		// [m] Focal Depths for each transmit
	float transmit_angles[256];		// [radians] Transmit Angles for each transmit

	/*
	*	Extra parameters (not part of the standard BP)
	*/
	uint readi_group_count;	// Number of READI groups in the scheme
	uint readi_group_id;	// Which READI group this represents
	ReadiOrdering readi_ordering;	// Ordering of the READI groups

	uint mixes_count;		// Number of mixes crosses
	uint mixes_offset;		// Cross offset at the center of the array
	uint mixes_rows[128];	// Cross row IDs (same for columns)

	uint filter_length;		// Length of the filter
	float rf_filter[1024];			// Time domain kernel of the filter (assumed to be sampled at fs)

	InputDataTypes data_type;		// Type of the raw data being passed in
} CudaBeamformerParameters;

#ifdef __cplusplus
}
#endif
#endif // !CUDA_BEAMFORMER_PARAMETERS_H