#ifndef PARAMETER_DEFS_H
#define PARAMETER_DEFS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_beamformer_parameters.h"

#define COMMAND_PIPE_NAME "\\\\.\\pipe\\cuda_command"
#define DATA_SMEM_NAME "Local\\cuda_data"
#define PARAMETERS_SMEM_NAME "Local\\cuda_parameters"

#define MEGABYTE (1024U * 1024U)
#define GIGABYTE (1024U * 1024U * 1024U)

#define COMMAND_PIPE_SIZE MEGABYTE
#define DATA_SMEM_SIZE 0x80000000UL // 2GB

/*
* What test function we're calling from MATLAB
*/
typedef enum CudaCommand
{
	ERR = 0,
	ACK = 1,
	SUCCESS = 2,
	BEAMFORM_VOLUME = 3,
	SVD_FILTER = 4,
	NCC_MOTION_DETECT = 5,
} CudaCommand;

typedef struct CommandPipeMessage
{
	CudaCommand opcode;
	unsigned long long data_size;
	int frame_count;
} CommandPipeMessage;

typedef struct SVDParameters
{
	int frame_count;
	int filter_indicies[256];	// Singular values to remove
} SVDParameters;

typedef struct SharedMemoryParams
{
	CudaBeamformerParameters beamformerParameters;
	SVDParameters svdParameters;
	NCCMotionParameters nccMotionParameters;
} SharedMemoryParams;


#ifdef __cplusplus
}   // extern "C"  
#endif
#endif // !PARAMETER_DEFS_H

