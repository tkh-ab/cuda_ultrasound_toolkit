#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#include "cuda_transfer.h"

#define RESPONSE_POLL_PERIOD 100 // 100 ms
#define RESPONSE_POLL_TIMEOUT 1200000 // 20 minutes

#ifdef MATLAB_CONSOLE  // Define this in the makefile
#define mexErrMsgIdAndTxt  mexErrMsgIdAndTxt_800
#define mexWarnMsgIdAndTxt mexWarnMsgIdAndTxt_800
void mexErrMsgIdAndTxt(const char*, const char*, ...);
void mexWarnMsgIdAndTxt(const char*, const char*, ...);
#define error_tag "matlab_transfer:error"
#define error_msg(...)   mexErrMsgIdAndTxt(error_tag, __VA_ARGS__)
#define warning_msg(...) mexWarnMsgIdAndTxt(error_tag, __VA_ARGS__)
#else
#define error_msg(...)   {fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n");}
#define warning_msg(...) {fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n");}
#endif

static SharedMemoryParams* g_params = NULL;
static void* g_data = NULL;
static HANDLE g_command_pipe = INVALID_HANDLE_VALUE;

static inline const char* format_error_message(DWORD code) {
    static char message[512];
    DWORD size = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        message,
        sizeof(message),
        NULL
    );

    if (size <= 0) {
        snprintf(message, sizeof(message), "Unknown error code: %lu", (unsigned long)code);
    }
    else {
        // Remove trailing newline if present
        size_t len = strlen(message);
        if (len > 0 && message[len - 1] == '\n') {
            message[len - 1] = '\0';
        }
    }

    return message;
}

static int64_t get_current_time_us()
{
    static LARGE_INTEGER frequency = {0};
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);

    return (int64_t)((now.QuadPart * 1000000) / frequency.QuadPart);
}

void
_cleanup_shared_resources()
{
    if (g_data)
    {
        UnmapViewOfFile(g_data);
        g_data = NULL;
    }

    if (g_params)
    {
        UnmapViewOfFile(g_params);
        g_params = NULL;
    }

    if (g_command_pipe != INVALID_HANDLE_VALUE)
    {
        CloseHandle(g_command_pipe);
        g_command_pipe = INVALID_HANDLE_VALUE;
    }
}

HANDLE 
open_command_pipe()
{
    HANDLE h = CreateFileA(COMMAND_PIPE_NAME, GENERIC_READ | GENERIC_WRITE,
                                 0, NULL, OPEN_EXISTING, 0, NULL);

    if (h == INVALID_HANDLE_VALUE)
    {
        DWORD error = GetLastError();
        warning_msg("Failed to open command pipe with error: %lu, '%s'", 
                    error, format_error_message(error));
    }
    return h;
}

bool
open_smem(char* name, void** smem, size_t size)
{
    HANDLE h = OpenFileMappingA(FILE_MAP_ALL_ACCESS, 0, name);

    if (h == NULL || h == INVALID_HANDLE_VALUE)
    {
        DWORD error = GetLastError();
        warning_msg("Failed to create data shared memory '%s' with error: %lu, '%s'", 
                    name, error, format_error_message(error));
        return false;
    }

    *smem = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);

    if (*smem == NULL)
    {
        DWORD error = GetLastError();

        const char* format_error = format_error_message(error);
        warning_msg("Failed to map data shared memory '%s' with error: %lu, '%s'", 
                    name, error, format_error);
        CloseHandle(h);
        return false;
    }

    CloseHandle(h);
    return true;
}

bool setup_shared_resources()
{
    if(!g_data)
    {
        if (!open_smem(DATA_SMEM_NAME, &g_data, DATA_SMEM_SIZE))
        {

            return false;
        }
    }

    if(!g_params)
    {
        if (!open_smem(PARAMETERS_SMEM_NAME, (void**)&g_params, sizeof(SharedMemoryParams)))
        {
            return false;
        }
    }
    
    if(g_command_pipe == INVALID_HANDLE_VALUE)
    {
        g_command_pipe = open_command_pipe();
        if (g_command_pipe == INVALID_HANDLE_VALUE)
        {
            return false;
        }
    }

    return true;
}

bool send_command(CudaCommand command, size_t data_size)
{
    CommandPipeMessage message;
    message.opcode = command;
    message.data_size = data_size;
    message.frame_count = 0;

    DWORD bytes_written = 0;
    if (!WriteFile(g_command_pipe, &message, sizeof(message), &bytes_written, NULL))
    {
        DWORD error = GetLastError();
        warning_msg("Failed to write to command pipe with error: %lu, '%s'", error, format_error_message(error));
        return false;
    }

    return true;
}

bool wait_for_response(CommandPipeMessage* message)
{
    DWORD bytes_read = 0;
    uint32_t elapsed = 0;
    while(elapsed < RESPONSE_POLL_TIMEOUT)
    {
        if (!ReadFile(g_command_pipe, message, sizeof(*message), &bytes_read, NULL))
        {
            DWORD error = GetLastError();
            warning_msg("Failed to read from command pipe with error: %lu", error);
            return false;
        }

        if (bytes_read == 0)
        {
            Sleep(RESPONSE_POLL_PERIOD);
            elapsed += RESPONSE_POLL_PERIOD;
            continue;
        }

        if (bytes_read != sizeof(*message))
        {
            warning_msg("Invalid message size read from command pipe: %lu", bytes_read);
            return false;
        }
        else
        {
            return true;
        }
    }

    warning_msg("Timed out waiting for response from command pipe after %d ms", RESPONSE_POLL_TIMEOUT);
    return false;
}

bool wait_for_ack()
{
    CommandPipeMessage message;
    if (!wait_for_response(&message))
    {
        warning_msg("Failed to receive response from command pipe");
        return false;
    }

    if (message.opcode != ACK)
    {
        warning_msg("Invalid command received from command pipe: %d", message.opcode);
        return false;
    }

    return true;
}

CommandPipeMessage 
wait_for_result()
{
    CommandPipeMessage message;
    if (!wait_for_response(&message))
    {
        warning_msg("Failed to receive response from command pipe");
        return message;
    }
    return message;
}

static void
_motion_detect(const void* images, size_t data_size,
			  const NCCMotionParameters* params, int* motion_maps)
{
	if (data_size > DATA_SMEM_SIZE)
	{
		error_msg("Data size too large: %zu, Max: %lu", data_size, DATA_SMEM_SIZE);
		return;
	}

	if(!setup_shared_resources())
	{
		error_msg("Failed to setup shared resources");
		return;
	}

	// Load the motion detection parameters
	memcpy(&(g_params->nccMotionParameters), params, sizeof(NCCMotionParameters)); 

	// Copy the raw data to shared memory
	memcpy(g_data, images, data_size);

	if (!send_command(NCC_MOTION_DETECT, data_size))
	{
		_cleanup_shared_resources();
		error_msg("Failed to send command to CUDA server");
		return;
	}

	if (!wait_for_ack())
	{
		_cleanup_shared_resources();
		error_msg("Failed to receive ack from CUDA server");
		return;
	}

	CommandPipeMessage result = wait_for_result();
	if (result.opcode == ERR)
	{
		error_msg("Failed to receive valid result from CUDA server");
		_cleanup_shared_resources();
		return;
	}

	size_t output_size = result.data_size;
	memcpy(motion_maps, g_data, output_size);
}

static void 
_beamform(const void* data, size_t data_size,
              const CudaBeamformerParameters* bp, float* output)
{
    if (data_size > DATA_SMEM_SIZE)
    {
        error_msg("Data size too large: %zu, Max: %lu", data_size, DATA_SMEM_SIZE);
        return;
    }

    if(!setup_shared_resources())
    {
        error_msg("Failed to setup shared resources");
        return;
    }

    // Load the beamformer parameters
    memcpy(g_params, bp, sizeof(CudaBeamformerParameters)); 

    // Copy the raw data to shared memory
    memcpy(g_data, data, data_size);

    if (!send_command(BEAMFORM_VOLUME, data_size))
    {
        _cleanup_shared_resources();
        error_msg("Failed to send command to CUDA server");
        return;
    }

    if (!wait_for_ack())
    {
        _cleanup_shared_resources();
        error_msg("Failed to receive ack from CUDA server");
        return;
    }

    CommandPipeMessage result = wait_for_result();
    if (result.opcode == ERR)
    {
        error_msg("Failed to receive valid result from CUDA server");
        _cleanup_shared_resources();
        return;
    }

    size_t output_size = result.data_size;
    memcpy(output, g_data, output_size);
}

void 
beamform_i16( const short* data, CudaBeamformerParameters bp, float* output)
{
    size_t data_size = bp.rf_raw_dim[0] * bp.rf_raw_dim[1] * sizeof(short);
    _beamform((void*)data, data_size, &bp, output);
}

void
beamform_f32(const float* data, CudaBeamformerParameters bp, float* output)
{
    size_t data_size = bp.rf_raw_dim[0] * bp.rf_raw_dim[1] * sizeof(float);
    _beamform((void*)data, data_size, &bp, output);
}

void
motion_detect_f32(const float* images, NCCMotionParameters params, int* motion_maps)
{
	size_t data_size = params.image_dims[0] * params.image_dims[1] * params.frame_count * sizeof(float);
	_motion_detect((void*)images, data_size, &params, motion_maps);
}

void
deinit()
{
    _cleanup_shared_resources();
}

BOOL APIENTRY 
DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        
        break;
    case DLL_PROCESS_DETACH:
        _cleanup_shared_resources();
        break;
    }
    return TRUE;
}


