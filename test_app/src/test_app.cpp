#include <vector>
#include <chrono>

#include <cuda_toolkit.hpp>
#include "test_app.h"

TestApp::TestApp()
{
    _transfer_server = NULL;
    try
	{
		_transfer_server = new TransferServer(COMMAND_PIPE_NAME, DATA_SMEM_NAME, PARAMETERS_SMEM_NAME, DATA_SMEM_SIZE);
	}
	catch(const std::runtime_error& e)
	{
		std::cerr << e.what() << '\n';
		throw e;
	}
}

TestApp::~TestApp()
{
    cleanup();
}

void 
TestApp::cleanup()
{
    if (_transfer_server)
    {
        delete _transfer_server;
		_transfer_server = nullptr;
    }
}

void 
TestApp::run()
{
    _message_loop();
}

void
TestApp::_message_loop()
{
    int commands_received = 0;
	bool exit = false;
	while(!exit)
	{
		std::cout << "Waiting for command..." << std::endl;
		auto command_opt = _transfer_server->wait_for_command();
		if(!command_opt.has_value())
		{
			std::cerr << "Error waiting for command." << std::endl;
			break;
		}
		commands_received++;

		CommandPipeMessage command = command_opt.value();
		switch(command.opcode)
		{
			case CudaCommand::BEAMFORM_VOLUME:
				std::cout << "Beamform volume command received." << std::endl;
				if(!_handle_beamform_command(command))
					throw std::runtime_error("Error handling beamform command");
				break;

			case CudaCommand::SVD_FILTER:
				std::cout << "SVD filter command." << std::endl;
				break;

			case CudaCommand::NCC_MOTION_DETECT:
				std::cout << "NCC motion detect command." << std::endl;
				if(!_handle_motion_detection_command(command))
					throw std::runtime_error("Error handling motion detection command");
				break;

			case CudaCommand::ACK:
				std::cout << "Acknowledged command." << std::endl;
				break;

			case CudaCommand::ERR:
				std::cerr << "Error command received." << std::endl;
			    break;

			default:
				std::cerr << "Unknown command opcode (" << command.opcode << ") received." << std::endl;
		}

		std::cout << "Command handled, total processed: " << commands_received << std::endl << std::endl;
		//exit = true;
	}
}


bool
TestApp::_handle_beamform_command(const CommandPipeMessage& command)
{
    const CudaBeamformerParameters* bp = &((_transfer_server->get_parameters_smem())->beamformerParameters);

	if (!bp)
	{
		std::cerr << "Error: Beamformer parameters not set." << std::endl;
		return false;
	}

	size_t data_size = command.data_size;
	auto data_buffer = _transfer_server->get_data_smem();
	if (data_buffer.empty())
	{
		std::cerr << "Error: Data buffer not set." << std::endl;
        _transfer_server->respond_error();
		return false;
	}

    _transfer_server->respond_ack();
    // Beamform the data

    uint output_size = bp->output_points[0] * bp->output_points[1] * bp->output_points[2] * sizeof(float) * 2; // Assuming output is float2
    if (output_size > _transfer_server->get_data_smem().size())
    {
        std::cerr << "Error: Output size exceeds shared memory size." << std::endl;
        _transfer_server->respond_error();
        return false;
    }

    u8* output_data = new u8[output_size];

	if (!cuda_toolkit::beamform(
			std::span<const u8>(data_buffer.data(), data_size),
			std::span<u8>(output_data, output_size),
			*bp))
	{
		std::cerr << "Error: Beamforming failed." << std::endl;
		delete[] output_data;
		_transfer_server->respond_error();
		return false;
	}

    _transfer_server->write_output_data(std::span<const u8>(output_data, output_size));
    if (!_transfer_server->respond_success(output_size))
    {
        std::cerr << "Error: Failed to respond with success." << std::endl;
        return false;
    }

	delete[] output_data;
	return true;
}

bool
TestApp::_handle_motion_detection_command(const CommandPipeMessage& command)
{
	const NccMotionParameters* params = &((_transfer_server->get_parameters_smem())->nccMotionParameters);

	if (!params)
	{
		std::cerr << "Error: NCC parameters not set." << std::endl;
		return false;
	}

	size_t data_size = command.data_size;
	auto data_buffer = _transfer_server->get_data_smem();
	if (data_buffer.empty())
	{
		std::cerr << "Error: Data buffer not set." << std::endl;
		_transfer_server->respond_error();
		return false;
	}

	_transfer_server->respond_ack();
	// Process motion detection

	uint output_size = params->motion_grid_spacing * params->motion_grid_spacing * sizeof(int) * 2;
	if (output_size > _transfer_server->get_data_smem().size())
	{
		std::cerr << "Error: Output size exceeds shared memory size." << std::endl;
		_transfer_server->respond_error();
		return false;
	}

	u8* output_data = new u8[output_size];

	bool result = cuda_toolkit::motion_detection(
		std::span<const u8>(data_buffer.data(), data_size),
		std::span<u8>(output_data, output_size), // Placeholder for motion maps
		*params);
	
	_transfer_server->write_output_data(std::span<const u8>(output_data, output_size));
	if (!_transfer_server->respond_success(output_size))
	{
		std::cerr << "Error: Failed to respond with success." << std::endl;
		return false;
	}

	delete[] output_data;
	return true;
}
