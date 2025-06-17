# pragma once
#include "communication_params.h"
#include "transfer_server.h"


class TestApp
{
public:
    TestApp();
    ~TestApp();

    void run();
    void cleanup();

private:

    bool _handle_beamform_command(const CommandPipeMessage& command);
	bool _handle_motion_detection_command(const CommandPipeMessage& command);
    void _message_loop();

    TransferServer* _transfer_server;

};
