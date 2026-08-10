classdef CudaBeamformerParameters
    properties (Constant)
        MAX_ACQ_COUNT = uint32(256)
        MAX_CHANNEL_COUNT = uint32(256)
        MAX_FILTER_LENGTH = uint32(1024)
    end

    properties
        xdc_transform (1, 16) single = single(reshape(eye(4), 1, 16))
        xdc_element_pitch (1, 2) single = single(zeros(1, 2))

        rf_raw_dim (1, 4) uint32 = uint32(zeros(1, 4))
        dec_data_dim (1, 4) uint32 = uint32(zeros(1, 4))

        decode (1, 1) EncodingMatrix = EncodingMatrix.NONE
        das_shader_id (1, 1) SequenceId = SequenceId.FORCES
        time_offset (1, 1) single = single(0)

        output_points (1, 4) uint32 = uint32(zeros(1, 4))
        output_min_coordinate (1, 4) single = single(zeros(1, 4))
        output_max_coordinate (1, 4) single = single(zeros(1, 4))

        sampling_frequency (1, 1) single = single(0)
        center_frequency (1, 1) single = single(0)
        speed_of_sound (1, 1) single = single(0)

        off_axis_pos (1, 1) single = single(0)
        beamform_plane (1, 1) BeamformPlane = BeamformPlane.PLANE_XZ

        fn_tx (1, 1) single = single(0)
        fn_rx (1, 1) single = single(0)
        interpolate (1, 1) logical = false
        coherency_weighting (1, 1) single = single(0)

        channel_mapping (1, 256) int16 = int16(zeros(1, 256))
        sparse_elements (1, 256) int16 = int16(zeros(1, 256))
        foci (1, 768) single = single(zeros(1, 256 * 3))

        tx_orientation (1, 1) RCAOrientation = RCAOrientation.ORIENT_NONE
        rx_orientation (1, 1) RCAOrientation = RCAOrientation.ORIENT_NONE

        readi_group_count (1, 1) uint32 = uint32(0)
        readi_group_id (1, 1) uint32 = uint32(0)

        mixes_count (1, 1) uint32 = uint32(0)
        mixes_offset (1, 1) uint32 = uint32(0)
        mixes_rows (1, 128) uint32 = uint32(zeros(1, 128))

        filter_length (1, 1) uint32 = uint32(0)
        rf_filter (1, 1024) single = single(zeros(1, 1024))

        data_type (1, 1) InputDataTypes = InputDataTypes.INVALID_TYPE
        apo_type (1, 1) ApoType = ApoType.RX_HANN
        to_power (1, 1) int32 = int32(0)

        sample_type (1, 1) SampleType = SampleType.SAMPLE_NORMAL
    end

    properties (Constant, Access = private)
        FieldNames = {'xdc_transform', 'xdc_element_pitch', 'rf_raw_dim', ...
            'dec_data_dim', 'decode', 'das_shader_id', 'time_offset', ...
            'output_points', 'output_min_coordinate', 'output_max_coordinate', ...
            'sampling_frequency', 'center_frequency', 'speed_of_sound', ...
            'off_axis_pos', 'beamform_plane', 'fn_tx', 'fn_rx', ...
            'interpolate', 'coherency_weighting', 'channel_mapping', ...
            'sparse_elements', 'foci', 'tx_orientation', 'rx_orientation', ...
            'readi_group_count', 'readi_group_id', 'mixes_count', ...
            'mixes_offset', 'mixes_rows', 'filter_length', 'rf_filter', ...
            'data_type', 'apo_type', 'to_power', 'sample_type'}
    end

    methods
        function obj = CudaBeamformerParameters(varargin)
            if nargin == 1 && isstruct(varargin{1})
                obj = obj.applyStruct(varargin{1});
            else
                obj = obj.applyNameValues(varargin{:});
            end
        end

        function s = toStruct(obj)
            s = struct();
            for k = 1:numel(obj.FieldNames)
                name = obj.FieldNames{k};
                s.(name) = obj.cValue(obj.(name));
            end
        end
    end

    methods (Access = private)
        function obj = applyNameValues(obj, varargin)
            if mod(numel(varargin), 2) ~= 0
                error('CudaBeamformerParameters:InvalidInput', ...
                    'Constructor arguments must be name-value pairs.');
            end

            validNames = obj.FieldNames;
            for k = 1:2:numel(varargin)
                name = char(varargin{k});
                if ~ismember(name, validNames)
                    error('CudaBeamformerParameters:InvalidProperty', ...
                        'Unknown property "%s".', name);
                end
                obj.(name) = varargin{k + 1};
            end
        end

        function obj = applyStruct(obj, values)
            names = fieldnames(values);
            for k = 1:numel(names)
                name = names{k};
                if ~ismember(name, obj.FieldNames)
                    error('CudaBeamformerParameters:InvalidProperty', ...
                        'Unknown property "%s".', name);
                end
                obj.(name) = values.(name);
            end
        end
    end

    methods (Static, Access = private)
        function value = cValue(value)
            if isa(value, 'EncodingMatrix') || isa(value, 'SequenceId') || ...
                    isa(value, 'BeamformPlane') || isa(value, 'RCAOrientation') || ...
                    isa(value, 'InputDataTypes') || isa(value, 'ApoType') || ...
                    isa(value, 'SampleType')
                value = uint32(value);
            end
        end
    end
end
