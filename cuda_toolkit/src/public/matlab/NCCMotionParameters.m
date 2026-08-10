classdef NCCMotionParameters
    properties
        patch_size (1, 1) uint32 = uint32(0)
        motion_grid_spacing (1, 1) uint32 = uint32(0)
        motion_grid_dims (1, 2) uint32 = uint32(zeros(1, 2))
        scale_dims (1, 2) uint32 = uint32(zeros(1, 2))
        search_margins (1, 2) uint32 = uint32(zeros(1, 2))
        abs_cor_threshold (1, 1) single = single(0)
        rel_cor_threshold (1, 1) single = single(0)
        min_patch_variance (1, 1) single = single(0)
        use_subpixel (1, 1) logical = false
        reference_frame (1, 1) uint32 = uint32(0)
        neighbour_compare (1, 1) logical = false
        frame_count (1, 1) uint32 = uint32(0)
        image_dims (1, 2) uint32 = uint32(zeros(1, 2))
        data_type (1, 1) InputDataTypes = InputDataTypes.INVALID_TYPE
    end

    properties (Constant, Access = private)
        FieldNames = {'patch_size', 'motion_grid_spacing', 'motion_grid_dims', ...
            'scale_dims', 'search_margins', 'abs_cor_threshold', ...
            'rel_cor_threshold', 'min_patch_variance', 'use_subpixel', ...
            'reference_frame', 'neighbour_compare', 'frame_count', ...
            'image_dims', 'data_type'}
    end

    methods
        function obj = NCCMotionParameters(varargin)
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
                error('NCCMotionParameters:InvalidInput', ...
                    'Constructor arguments must be name-value pairs.');
            end

            validNames = obj.FieldNames;
            for k = 1:2:numel(varargin)
                name = char(varargin{k});
                if ~ismember(name, validNames)
                    error('NCCMotionParameters:InvalidProperty', ...
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
                    error('NCCMotionParameters:InvalidProperty', ...
                        'Unknown property "%s".', name);
                end
                obj.(name) = values.(name);
            end
        end
    end

    methods (Static, Access = private)
        function value = cValue(value)
            if isa(value, 'InputDataTypes')
                value = uint32(value);
            end
        end
    end
end
