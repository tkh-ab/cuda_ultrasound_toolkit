classdef CorrImagesParameters
    properties
        template_dims (1, 2) uint32 = uint32(zeros(1, 2))
        source_dims (1, 2) uint32 = uint32(zeros(1, 2))
    end

    properties (Constant, Access = private)
        FieldNames = {'template_dims', 'source_dims'}
    end

    methods
        function obj = CorrImagesParameters(varargin)
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
                s.(name) = obj.(name);
            end
        end
    end

    methods (Access = private)
        function obj = applyNameValues(obj, varargin)
            if mod(numel(varargin), 2) ~= 0
                error('CorrImagesParameters:InvalidInput', ...
                    'Constructor arguments must be name-value pairs.');
            end

            validNames = obj.FieldNames;
            for k = 1:2:numel(varargin)
                name = char(varargin{k});
                if ~ismember(name, validNames)
                    error('CorrImagesParameters:InvalidProperty', ...
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
                    error('CorrImagesParameters:InvalidProperty', ...
                        'Unknown property "%s".', name);
                end
                obj.(name) = values.(name);
            end
        end
    end
end
