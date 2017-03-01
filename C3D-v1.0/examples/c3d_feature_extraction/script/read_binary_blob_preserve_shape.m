%
%  Licensed under the Creative Commons Attribution-NonCommercial 3.0 
%  License (the "License"). You may obtain a copy of the License at 
%  https://creativecommons.org/licenses/by-nc/3.0/.
%  Unless required by applicable law or agreed to in writing, software 
%  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
%  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
%  License for the specific language governing permissions and limitations 
%  under the License.
%

function [s, blob, read_status] = read_binary_blob_preserve_shape(filename, precision)
% 
% Read binary blob file from C3D
% INPUT
% filename    : input filename.
% presicion   : data precision e.g. 'double', 'single', 'int32', default is
%               'single', which is used by C3D output features.
% OUTPUT
% s           : a 1x5 matrix indicates the size of the blob 
%               which is [num channel length height width].
% blob        : a 5-D tensor size num x channel x length x height x width
%               containing the blob data.
% read_status : a scalar value = 1 if sucessfully read, 0 otherwise. 

% If only one argument, use 'single' as default precision
if nargin < 2
    precision = 'single';
end

% open file and read size and data buffer
read_status = 1;
f = fopen(filename, 'r');
[s, c] = fread(f, [1 5], 'int32');
if c==5
    m = s(1)*s(2)*s(3)*s(4)*s(5);
    [data, c] = fread(f, [1 m], precision);
    if c~=m
        read_status = 0;
    end
else
    read_status = 0;
end
% close file
fclose(f);

% If failed to read, set empty output and return
if ~read_status
    s = [];
    blob = [];    
    return;
end

% reshape the data buffer to blob
% note that MATLAB use column order, while C3D uses row-order
blob = zeros(s(1), s(2), s(3), s(4), s(5), precision);
off = 0;
image_size = s(4)*s(5);
for n=1:s(1)
    for c=1:s(2)
        for l=1:s(3)
            tmp = data(off+1:off+image_size);
            blob(n,c,l,:,:) = reshape(tmp, [s(5) s(4)])';
            off = off+image_size;
        end
    end
end

end