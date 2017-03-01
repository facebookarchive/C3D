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

function [s, data] = read_binary_blob(fn)

f = fopen(fn, 'r');
s = fread(f, [1 5], 'int32');

% s contains size of the blob e.g. num x chanel x length x height x width
m = s(1)*s(2)*s(3)*s(4)*s(5);

% data is the blob binary data in single precision (e.g float in C++)
data = fread(f, [1 m], 'single');
fclose(f);

end
