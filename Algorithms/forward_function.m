function [ Y ] = forward_function( x,a,dims)
%FRFT2DMASK Summary of this function goes here
%   Detailed explanation goes here
X = reshape(x, dims);   % image comes in as a vector.  Reshape to rectangle
L = length(a);
[m,n] = size(X);
Y = zeros(m,n,L);
for k = 1:L
    Y(:,:,k) = frft2dI(X,a(k));
end
Y = Y(:);
end

