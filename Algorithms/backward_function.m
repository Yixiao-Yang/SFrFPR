function [ Z ] = backward_function( Y,a, dims)
%IFRFT2DMASK Summary of this function goes here
%   Detailed explanation goes here
n1 = dims(1);
n2 = dims(2);
L = length(a);
Y = reshape(Y,[n1,n2,L]);
a = -a;
for k = 1:L
    Y(:,:,k) = frft2dI(Y(:,:,k),a(k));
    Z = mean(Y,3);
end
Z = Z(:);
end

