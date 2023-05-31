function [matrix] = frft2dI(matrix,angles)
%
% computes 2-D FRFT of given matrix with given angles
%
% IN : matrix: matrix to be transformed
%      angles : angles of FrFT in x and y direction
%

for i = 1:size(matrix,1)
    matrix(:,i) = Disfrft_new(double(matrix(:,i)),angles);
end

for i = 1:size(matrix,2)
    matrix(i,:) = Disfrft_new(double(matrix(i,:)),angles);
end


end
