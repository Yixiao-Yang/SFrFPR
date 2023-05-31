function [z0, im_r, Relerrs] = PRDenoiser(Y, n1, n2, Masks, pupil, T, newfolder, x, u, eta, sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By Yixiao Yang, Nov 5th, 2019. Contact me: yangyixiao1996@gmail.com.
% This function runs the PRDenoiser algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output: z0: n1 * n2, spectrum initialization (for result comparison);
%         im_r: n1 * n2, recovered HR pluraql image;
%         Relerrs: recovery errors in each iteration.
% Input:  Y: n1_LR * n2_LR * L, captured LR images;
%         n1 and n2 are the pixel numbers of im_r (HR) in two dimensions;
%         sigma2: variance of additive noise;
%         Masks: L * 2 (each point indicates the index of the left-upper point of the LR image in the HR spectrum);
%         pupil: the pupil function;
%         T: the number of iteration;
%         mu_max: the stepsize parameter;
%         weight: the weighting parameter;
%         newfolder: folder for saving results;
%         x: original benchmark HR spectrum for calculating recovery error.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Refference:
% ICASSP 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Initialization
[n1_LR,n2_LR,L] = size(Y);
z0 = (1+1i)*ones(n1, n2);
z = z0;
v_init = zeros(n1, n2, 2);
Relerrs = zeros(T+1,1) ;
Relerrs( 1 ) = sum(sum(abs(abs(z)-abs(x))))/sum(sum(abs(x))) ;
N = zeros(size(Y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iterations
for t = 1 : T
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
      v_init(:,:,1) = real(z);
      v_init(:,:,2) = imag(z);
      v_init = denoise(v_init,sigma, n1, n2,'DnCNN');
      v_hat = v_init(:,:,1) + 1j*v_init(:,:,2);

%     v_r = denoise(real(z),sigma, n1, n2,'DnCNN');
%     v_i = denoise(imag(z),sigma+10, n1, n2,'DnCNN');
%     v_hat = v_r+1i*v_i;
%     v_hat = reshape(v_hat,size(z));
    
    grad = A_Inverse_PnP((abs(A_PnP(z,Masks,n1_LR,n2_LR,pupil)).^2-Y).*A_PnP(z,Masks,n1_LR,n2_LR,pupil),Masks,n1,n2,pupil);
    grad = reshape(grad,size(z));
    z = z - u*(grad+eta*(z-v_hat));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calculate recovery error
    Relerrs( t + 1 ) = sum(sum(abs(abs(z)-abs(x))))/sum(sum(abs(x))) ;
    % save iterative results
    if mod(t,10) == 0
        im_r = z;
        imwrite(uint8(255*abs(im_r)),[newfolder '\im_r_' num2str(t) '.png'],'png');        
        im_r_ang = angle(im_r)/pi;
        im_r_ang = im_r_ang - min(min(im_r_ang));
        im_r_ang = im_r_ang/max(max(im_r_ang));
        imwrite(uint8(255*abs(im_r_ang)),[newfolder '\im_r_ang_' num2str(t) '.png'],'png');

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

end