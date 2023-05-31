%% Set the path
clear;clc;
addpath(genpath('..'))

%% Parameters
SamplingRate = 1;
T = 200;

%% read images
ext         =  {'*.png'};
filePaths   =  [];
folderTest  = '.\data\Set12';
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%% reconstructing
for priors = 1   % choose signal prior, 0:only real-valued, 1:tv, 2:wnnm, 3:bm3d, 4:dncnn
    for p = 0.5   % choose fractional orders
        PSNR_all = zeros(1,length(filePaths));
        SSIM_all = zeros(1,length(filePaths));
        Time_all = zeros(1,length(filePaths));
        for j = 1:12
            x_0 = double(imread(fullfile(folderTest,filePaths(j).name)));
            x_0 = imresize(x_0,[128,128]);
%             x_0 = shift(x_0,128);
%             x_0 = flip(flip(x_0,1),2);

            [height, width]=size(x_0);
            n = length(x_0(:));
            m = round(n*SamplingRate);%May be overwritten when using Fourier measurements

            M = @(x) forward_function(x(:), p, size(x_0));
            Mt = @(y) backward_function(y(:), p, size(x_0));
            
            z = M(x_0);
            y = abs(z);

            x_init = ones(height,width);
%             x_init = flip(flip(x_0,1),2) + 100*randn(128,128);
%             x_init = shift(x_0,32) + 100*randn(128,128);
            Pm=@(x) Mt(y.*exp(1i*angle(M(x))));
            err_norm = zeros(1,T+1);

            t0 = tic;
    %         priors = 0;	% choose signal prior, 0:only real-valued, 1:tv, 2:wnnm, 3:bm3d, 4:dncnn
            x_hat = x_init;
            err_norm(1) = norm(x_hat/255-x_0/255);
            for i=1:T
                % project on data-fidelity sets via AP
                v_hat = Pm(x_hat);
                % project on signal prior sets
                switch priors
                    case 0
                        denoiser = 'real'; % signal prior is real-valued
                        x_hat = reshape(real(v_hat), size(x_0));
                    case 1
                        denoiser = 'tv'; % signal prior is tv
                        Nbiter= 5;	% number of iterations
                        lambda = 0.1; 	% regularization parameter
                        tau = 0.01;		% proximal parameter >0; influences the
                        v_hat = reshape(real(v_hat), size(x_0));
                        x_hat = 255*TVdenoising(v_hat/255, lambda, tau, Nbiter);
                        x_hat = reshape(x_hat, size(x_0));
                    case 2
                        denoiser = 'wnnm'; % signal prior is wnnm
                        sigma = 5; % denoising strength
                        Par   = ParSet(sigma);   
                        v_hat = reshape(real(v_hat), size(x_0));
                        x_hat = WNNM_DeNoising(v_hat, x_0, Par );     
                        x_hat = reshape(x_hat, size(x_0));
                     case 3
                        denoiser = 'BM3D'; % signal prior is bm3d
                        sigma = 5; % denoising strength
                        v_hat = reshape(real(v_hat), size(x_0));
                        x_hat = denoise(v_hat, sigma, height, width, denoiser);
                        x_hat = reshape(x_hat, size(x_0));
                    case 4
                        denoiser = 'DnCNN'; % signal prior is bm3d
                        n_DnCNN_layers = 17;%Other option is 17
                        LoadNetworkWeights(n_DnCNN_layers);
                        sigma = 5;
                        v_hat = reshape(real(v_hat), size(x_0));
                        x_hat = denoise(v_hat, sigma, height, width, denoiser);
                        x_hat = reshape(x_hat, size(x_0));
                end
                err_norm(i+1) = norm(x_hat/255-x_0/255);

            end
            x_hat_final = reshape(x_hat, size(x_0));
            t_ours = toc(t0);

            PSNR_all(j) = psnr(x_0/255, x_hat_final/255);
            SSIM_all(j) = ssim(x_0, x_hat_final);
            Time_all(j) = t_ours;
%             plot(1:10:201,err_norm(1:10:201),'linewidth',1.5);
%             xlabel('k-th iteration');
%             ylabel('|| x_{k}-x^{*}||_{F}');
%             hold on
            display([num2str(SamplingRate*100),'% Sampling gap: PSNR=',num2str(PSNR_all(j)),' Time=',num2str(Time_all(j))])
            Output_path        =    ['./Results/Set12/Alpha_',num2str(p),'/NNGAP_',denoiser,'/'];
%             Output_path        =    ['./Results/Set12/Alpha_',num2str(p),'/WF_real''/'];
            if ~exist(Output_path, 'dir')
                mkdir(Output_path);
            end
%             imwrite( x_hat_final/255, [Output_path, filePaths(j).name, num2str(PSNR_all(j)), '.png'] );

        end

        fprintf('AVG PSNR: %.2f\n', mean(PSNR_all),mean(SSIM_all))
%         dlmwrite([Output_path,'averageresults.txt'], [mean(PSNR_all),mean(Time_all)]);

    end
end
