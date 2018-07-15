% The script demonstrates how to aggregate several R-CSPs to achieve good
% performance. You need to customize the code accordingly and write some
% functions yourself. 
%
% %[Author Notes]%
% Author: Haiping LU
% Email : hplu@ieee.org   or   eehplu@gmail.com
% Release date: March 20, 2012 (Version 1.0)
% Please email me if you have any problem, question or suggestion
%
% %[Algorithm]%:
% This script demonstrates how to aggregate the regularized CSPs as 
% detailed  in the follwing paper:
%    Haiping Lu, How-Lung Eng, Cuntai Guan, K.N. Plataniotis, and A.N. Venetsanopoulos,
%    "Regularized Common Spatial Pattern With Aggregation for EEG Classification in Small-Sample Setting",
%    IEEE Trans. on Biomedical Engineering, Vol. 57, No. 12, Pages
%    2936-2946, Dec. 2010.
% Please reference this paper when reporting work done using this code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %[Notes]%:
% A. Developed using Matlab R2006a
% B. Revision history:
%       Version 1.0 released on March 20, 2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function prjW = get_W_regcsp(EEG,gnd,betas,gammas,genSs,genMs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
selCh=6;%number of selected channels/columns
numCls=2;%two classes only
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Regularization parameter used in the paper
% betas=[ 0,1e-2,1e-1,2e-1,4e-1,6e-1];
% gammas=[0,1e-3,1e-2,1e-1,2e-1];
numBeta=length(betas);
numGamma=length(gammas);
numRegs=numBeta*numGamma;
regParas=zeros(numRegs,2);
iReg=0;
for ibeta=1:numBeta
    beta=betas(ibeta);
    for igamma=1:numGamma
        gamma=gammas(igamma);
        iReg=iReg+1;
        regParas(iReg,1)=beta;
        regParas(iReg,2)=gamma;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculate the generic covariance matrices from generic training trials for
%two classes, equation (6) in the paper
% [genSs,genMs]=genericCov;%implement, please refer to line 65-79 of RegCsp.m
%  W=RegCsp(EEG,gnd,genSs,genMs,beta,gamma)
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%Multiple R-CSPs%%%%%%%%%%%
prjW = cell(numRegs,1);
for iReg=1:numRegs
    regPara=regParas(iReg,:);
    %=================RegCSP==========================%     
    %Prototype: W=RegCsp(EEGdata,gnd,genSs,genMs,beta,gamma)
    prjW{iReg} = RegCsp(EEG,gnd,genSs,genMs,regPara(1),regPara(2));
end

end