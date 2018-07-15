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

function allNewFtrs = get_feat_using_w_regcsp(EEG,prjW,selCh)
 
[~,~,nTrl] = size(EEG);%suppose data is loaded to variable "EEG"
fea2D = EEG;
numRegs = length(prjW);
%%%%%%%%%Multiple R-CSPs%%%%%%%%%%%
allNewFtrs=zeros(nTrl,selCh,numRegs);%Projected Features for R-CSP
for iReg=1:numRegs
    newfea=zeros(nTrl,selCh);
    for iCh=1:selCh
        for iTr=1:nTrl
            infea=fea2D(:,:,iTr)';
            prjfea=prjW{iReg}(iCh,:)*infea;
            newfea(iTr,iCh)=log(var(prjfea));
        end
    end
    allNewFtrs(:,:,iReg) = newfea;
end
end