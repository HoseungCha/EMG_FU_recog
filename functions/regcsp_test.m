function regcsp_test(allFDAFtrs, prjW)
%%%%%%%%%%%%%%%%%%%%%
%Apply FDA
allFDAFtrs=zeros(nTrl,numRegs);%Features after FDA
for iReg=1:numRegs
    newfea=allNewFtrs(1:selCh,:,iReg);
    FDAU=FDA(newfea(:,trainIdx),gnd);
    newfea=LDAU'*newfea;
    newfea=newfea';
    allFDAFtrs(:,iReg)=newfea;
end
%%%%%%%%%R-CSP-A%%%%%%%%%%%
end