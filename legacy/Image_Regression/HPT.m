function [ results ] = HPT(t1,t2,mask,ROI)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[s1,s2,s3a] = size(t1);
m = s1*s2;
%mask_small = false(s1,s2);
%mask_small(1501:2000,851:1350) = 1;
t1 = reshape(double(t1),m,s3a);
[~,~,s3b] = size(t2);
t2 = reshape(double(t2),m,s3b);
t1 = 2*t1/max(t1(:))-1;
t2 = 2*t2/max(t2(:))-1;
train1 = t1(mask,:);
train2 = t2(mask,:);
%t1 = t1(mask_small(:),:);
%t2 = t2(mask_small(:),:);
%ROI = ROI(mask_small(:));
%display(sum(ROI(:)))
%s1 = 500;
%s2 = 500;
%m = s1*s2;
results = [];
for k = 50
    tic
    key = k
    display('Doing the first KNNsearch')
    [n1,d1] = knnsearch(train1,t1,'K',key);
    d1(d1 == Inf) = max(max(d1(d1 ~= Inf)));
    % outliers = d1 - mean(d1(:)) > 3*std(d1(:));
    % d1(outliers) = max(max(d1(~outliers)));
    a = max(d1,[],2);
    % d1 = d1./repmat(a,1,key);
    d1 = d1/max(a);
    clear a
    display('Doing the second KNNsearch')
    [n2,d2] = knnsearch(train2,t2,'K',key);
    d2(d2 == Inf) = max(max(d2(d2 ~= Inf)));
    % outliers = d2 - mean(d2(:)) > 3*std(d2(:));
    % d2(outliers) = max(max(d2(~outliers)));
    a = max(d2,[],2);
    % d2 = d2./repmat(a,1,key);
    d2 = d2/max(a);
    clear a
    for g=2
        gamma = 10^g
        W1 = exp(-d1*gamma);
        W1 = W1./repmat(sum(W1,2),1,size(W1,2));
        z1 = zeros(m,s3b);
        W2 = exp(-d2*gamma);
        W2 = W2./repmat(sum(W2,2),1,size(W2,2));
        z2 = zeros(m,s3a);
        para = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(para)
            parpool;
        end
        ppm = ParforProgressStarter2('Training HPT',m);
        for i =1:m
            a = n1(i,:);
            temp = train2(a,:);
            tempW = repmat(W1(i,:)',1,s3b);
            z1(i,:) = sum((tempW.*temp),1);
            a = n2(i,:);
            temp = train1(a,:);
            tempW = repmat(W2(i,:)',1,s3a);
            z2(i,:) = sum((tempW.*temp),1);
            ppm.increment(i);
        end
        delete(ppm)
        dt1 = zeros(s1,s2);
        dt2 = dt1;
        para = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(para)
            parpool;
        end
        ppm = ParforProgressStarter2('Difference in HPT',m);
        parfor i=1:m
            dt1(i) = norm(squeeze(t1(i,:)-z2(i,:)));
            dt2(i) = norm(squeeze(t2(i,:)-z1(i,:)));
            ppm.increment(i);
        end
        delete(ppm)
        outliers = dt1 - mean(dt1(:)) > 3*std(dt1(:));
        dt1(outliers) = max(max(dt1(~outliers)));
        dt1 = dt1/max(dt1(:));
        outliers = dt2 - mean(dt2(:)) > 3*std(dt2(:));
        dt2(outliers) = max(max(dt2(~outliers)));
        dt2 = dt2/max(dt2(:));
        et = toc;
        d = reshape(dt1+dt2,s1,s2);
        d_m = medfilt2(d);
        [~,~,~,AUCdl] = perfcurve(ROI(:),d_m(:),1);
        temp = [AUCdl,et,key,gamma];
        results = [results;temp];
        save HPT_Cal_Aff_Matr_k20 d_m
    end
end
disp('HPT done')
end