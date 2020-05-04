function [ results ] = mimosvr_Camps_Valls(t1,t2,mask,ROI)
mask = mask(:);
[s1,s2,s3a] = size(t1);
m = s1*s2;
t1 = reshape(double(t1),m,s3a);
[~,~,s3b] = size(t2);
t2 = reshape(double(t2),m,s3b);
if ~any(t1 < 0)
    t1 = 2*t1/max(t1(:))-1;
    t2 = 2*t2/max(t2(:))-1;
end
train1 = t1(mask,:);
train2 = t2(mask,:);
ker     = 'rbf';
tol     = 1e-10;
% epsilon = 0.1;
% C       = 2;
% sigma   = 1;
n = sum(mask(:));
c = cvpartition(n,'HoldOut',1/8);
idx_tr = training(c);
idx_te = test(c);


% train1_r = 2*train1_r/max(train1_r(:)) - 1;
% train2_r = 2*train2_r/max(train2_r(:)) - 1;

tic
l1 = Inf;
l2 = Inf;
for i=1
    for j=-3
        for k =0
            C = 2^i;
            epsilon = 2^j;
            sigma = 2^k;
            idx = randi(size(train1,1),n,1);
            train1_r = train1(idx(idx_tr),:);
            train2_r = train2(idx(idx_tr),:);
            my1 = mean(train1_r);
            my2 = mean(train2_r);
            test1 = train1(idx(idx_te),:);
            test2 = train2(idx(idx_te),:);
            [Beta,~,~] = msvr(train1_r,train2_r-repmat(my2,size(train2_r,1),1)...
                ,ker,C,epsilon,sigma,tol);
            Ktest = kernelmatrix(ker,test1,train1_r,sigma);
%             test1_hat = Ktest*Beta;
%             Ntest = size(test2,1);
%             testmeans = repmat(my2,Ntest,1);
            test1_hat = Ktest*Beta + repmat(my2,size(test2,1),1);

            RMSE = sqrt(sum(sum((test1_hat-test2).^2)) / numel(test2));
            if RMSE < l1
                l1 = RMSE;
                best_C1 = C;
                best_eps1 = epsilon;
                best_Beta1 = Beta;
                best_sigma1 = sigma;
            end
            [Beta,~,~] = msvr(train2_r,train1_r-repmat(my1,size(train1_r,1),1)...
                ,ker,C,epsilon,sigma,tol);
            Ktest = kernelmatrix(ker,test2,train2_r,sigma);
%             test2_hat = Ktest*Beta;
%             Ntest = size(test1,1);
%             testmeans = repmat(my1,Ntest,1);
            test2_hat = Ktest*Beta + repmat(my1,size(test1,1),1);
            RMSE = sqrt(sum(sum((test2_hat-test1).^2)) / numel(test1));
            if RMSE < l2
                l2 = RMSE;                
                best_C2 = C;
                best_eps2 = epsilon;
                best_Beta2 = Beta;
                best_sigma2 = sigma;
            end
        end
    end
end
disp('Training MSVM done')
temp1 = zeros(m,1);
temp2 = temp1;
n = 1;
p = 25000;
while n < m
    a = n;
    n = n + p;
    b = min(n-1,m);
    te1 = t1(a:b,:);
    te2 = t2(a:b,:);
    Ktest = kernelmatrix(ker,te1,train1_r,best_sigma1);
    Ntest = size(te1,1);
    testmeans = repmat(my2,Ntest,1);
    t1_hat = Ktest*best_Beta1 + testmeans;
    temp1(a:b) = sqrt(sum((te2-t1_hat).^2,2));    
    Ntest = size(te2,1);
    testmeans = repmat(my1,Ntest,1);
    Ktest = kernelmatrix(ker,te2,train2_r,best_sigma2);
    t2_hat = Ktest*best_Beta2 + testmeans;    
    temp2(a:b) = sqrt(sum((te1-t2_hat).^2,2)); 
    display(100*b/m)
end
outliers = temp1 - mean(temp1(:)) > 3*std(temp1(:));
temp1(outliers) = max(max(temp1(~outliers)));
temp1 = temp1/max(temp1);
outliers = temp2 - mean(temp2(:)) > 3*std(temp2(:));
temp2(outliers) = max(max(temp2(~outliers)));
temp2 = temp2/max(temp2);
d = reshape(temp1+temp2,s1,s2);
d_m = medfilt2(d);
[~,~,~,AUCdl] = perfcurve(ROI(:),d_m(:),1);
et = toc;
results = [AUCdl,et,best_C1,best_C2,best_eps1,best_eps2,best_sigma1,best_sigma2];
% save SVR_Texas_Aff_Matr_k20 d_m
figure,imshow(histeq(d_m)),colormap(gray)
disp('MSVM done')
end