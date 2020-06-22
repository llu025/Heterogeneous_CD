function [ results ] = Affinity_matrices(t1,t2,ROI,k,kth,M)

[s1,s2,s3a] = size(t1);

[~,~,s3b] = size(t2);

% filepath = strcat(pwd,'\Results\Affinity_Matrix_Method');
filename = pwd;
if s1 == 3500
    filename = strcat(filename,'\Cal\Frob_norms.mat');
elseif s1 == 1534
    filename = strcat(filename,'\Tx\Frob_norms.mat');
else
    filename = strcat(filename,'\Unknown_dataset\Frob_norms.mat');
end

[filepath,~,~] = fileparts(filename);
if exist(filepath,'dir') ~= 7
    mkdir(filepath);
end
if nargin < 6
    M = 10000;
    if nargin < 5
        kth = 7;
        if nargin < 4
            k = 20;
        end
    end
end


patch_size = k^2;

parallel_computing = true;

n_reduced = (s1-k+1)*(s2-k+1);
Frobs = zeros(n_reduced,4);
if exist(filename, 'file') == 2
    load(filename)
else
    t1 = double(t1);
    t2 = double(t2);
    format long g
    if parallel_computing
        para = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(para)
            parpool
        end
        ppm = ParforProgressStarter2('Test',n_reduced);
        parfor i=1:n_reduced
            [c,a] = ind2sub([s2-k+1,s1-k+1],i);
            b = a + (k-1);
            d = c + (k-1);
            batch_t1 = reshape(t1(a:b,c:d,:),patch_size,s3a);
            batch_t2 = reshape(t2(a:b,c:d,:),patch_size,s3b);
            d1 = pdist(batch_t1).^2;
            d2 = pdist(batch_t2).^2;
            A1 = squareform(d1);
            A2 = squareform(d2);
            kw1 = sort(A1);
            kw2 = sort(A2);
            kw1 = mean(kw1(:,kth+1));
            kw2 = mean(kw2(:,kth+1));
            A1 = exp(-A1/kw1);
            A2 = exp(-A2/kw2);
            Frobs(i,:) = [100*sum(sum(ROI(a:b,c:d)))/patch_size,norm(A1-A2,'fro'),a,c];
            ppm.increment(i);
        end
        delete(ppm)
    else
        display_time = 5000;
        tic
        for i=1:n_reduced
            [c,a] = ind2sub([s2-k+1,s1-k+1],i);
            b = a + (k-1);
            d = c + (k-1);
            batch_t1 = reshape(t1(a:b,c:d,:),patch_size,s3a);
            batch_t2 = reshape(t2(a:b,c:d,:),patch_size,s3b);
            d1 = pdist(batch_t1).^2;
            d2 = pdist(batch_t2).^2;
            A1 = squareform(d1);
            A2 = squareform(d2);
            kw1 = sort(A1);
            kw2 = sort(A2);
            kw1 = mean(kw1(:,kth+1));
            kw2 = mean(kw2(:,kth+1));
            A1 = exp(-A1/kw1);
            A2 = exp(-A2/kw2);
            Frobs(i,:) = [100*sum(sum(ROI(a:b,c:d)))/patch_size,norm(A1-A2,'fro'),a,c];
            time = toc;
            Elapsed_and_remaining_time(i,n_reduced,time,display_time);
        end
    end
    [~,I] = sort(Frobs(:,2));
    Frobs = Frobs(I,:);
    save(filename,'Frobs')
end

bar_plot(Frobs)
covers = zeros(size(ROI));
heatmap = zeros(size(ROI));
for i=1:n_reduced
    a = Frobs(i,3);
    c = Frobs(i,4);
    b = a + (k-1);
    d = c + (k-1);
    heatmap(a:b,c:d) = heatmap(a:b,c:d) + Frobs(i,2);
    covers(a:b,c:d) = covers(a:b,c:d) + 1;
end
heatmap = heatmap./covers;
clear covers

heatmap = heatmap -min(heatmap(:));
heatmap = heatmap./max(heatmap(:));
results = struct;
results.heatmap = heatmap;
[Xh,Yh,~,results.AUC] = perfcurve(ROI(:),heatmap(:),1);
figure; imshow(histeq(mat2gray(heatmap))); colormap(gray)
figure,plot(Xh,Yh);xlabel('False positive rate','FontSize',14); ylabel('True positive rate','FontSize',14);legend({['Heatmap, AUC = ' num2str(results.AUC)]},'FontSize',14,'Location','best')

[~,I] = sort(heatmap(:));

idx_tr2 = I(1:M);
results.idx_tr = false(size(ROI));
results.idx_tr(idx_tr2) = 1;

% BLACK: UNCHANGED, NOT IN TR SET;
% WHITE: CHANGED, NOT IN TR SET;
% GREEN, GOOD SELECTION: UNCHANGED, IN TR SET;
% RED, BAD SELECTION: CHANGED, IN TR SET;
ROI_idx_int = ROI;
ROI_idx_int(:,:,2) = (ROI | results.idx_tr) & ~(ROI & results.idx_tr);
ROI_idx_int(:,:,3) = ROI & ~ results.idx_tr;
figure,imshow(double(ROI_idx_int));

subset = intersect(find(results.idx_tr),find(ROI));
results.errors = 100*length(subset)/M;
sprintf('Percentage of training pixels from changed areas: %d .\n',results.errors);

BH1 = 0;
for i=1:s3a
    temp = reshape(t1(:,:,i),[],1);
    temp_tr = temp(idx_tr2);
    [N,edges] = histcounts(temp,1000,'Normalization','probability');
    [Ntr,~] = histcounts(temp_tr,edges,'Normalization','probability');
    BH1 = BH1 + sum(sqrt(N.*Ntr));
    clear temp N Ntr edges
end
BH1 = sqrt(s3a - BH1)/sqrt(s3a);
sprintf('Hellinger distance t1: %f .\n',BH1);
results.BH1 = BH1;

BH2 = 0;
for i=1:s3b
    temp = reshape(t2(:,:,i),[],1);
    temp_tr = temp(idx_tr2);
    [N,edges] = histcounts(temp,1000,'Normalization','probability');
    [Ntr,~] = histcounts(temp_tr,edges,'Normalization','probability');
    BH2 = BH2 + sum(sqrt(N.*Ntr));
    clear temp N Ntr edges
end
BH2 = sqrt(s3b - BH2)/sqrt(s3b);
sprintf('Hellinger distance t2: %f .\n',BH2);
results.BH2 = BH2;

mask = results.idx_tr;
save("Data_and_training_sample.mat", "t1", "t2", "mask")
disp('Training set selection done')
end
