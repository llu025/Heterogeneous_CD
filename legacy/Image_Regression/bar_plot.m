function [] = bar_plot(A)

fr = A(:,2);
ch = A(:,1);
barpl = [];
for i=1:100
    barpl(i,1) = i;
    idx = intersect(find(ch >= (i-1)),find( ch <= i));
    barpl(i,2) = mean(fr(idx));
end
figure,bar(barpl(:,1),barpl(:,2))