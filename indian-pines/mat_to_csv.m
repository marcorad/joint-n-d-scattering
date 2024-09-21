load Indian_pines_corrected.mat
load Indian_pines_gt.mat
writematrix(indian_pines_gt, 'labels.csv')
for i = 1:size(indian_pines_corrected, 3)
    writematrix(indian_pines_corrected(:, :, i),int2str(i) + ".csv")
end
indian_pines_corrected(1:5, 1:10, 1)