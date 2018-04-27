clear all; close all; clc;

%%

prefix = '16_40_52Z_Test 1_';
suffix = '.png';
range = 1:21;


threshold = 20;
show_centroid = false;
show_threshold = false;



%%

centroid_centers = zeros(length(range),2);
threshold_centers = zeros(length(range),2);

for indi=1:length(range)
    Im = imread(horzcat(prefix,sprintf('%04d',range(indi)),suffix)); % Load up the image
    
    % Find the center by centroiding
    stats = regionprops(Im);
    center = stats.Centroid;
    
    [y, x] = ndgrid(1:size(Im, 1), 1:size(Im, 2));
    xbar = sum(x(logical(Im)).*double(Im(logical(Im))))/sum(double(Im(logical(Im))));
    ybar = sum(y(logical(Im)).*double(Im(logical(Im))))/sum(double(Im(logical(Im))));
    
    centroid = [xbar,ybar];
    
    centroid_centers(indi,:) = centroid;
    
    if(show_centroid)
        figure
        hold on
        imshow(Im)
        text(center(1),center(2),'\leftarrow Center','color','Red');
        text(xbar,ybar,'\leftarrow [xbar,ybar]','color','Green');
    end
    
    % Find the center by thresholding
    map = (Im >= threshold);
    stats = regionprops(map);
    center = stats.Centroid;
    
    [y, x] = ndgrid(1:size(map, 1), 1:size(map, 2));
    centroid = mean([x(logical(map)), y(logical(map))]);
    
    threshold_centers(indi,:) = centroid;
    
    if(show_threshold)
        figure
        imshow(map)
        text(center(1),center(2),'\leftarrow Center','color','Red');
        text(centroid(1),centroid(2),'\leftarrow Centroid','color','Blue');
    end
    
end


%%

figure
hold on
plot(range,centroid_centers(:,1),'b')
plot(range,threshold_centers(:,1),'--b')

plot(range,centroid_centers(:,2),'r')
plot(range,threshold_centers(:,2),'--r')

legend('X by centroid','X by threshold','Y by centroid','Y by threshold')
title('Spot Center Coordinates wrt Image Number','fontsize',20)
xlabel('Image Number','fontsize',20)
ylabel('Spot Center Coordinates (pixels)','fontsize',20)


%% Set 1 - tip resolution

set1 = 1:10;

figure
hold on
scatter(threshold_centers(set1,1),threshold_centers(set1,2),100,set1)
colormap('jet')
colorbar

figure
hold on
plot(set1,threshold_centers(set1,1))
plot(set1,threshold_centers(set1,2))
legend('x center','y center')
xlabel('image number')


pixel_size = 5.86*10^-6;    % m
meter2inch = 39.3701;       % in/m
focal_length = 24;          % inches
rad2arcsec = 206265;        % arcsec/rad

distances = meter2inch * pixel_size * sqrt(diff(threshold_centers(set1,1)).^2 + diff(threshold_centers(set1,2)).^2) % inches
angles = rad2arcsec * atan(distances/focal_length) % arcsec

anglemean = mean(angles)
anglestd = std(angles)



%% Check stability of image source (yeah, it was the laser source. So what...
total_I = zeros(length(range),1);
for indi=1:length(range)
    Im = imread(horzcat(prefix,sprintf('%04d',range(indi)),suffix)); % Load up the image
    
    total_I(indi) = sum(sum(Im));
    
end


meanI = mean(total_I)
stdI = std(total_I)

stdI/meanI


