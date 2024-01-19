clear all
close all

% Load images.
imgDir = uigetdir('Please select a folder...');%'/Users/jacoblamb/Documents/University_of_Cambridge/PhD/Patterning/Patterning_widefield/60x_tiling fibAF546_UV350s_300ms_multi pos_full grid_3';%folder name where the widefield images are
imgSet = imageDatastore(imgDir);
str = imgSet.Files{end};
str = str(end-14:end-8);
I   = sscanf(str,'%d _ %d');
heigth_no = I(1)+1;
width_no = I(2)+1;

% if contains(imgDir,'/')
%     level = wildcardPattern + '/'; %Apple folders
% else
%     level = wildcardPattern + '\'; %Windows folders
% end
% pat = asManyOfPattern(level);
% folder = extractAfter(imgDir,pat);
% 
% Read the first image from the image set.
info = imfinfo(imgSet.Files{1});
channels = numel(info);
test = zeros(1,1,channels);

answer = questdlg('Is the Overlap in um or Percent?', ...
	'Tiff Saving', ...
	'um','Percent','um');
% Handle response
switch answer
    case 'um'
        % Overlap of each image in each direction
        prompt = {'What is the size if each pixel? (nm)'};
        dlgtitle = 'Overlap Value';
        definput = {'107.3'};
        dims = [1 40];
        opts.Interpreter = 'tex';
        pixel_pitch = inputdlg(prompt,dlgtitle,dims,definput,opts);
        x_y_pixel_pitch = str2double(pixel_pitch{1});

        % Overlap of each image in each direction
        prompt = {'What is the image overlap? (um)'};
        dlgtitle = 'Overlap Value';
        definput = {'200'};
        dims = [1 40];
        opts.Interpreter = 'tex';
        overlap = inputdlg(prompt,dlgtitle,dims,definput,opts);
        x_y_overlap = str2double(overlap{1});

        I = double(imread(imgSet.Files{1},1));
        I = imrotate(I,90);
        sz = size(I);

        xoverlap = (x_y_overlap*1E3/(sz(2)*x_y_pixel_pitch));
        yoverlap = (x_y_overlap*1E3/(sz(1)*x_y_pixel_pitch));

    case 'Percent'
        % Overlap of each image in each direction
        prompt = {'What is the image percentage overlap? (%)'};
        dlgtitle = 'Overlap Value';
        definput = {'20'};
        dims = [1 40];
        opts.Interpreter = 'tex';
        answer = inputdlg(prompt,dlgtitle,dims,definput,opts);

        
        xoverlap = str2double(answer{1})/100;
        yoverlap = str2double(answer{1})/100;
end


f = waitbar(0,'Please wait...');
for k = 1:channels
    %     I = loadtiff(imgSet.Files{1});
    I = double(imread(imgSet.Files{1},k));
    I = I-min(I(:));
    I = I./max(I(:));
    I = imrotate(I,90);
    Sz = size(I);
    
    %Stitch Images together
    count = 1;
    for i = [heigth_no:-1:1] % [heigth_no:-1:1] = snaked. [1:heigth_no] = no snake
        for j = [1:width_no]
            I = double(imread(imgSet.Files{count},k));
            I = I-min(I(:));
            I = I./(max(I(:))*0.5);
            I = imrotate(I,90);
            if i == 1
                ystart = 1;
                ystop = Sz(1);
            else
                ystart = round((Sz(1)-Sz(1)*yoverlap)+((i-2)*Sz(1)*(1-yoverlap)));
                ystop = round((2*Sz(1)-Sz(1)*yoverlap-1)+((i-2)*Sz(1)*(1-yoverlap)));
            end
            if j == 1
                xstart = 1;
                xstop = Sz(2);
            else
                xstart = round((Sz(2)-Sz(2)*xoverlap)+((j-2)*Sz(2)*(1-xoverlap)));
                xstop = round((2*Sz(2)-Sz(2)*xoverlap)-1+((j-2)*Sz(2)*(1-xoverlap)));
            end
            Test(xstart:xstop , ystart:ystop,k) = I;
            count = count + 1;
            waitbar(count/(heigth_no*width_no),f,['Stitching Colour ',num2str(k)]);
        end
    end
    waitbar(1,f,['Stitching Complete']);
end
Test = imrotate(Test,-90);
close(f)

answer = questdlg('Would you save this stack?', ...
	'Tiff Saving', ...
	'Yes','No','Yes');
% Handle response
switch answer
    case 'Yes'
        prompt = {'Please Enter Save Name'};
        dlgtitle = 'Overlap Value';
        definput = {'Test 1'};
        dims = [1 40];
        opts.Interpreter = 'tex';
        folder = inputdlg(prompt,dlgtitle,dims,definput,opts);

        saveDir = uigetdir('Please select a save folder...');%'/Users/jacoblamb/Documents/University_of_Cambridge/PhD/Patterning/Patterning_widefield/60x_tiling fibAF546_UV350s_300ms_multi pos_full grid_3';%folder name where the widefield images are
        imwrite(Test(:,:,1),strcat(saveDir,'/',folder{1},'Stitched.tiff'));
        if channels > 1
            for k = 2:channels 
                imwrite(Test(:,:,k),strcat(saveDir,'/',folder{1},'Stitched.tiff'),'WriteMode','append');
            end
        end
    case 'No'
end

if channels == 2
    figure(1)
    imshowpair(Test(:,:,1),Test(:,:,2),'Scaling','none','ColorChannels','red-cyan')
    
    figure(2)
    imshowpair(imadjust(Test(:,:,1)),imadjust(Test(:,:,2)),'Scaling','none','ColorChannels','red-cyan')
elseif channels == 1
    figure(1)
    imshow(Test)
    
    figure(2)
    imshow(imadjust(Test))
end