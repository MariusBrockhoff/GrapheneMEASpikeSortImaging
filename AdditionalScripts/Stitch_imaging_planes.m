clear all
close all

% Load images.
imgDir = uigetdir('Please select a folder...'); % Open a dialog to select the folder containing images.
imgSet = imageDatastore(imgDir); % Create an image datastore for all images in the selected folder.
str = imgSet.Files{end}; % Get the file name of the last image in the folder.
str = str(end-14:end-8); % Extract the specific part of the file name that contains positional information.
I   = sscanf(str,'%d _ %d'); % Read the positional information as two integers.
heigth_no = I(1)+1; % Determine the number of image tiles in height.
width_no = I(2)+1; % Determine the number of image tiles in width.

% Read the first image from the image set.
info = imfinfo(imgSet.Files{1}); % Get information about the first image.
channels = numel(info); % Determine the number of channels in the image.
test = zeros(1,1,channels); % Initialize a 3D array for storing images.

% Ask user whether the overlap is in micrometers or percentage.
answer = questdlg('Is the Overlap in um or Percent?', ...
    'Tiff Saving', ...
    'um','Percent','um');

% Handle response for overlap unit.
switch answer
    case 'um'
        % Get pixel size and overlap in micrometers from user.
        prompt = {'What is the size of each pixel? (nm)'};
        dlgtitle = 'Overlap Value';
        definput = {'107.3'};
        dims = [1 40];
        opts.Interpreter = 'tex';
        pixel_pitch = inputdlg(prompt,dlgtitle,dims,definput,opts);
        x_y_pixel_pitch = str2double(pixel_pitch{1});

        prompt = {'What is the image overlap? (um)'};
        dlgtitle = 'Overlap Value';
        definput = {'200'};
        dims = [1 40];
        opts.Interpreter = 'tex';
        overlap = inputdlg(prompt,dlgtitle,dims,definput,opts);
        x_y_overlap = str2double(overlap{1});

        % Calculate the overlap in pixels.
        I = double(imread(imgSet.Files{1},1));
        I = imrotate(I,90); % Rotate the image by 90 degrees.
        sz = size(I);

        xoverlap = (x_y_overlap*1E3/(sz(2)*x_y_pixel_pitch)); % Calculate x overlap in pixels.
        yoverlap = (x_y_overlap*1E3/(sz(1)*x_y_pixel_pitch)); % Calculate y overlap in pixels.

    case 'Percent'
        % Get the percentage overlap from user.
        prompt = {'What is the image percentage overlap? (%)'};
        dlgtitle = 'Overlap Value';
        definput = {'20'};
        dims = [1 40];
        opts.Interpreter = 'tex';
        answer = inputdlg(prompt,dlgtitle,dims,definput,opts);

        % Calculate the overlap as a fraction.
        xoverlap = str2double(answer{1})/100;
        yoverlap = str2double(answer{1})/100;
end

% Initialize waitbar for stitching process.
f = waitbar(0,'Please wait...');

for k = 1:channels
    I = double(imread(imgSet.Files{1},k));
    I = I-min(I(:)); % Normalize the image.
    I = I./max(I(:));
    I = imrotate(I,90); % Rotate the image.
    Sz = size(I);

    % Initialize a counter for image stitching.
    count = 1;
    for i = [heigth_no:-1:1] % Loop through the height tiles.
        for j = [1:width_no] % Loop through the width tiles.
            I = double(imread(imgSet.Files{count},k));
            I = I-min(I(:)); % Normalize the image.
            I = I./(max(I(:))*0.5);
            I = imrotate(I,90); % Rotate the image.
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
            Test(xstart:xstop , ystart:ystop,k) = I; % Stitch the image.
            count = count + 1;
            waitbar(count/(heigth_no*width_no),f,['Stitching Colour ',num2str(k)]); % Update waitbar.
        end
    end
    waitbar(1,f,['Stitching Complete']); % Indicate stitching completion.
end

Test = imrotate(Test,-90); % Rotate the final stitched image back.
close(f) % Close the waitbar.

% Ask the user whether to save the stitched image.
answer = questdlg('Would you save this stack?', ...
    'Tiff Saving', ...
    'Yes','No','Yes');

% Handle response for saving the stitched image.
switch answer
    case 'Yes'
        prompt = {'Please Enter Save Name'};
        dlgtitle = 'Save Name';
        definput = {'Test 1'};
        dims = [1 40];
        opts.Interpreter = 'tex';
        folder = inputdlg(prompt,dlgtitle,dims,definput,opts);

        saveDir = uigetdir('Please select a save folder...'); % Open a dialog to select the save folder.
        imwrite(Test(:,:,1),strcat(saveDir,'/',folder{1},'Stitched.tiff')); % Save the first channel.
        if channels > 1
            for k = 2:channels 
                imwrite(Test(:,:,k),strcat(saveDir,'/',folder{1},'Stitched.tiff'),'WriteMode','append'); % Append other channels.
            end
        end
    case 'No'
end

% Display the stitched image.
if channels == 2
    figure(1)
    imshowpair(Test(:,:,1),Test(:,:,2),'Scaling','none','ColorChannels','red-cyan') % Display as red-cyan pair.

    figure(2)
    imshowpair(imadjust(Test(:,:,1)),imadjust(Test(:,:,2)),'Scaling','none','ColorChannels','red-cyan') % Adjust and display as red-cyan pair.
elseif channels == 1
    figure(1)
    imshow(Test) % Display single channel image.

    figure(2)
    imshow(imadjust(Test)) % Adjust and display single channel image.
end
