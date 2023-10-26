%[f,p]=uigetfile();
%I=imread([p,f]);
%I=imresize(I,[512 512]);
I = imread("C:\Users\rosha\Downloads\WhatsApp Image 2023-09-03 at 23.46.01.jpg");
I2=rgb2gray(I);
%BW2 = bwmorph(I2,'remove');
%figure,imshow(BW2)
%title('figure1:bwmorph remove')
%BW3 = bwmorph(I2,'skel',Inf);
%figure,imshow(BW3)
%title('figure:2 bwmorph inf')

SE = strel('arbitrary',eye(5));

BW2 = imerode(I2,SE);
figure,imshow(BW2)
title('figure:3 imerode')
BW3 = imdilate(BW2,SE);
figure,imshow(BW3)
title('figure:4 imdilate')
closeBW = imclose(BW3,SE);
figure, imshow(closeBW)
title('figure:5 imclose')
hy = fspecial('sobel');

hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
Iobrd = imdilate(BW2, SE );
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(closeBW));
Iobrcbr = imcomplement(Iobrcbr);
figure, imshow(Iobrcbr)
title('figure:6 imcomplement')
fgm = imregionalmax(Iobrcbr);
figure, imshow(fgm)
figure,imshow(imcomplement(fgm))
imwrite(fgm,"C:\Users\rosha\Downloads\rainmap.jpg")
title('figure:7 imregionalmax')
I2 = I;
I2(fgm) = 200;
figure, imshow(I2)
title('figure:8 fgm')
se2 = strel(ones(5,5));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);

way;
