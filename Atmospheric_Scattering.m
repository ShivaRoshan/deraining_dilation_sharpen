function Haze = Atmospheric_Scattering(HazeImg,A, T)
delta = 1;
T = max(abs(T), 0.0001).^delta;
HazeImg = double(HazeImg);
Haze = double(HazeImg);
if length(A) == 1
    A = A * ones(3, 1);
end
for c=1:3
    Haze(:,:,c) = HazeImg(:, :, c) .* T + (A(c).*(abs(ones(size(T))-T)));   
end 
end
    
