function main(accuracy)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
%Hossein Ghasemi Ramshe  9433590
%Ali Taheri   9433597
a = int64(imread('cameraman.tif')); 
tmp=zeros(256,1);
%tmp=uint16(tmp);
%a=unit16(a);
n=256;
x = 4;
counter=1;
matrix=[];
%matrix=unit16(matrix);
matrix=[a(:,1)];
for i=2:n
   for j=0:x-2
      matrix=[matrix tmp] ;
      
   end
   matrix=[matrix a(:,i)];
   
end
   for j=0:x-2
      matrix=[matrix tmp] ;
      
   end
for i = 1 : 256
    for j = 1 : 1024
        if rem(j, 4)== 1
            continue
        else
            matrix(i,j) = int64(lagrange(j, i, accuracy, matrix));
            
        end
    end  
end
matrix = uint8(matrix);
imshow(matrix);
end