
clc
clear all

orl_folder = '../datasets/orl_faces/s'
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
A=[];
B=[];

a=20;
b=30;

 

por=1;
for folder=1:2
   for im=1:10
      for exp=1:3 
     file =[orl_folder sprintf('%d',folder) '/' sprintf('%d',im) '.pgm'];
     photo = imread(file);
      xx = (double(photo) + 1)/255;
      A=[A; xx(:)'];
      
      zz=xx;
      [m,n]=size(zz)
      le=randi(n-a);
      up=randi(m-b);
      if (im<=10)
        zz(up:up+b-1,le:le+a-1)=  round(rand(b,a));
      end    
      B=[B; zz(:)'];
      por=por+1;
      end
     %imshow(xx/255)
   end
end
for j=1:10
    B=[B; round(rand(size(zz(:))))' ] ;
end


save('orl_faces.mat')