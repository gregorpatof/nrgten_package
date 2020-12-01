% Spring Tensor Model
% Author: Tu-Liang Lin
% Created: Feb 2, 2010
%
% caArray is a n*3 matrix which contains the CA coordinates
% hv1 is the Hessian matrix obtained from the first term of the go-like potential
% hv2 is the Hessian matrix obtained from the second term of the go-like potential
% hv3 is the Hessian matrix obtained from the third term of the go-like potential
% hv4 is the Hessian matrix obtained from the fourth term of the go-like potential
% STeM Hessian=hv1+hv2+hv3+hv4
function [hv1,hv2, hv3, hv4]=STeM(caArray)
Epsilon=0.36;
K_r=100*Epsilon;
K_theta=20*Epsilon;
K_phi1=1*Epsilon;
K_phi3=0.5*Epsilon;
numOfResidues=size(caArray,1);
distance=squareform(pdist(caArray));

hessian=zeros(numOfResidues*3);
hv1=firstTerm(hessian,caArray,distance,numOfResidues,K_r);
hessian=zeros(numOfResidues*3);
hv2=secondTerm(hessian,caArray,distance,numOfResidues,K_theta);
hessian=zeros(numOfResidues*3);
hv3=thirdTerm(hessian,caArray,distance,numOfResidues,K_phi1,K_phi3);
hessian=zeros(numOfResidues*3);
hv4=fourthTerm(hessian,caArray,distance,numOfResidues,Epsilon);
end
function g=glength(X)
    g=sqrt(X(1)^2+X(2)^2+X(3)^2);
end

function hessian=firstTerm(hessian,caArray,distance,numOfResidues,K_r)
% derive the hessian of the first term (off diagonal)
for m=2:numOfResidues;
       i=m-1;
       j=m;          
          	bx=caArray(i,1) - caArray(j,1);
       		by=caArray(i,2) - caArray(j,2);
           	bz=caArray(i,3) - caArray(j,3);
            distijsqr=distance(i,j)^2;
                       
            %Hij
			% diagonals of off-diagonal super elements (1st term)

			hessian(3*i-2,3*j-2)       = hessian(3*i-2,3*j-2)-2*K_r*bx*bx/distijsqr;
			hessian(3*i-1,3*j-1)       = hessian(3*i-1,3*j-1)-2*K_r*by*by/distijsqr;
			hessian(3*i,3*j)           = hessian(3*i,3*j)-2*K_r*bz*bz/distijsqr;

			% off-diagonals of off-diagonal super elements (1st term)

			hessian(3*i-2,3*j-1)       = hessian(3*i-2,3*j-1)-2*K_r*bx*by/distijsqr;
			hessian(3*i-2,3*j)         = hessian(3*i-2,3*j)-2*K_r*bx*bz/distijsqr;
			hessian(3*i-1,3*j-2)       = hessian(3*i-1,3*j-2)-2*K_r*by*bx/distijsqr;
			hessian(3*i-1,3*j)         = hessian(3*i-1,3*j)-2*K_r*by*bz/distijsqr;
			hessian(3*i,3*j-2)         = hessian(3*i,3*j-2)-2*K_r*bz*bx/distijsqr;
			hessian(3*i,3*j-1)         = hessian(3*i,3*j-1)-2*K_r*bz*by/distijsqr;                                                
            
            
            %Hji
			% diagonals of off-diagonal super elements (1st term)

			hessian(3*j-2,3*i-2)       = hessian(3*j-2,3*i-2)-2*K_r*bx*bx/distijsqr;
			hessian(3*j-1,3*i-1)       = hessian(3*j-1,3*i-1)-2*K_r*by*by/distijsqr;
			hessian(3*j,3*i)           = hessian(3*j,3*i)-2*K_r*bz*bz/distijsqr;

			% off-diagonals of off-diagonal super elements (1st term)

			hessian(3*j-2,3*i-1)       = hessian(3*j-2,3*i-1)-2*K_r*bx*by/distijsqr;
			hessian(3*j-2,3*i)         = hessian(3*j-2,3*i)-2*K_r*bx*bz/distijsqr;
			hessian(3*j-1,3*i-2)       = hessian(3*j-1,3*i-2)-2*K_r*by*bx/distijsqr;
			hessian(3*j-1,3*i)         = hessian(3*j-1,3*i)-2*K_r*by*bz/distijsqr;
			hessian(3*j,3*i-2)         = hessian(3*j,3*i-2)-2*K_r*bz*bx/distijsqr;
			hessian(3*j,3*i-1)         = hessian(3*j,3*i-1)-2*K_r*bz*by/distijsqr;                                                            
            
            %Hii
            %update the diagonals of diagonal super elements
            
            hessian(3*i-2,3*i-2)       = hessian(3*i-2,3*i-2)+2*K_r*bx*bx/distijsqr;
            hessian(3*i-1,3*i-1)       = hessian(3*i-1,3*i-1)+2*K_r*by*by/distijsqr;
            hessian(3*i,3*i)           = hessian(3*i,3*i)+2*K_r*bz*bz/distijsqr;                        
            
    		% update the off-diagonals of diagonal super elements			
      		hessian(3*i-2,3*i-1)       = hessian(3*i-2,3*i-1)+2*K_r*bx*by/distijsqr;
           	hessian(3*i-2,3*i)         = hessian(3*i-2,3*i)+2*K_r*bx*bz/distijsqr;
   			hessian(3*i-1,3*i-2)       = hessian(3*i-1,3*i-2)+2*K_r*by*bx/distijsqr;
       		hessian(3*i-1,3*i)         = hessian(3*i-1,3*i)+2*K_r*by*bz/distijsqr;
           	hessian(3*i,3*i-2)         = hessian(3*i,3*i-2)+2*K_r*bz*bx/distijsqr;
   			hessian(3*i,3*i-1)         = hessian(3*i,3*i-1)+2*K_r*bz*by/distijsqr;                        

            %Hjj
            %update the diagonals of diagonal super elements
            
            hessian(3*j-2,3*j-2)       = hessian(3*j-2,3*j-2)+2*K_r*bx*bx/distijsqr;
            hessian(3*j-1,3*j-1)       = hessian(3*j-1,3*j-1)+2*K_r*by*by/distijsqr;
            hessian(3*j,3*j)           = hessian(3*j,3*j)+2*K_r*bz*bz/distijsqr;                        
            
    		% update the off-diagonals of diagonal super elements			
      		hessian(3*j-2,3*j-1)       = hessian(3*j-2,3*j-1)+2*K_r*bx*by/distijsqr;
           	hessian(3*j-2,3*j)         = hessian(3*j-2,3*j)+2*K_r*bx*bz/distijsqr;
   			hessian(3*j-1,3*j-2)       = hessian(3*j-1,3*j-2)+2*K_r*by*bx/distijsqr;
       		hessian(3*j-1,3*j)         = hessian(3*j-1,3*j)+2*K_r*by*bz/distijsqr;
           	hessian(3*j,3*j-2)         = hessian(3*j,3*j-2)+2*K_r*bz*bx/distijsqr;
   			hessian(3*j,3*j-1)         = hessian(3*j,3*j-1)+2*K_r*bz*by/distijsqr;                                                
end

end

function hessian=secondTerm(hessian,caArray,distance,numOfResidues,K_theta)
% derive the hessian of the second term
    for m=2:numOfResidues-1;
            i=m-1;
            j=m;
            k=m+1;           
                Xi=caArray(i,1);
                Yi=caArray(i,2);
                Zi=caArray(i,3);
                Xj=caArray(j,1);
                Yj=caArray(j,2);
                Zj=caArray(j,3);
                Xk=caArray(k,1);
                Yk=caArray(k,2);
                Zk=caArray(k,3);
            
                p=caArray(i,1:3)-caArray(j,1:3);
                lpl=distance(i,j);
                q=caArray(k,1:3)-caArray(j,1:3);
                lql=distance(k,j);
                G=dot(p,q)/(lpl*lql);                 
                
                %dG/dXi
                dGdXi=((Xk-Xj)*lpl*lql-dot(p,q)*(lql/lpl)*(Xi-Xj))/(lpl*lql)^2;
                %dG/dYi
                dGdYi=((Yk-Yj)*lpl*lql-dot(p,q)*(lql/lpl)*(Yi-Yj))/(lpl*lql)^2;
                %dG/dZi
                dGdZi=((Zk-Zj)*lpl*lql-dot(p,q)*(lql/lpl)*(Zi-Zj))/(lpl*lql)^2;
                
                %dG/dXj
                dGdXj=((2*Xj-Xi-Xk)*lpl*lql-dot(p,q)*(lql/lpl)*(Xj-Xi)-dot(p,q)*(lpl/lql)*(Xj-Xk))/(lpl*lql)^2;
                %dG/dYj
                dGdYj=((2*Yj-Yi-Yk)*lpl*lql-dot(p,q)*(lql/lpl)*(Yj-Yi)-dot(p,q)*(lpl/lql)*(Yj-Yk))/(lpl*lql)^2;
                %dG/dZj
                dGdZj=((2*Zj-Zi-Zk)*lpl*lql-dot(p,q)*(lql/lpl)*(Zj-Zi)-dot(p,q)*(lpl/lql)*(Zj-Zk))/(lpl*lql)^2;
                
                %dG/dXk
                dGdXk=((Xi-Xj)*lpl*lql-dot(p,q)*(lpl/lql)*(Xk-Xj))/(lpl*lql)^2;                                                            
                %dG/dYk
                dGdYk=((Yi-Yj)*lpl*lql-dot(p,q)*(lpl/lql)*(Yk-Yj))/(lpl*lql)^2;
                %dG/dZk
                dGdZk=((Zi-Zj)*lpl*lql-dot(p,q)*(lpl/lql)*(Zk-Zj))/(lpl*lql)^2;
                                
                %Hij
                %d^2V/dXidXj  d^2V/dXidYj  d^2V/dXidZj
                %d^2V/dYidXj  d^2V/dYidYj  d^2V/dYidZj
                %d^2V/dZidXj  d^2V/dZidYj  d^2V/dZidZj             
              
        		% diagonals of off-diagonal super elements             

            	hessian(3*i-2,3*j-2)       = hessian(3*i-2,3*j-2)+2*K_theta/(1-G^2)*dGdXi*dGdXj;
    			hessian(3*i-1,3*j-1)       = hessian(3*i-1,3*j-1)+2*K_theta/(1-G^2)*dGdYi*dGdYj;
        		hessian(3*i,3*j)           = hessian(3*i,3*j)+2*K_theta/(1-G^2)*dGdZi*dGdZj;

            
            	% off-diagonals of off-diagonal super elements

    			hessian(3*i-2,3*j-1)       = hessian(3*i-2,3*j-1)+2*K_theta/(1-G^2)*dGdXi*dGdYj;
        		hessian(3*i-2,3*j)         = hessian(3*i-2,3*j)+2*K_theta/(1-G^2)*dGdXi*dGdZj;
            	hessian(3*i-1,3*j-2)       = hessian(3*i-1,3*j-2)+2*K_theta/(1-G^2)*dGdYi*dGdXj;
                hessian(3*i-1,3*j)         = hessian(3*i-1,3*j)+2*K_theta/(1-G^2)*dGdYi*dGdZj;
    			hessian(3*i,3*j-2)         = hessian(3*i,3*j-2)+2*K_theta/(1-G^2)*dGdZi*dGdXj;
        		hessian(3*i,3*j-1)         = hessian(3*i,3*j-1)+2*K_theta/(1-G^2)*dGdZi*dGdYj;                                                                       
            
                %Hji
    			% diagonals of off-diagonal super elements            
        		hessian(3*j-2,3*i-2)       = hessian(3*j-2,3*i-2)+2*K_theta/(1-G^2)*dGdXj*dGdXi;
            	hessian(3*j-1,3*i-1)       = hessian(3*j-1,3*i-1)+2*K_theta/(1-G^2)*dGdYj*dGdYi;
    			hessian(3*j,3*i)           = hessian(3*j,3*i)+2*K_theta/(1-G^2)*dGdZj*dGdZi;

        		% off-diagonals of off-diagonal super elements
    
        		hessian(3*j-2,3*i-1)       = hessian(3*j-2,3*i-1)+2*K_theta/(1-G^2)*dGdXj*dGdYi;
    			hessian(3*j-2,3*i)         = hessian(3*j-2,3*i)+2*K_theta/(1-G^2)*dGdXj*dGdZi;
        		hessian(3*j-1,3*i-2)       = hessian(3*j-1,3*i-2)+2*K_theta/(1-G^2)*dGdYj*dGdXi;
    			hessian(3*j-1,3*i)         = hessian(3*j-1,3*i)+2*K_theta/(1-G^2)*dGdYj*dGdZi;
        		hessian(3*j,3*i-2)         = hessian(3*j,3*i-2)+2*K_theta/(1-G^2)*dGdZj*dGdXi;
    			hessian(3*j,3*i-1)         = hessian(3*j,3*i-1)+2*K_theta/(1-G^2)*dGdZj*dGdYi;                                                                                   
            
                %disp(G);
                %disp([i,j]);
                %disp(hessian(3*i-2:3*i,3*j-2:3*j));
                %disp(hessian(3*j-2:3*j,3*i-2:3*i));
                
                
                %Hjk
                %d^2V/dXjdXk  d^2V/dXjdYk  d^2V/dXjdZk
                %d^2V/dYjdXk  d^2V/dYjdYk  d^2V/dYjdZk
                %d^2V/dZjdXk  d^2V/dZjdYk  d^2V/dZjdZk
            
    			% diagonals of off-diagonal super elements

        		hessian(3*j-2,3*k-2)       = hessian(3*j-2,3*k-2)+2*K_theta/(1-G^2)*dGdXj*dGdXk;
    			hessian(3*j-1,3*k-1)       = hessian(3*j-1,3*k-1)+2*K_theta/(1-G^2)*dGdYj*dGdYk;
        		hessian(3*j,3*k)           = hessian(3*j,3*k)+2*K_theta/(1-G^2)*dGdZj*dGdZk;
            
            	% off-diagonals of off-diagonal super elements

    			hessian(3*j-2,3*k-1)       = hessian(3*j-2,3*k-1)+2*K_theta/(1-G^2)*dGdXj*dGdYk;
        		hessian(3*j-2,3*k)         = hessian(3*j-2,3*k)+2*K_theta/(1-G^2)*dGdXj*dGdZk;
    			hessian(3*j-1,3*k-2)       = hessian(3*j-1,3*k-2)+2*K_theta/(1-G^2)*dGdYj*dGdXk;
        		hessian(3*j-1,3*k)         = hessian(3*j-1,3*k)+2*K_theta/(1-G^2)*dGdYj*dGdZk;
            	hessian(3*j,3*k-2)         = hessian(3*j,3*k-2)+2*K_theta/(1-G^2)*dGdZj*dGdXk;
    			hessian(3*j,3*k-1)         = hessian(3*j,3*k-1)+2*K_theta/(1-G^2)*dGdZj*dGdYk;                                                                       
            
                %Hkj

    			% diagonals of off-diagonal super elements

    			hessian(3*k-2,3*j-2)       = hessian(3*k-2,3*j-2)+2*K_theta/(1-G^2)*dGdXk*dGdXj;
        		hessian(3*k-1,3*j-1)       = hessian(3*k-1,3*j-1)+2*K_theta/(1-G^2)*dGdYk*dGdYj;
    			hessian(3*k,3*j)           = hessian(3*k,3*j)+2*K_theta/(1-G^2)*dGdZk*dGdZj;
            
    			% off-diagonals of off-diagonal super elements
    
    			hessian(3*k-2,3*j-1)       = hessian(3*k-2,3*j-1)+2*K_theta/(1-G^2)*dGdXk*dGdYj;
        		hessian(3*k-2,3*j)         = hessian(3*k-2,3*j)+2*K_theta/(1-G^2)*dGdXk*dGdZj;
    			hessian(3*k-1,3*j-2)       = hessian(3*k-1,3*j-2)+2*K_theta/(1-G^2)*dGdYk*dGdXj;
    			hessian(3*k-1,3*j)         = hessian(3*k-1,3*j)+2*K_theta/(1-G^2)*dGdYk*dGdZj;
        		hessian(3*k,3*j-2)         = hessian(3*k,3*j-2)+2*K_theta/(1-G^2)*dGdZk*dGdXj;
    			hessian(3*k,3*j-1)         = hessian(3*k,3*j-1)+2*K_theta/(1-G^2)*dGdZk*dGdYj;                                                                       
            
                %Hik
                %d^2V/dXidXk  d^2V/dXidYk  d^2V/dXidZk
                %d^2V/dYidXk  d^2V/dYidYk  d^2V/dYidZk
                %d^2V/dZidXk  d^2V/dZidYk  d^2V/dZidZk
                                    
    			% diagonals of off-diagonal super elements

    			hessian(3*i-2,3*k-2)       = hessian(3*i-2,3*k-2)+2*K_theta/(1-G^2)*dGdXi*dGdXk;
        		hessian(3*i-1,3*k-1)       = hessian(3*i-1,3*k-1)+2*K_theta/(1-G^2)*dGdYi*dGdYk;
    			hessian(3*i,3*k)           = hessian(3*i,3*k)+2*K_theta/(1-G^2)*dGdZi*dGdZk;
            
    			% off-diagonals of off-diagonal super elements

        		hessian(3*i-2,3*k-1)       = hessian(3*i-2,3*k-1)+2*K_theta/(1-G^2)*dGdXi*dGdYk;
    			hessian(3*i-2,3*k)         = hessian(3*i-2,3*k)+2*K_theta/(1-G^2)*dGdXi*dGdZk;
    			hessian(3*i-1,3*k-2)       = hessian(3*i-1,3*k-2)+2*K_theta/(1-G^2)*dGdYi*dGdXk;
        		hessian(3*i-1,3*k)         = hessian(3*i-1,3*k)+2*K_theta/(1-G^2)*dGdYi*dGdZk;
    			hessian(3*i,3*k-2)         = hessian(3*i,3*k-2)+2*K_theta/(1-G^2)*dGdZi*dGdXk;
        		hessian(3*i,3*k-1)         = hessian(3*i,3*k-1)+2*K_theta/(1-G^2)*dGdZi*dGdYk;                                                                       

                %Hki
            
    			% diagonals of off-diagonal super elements

        		hessian(3*k-2,3*i-2)       = hessian(3*k-2,3*i-2)+2*K_theta/(1-G^2)*dGdXk*dGdXi;
    			hessian(3*k-1,3*i-1)       = hessian(3*k-1,3*i-1)+2*K_theta/(1-G^2)*dGdYk*dGdYi;
        		hessian(3*k,3*i)           = hessian(3*k,3*i)+2*K_theta/(1-G^2)*dGdZk*dGdZi;
            
    			% off-diagonals of off-diagonal super elements

        		hessian(3*k-2,3*i-1)       = hessian(3*k-2,3*i-1)+2*K_theta/(1-G^2)*dGdXk*dGdYi;
            	hessian(3*k-2,3*i)         = hessian(3*k-2,3*i)+2*K_theta/(1-G^2)*dGdXk*dGdZi;
    			hessian(3*k-1,3*i-2)       = hessian(3*k-1,3*i-2)+2*K_theta/(1-G^2)*dGdYk*dGdXi;
        		hessian(3*k-1,3*i)         = hessian(3*k-1,3*i)+2*K_theta/(1-G^2)*dGdYk*dGdZi;
            	hessian(3*k,3*i-2)         = hessian(3*k,3*i-2)+2*K_theta/(1-G^2)*dGdZk*dGdXi;
                hessian(3*k,3*i-1)         = hessian(3*k,3*i-1)+2*K_theta/(1-G^2)*dGdZk*dGdYi;                                                                       
                
                %Hii
                %update the diagonals of diagonal super elements
        
                hessian(3*i-2,3*i-2)       = hessian(3*i-2,3*i-2)+2*K_theta/(1-G^2)*dGdXi*dGdXi;
                hessian(3*i-1,3*i-1)       = hessian(3*i-1,3*i-1)+2*K_theta/(1-G^2)*dGdYi*dGdYi;
                hessian(3*i,3*i)           = hessian(3*i,3*i)+2*K_theta/(1-G^2)*dGdZi*dGdZi;                        
            
        		% update the off-diagonals of diagonal super elements			
            	hessian(3*i-2,3*i-1)       = hessian(3*i-2,3*i-1)+2*K_theta/(1-G^2)*dGdXi*dGdYi;
                hessian(3*i-2,3*i)         = hessian(3*i-2,3*i)+2*K_theta/(1-G^2)*dGdXi*dGdZi;
       			hessian(3*i-1,3*i-2)       = hessian(3*i-1,3*i-2)+2*K_theta/(1-G^2)*dGdYi*dGdXi;
           		hessian(3*i-1,3*i)         = hessian(3*i-1,3*i)+2*K_theta/(1-G^2)*dGdYi*dGdZi;
               	hessian(3*i,3*i-2)         = hessian(3*i,3*i-2)+2*K_theta/(1-G^2)*dGdZi*dGdXi;
                hessian(3*i,3*i-1)         = hessian(3*i,3*i-1)+2*K_theta/(1-G^2)*dGdZi*dGdYi;                        
                                
                %Hjj
                %update the diagonals of diagonal super elements
        
                hessian(3*j-2,3*j-2)       = hessian(3*j-2,3*j-2)+2*K_theta/(1-G^2)*dGdXj*dGdXj;
                hessian(3*j-1,3*j-1)       = hessian(3*j-1,3*j-1)+2*K_theta/(1-G^2)*dGdYj*dGdYj;
                hessian(3*j,3*j)           = hessian(3*j,3*j)+2*K_theta/(1-G^2)*dGdZj*dGdZj;                        
            
        		% update the off-diagonals of diagonal super elements			
            	hessian(3*j-2,3*j-1)       = hessian(3*j-2,3*j-1)+2*K_theta/(1-G^2)*dGdXj*dGdYj;
                hessian(3*j-2,3*j)         = hessian(3*j-2,3*j)+2*K_theta/(1-G^2)*dGdXj*dGdZj;
       			hessian(3*j-1,3*j-2)       = hessian(3*j-1,3*j-2)+2*K_theta/(1-G^2)*dGdYj*dGdXj;
           		hessian(3*j-1,3*j)         = hessian(3*j-1,3*j)+2*K_theta/(1-G^2)*dGdYj*dGdZj;
               	hessian(3*j,3*j-2)         = hessian(3*j,3*j-2)+2*K_theta/(1-G^2)*dGdZj*dGdXj;
                hessian(3*j,3*j-1)         = hessian(3*j,3*j-1)+2*K_theta/(1-G^2)*dGdZj*dGdYj;                                                                                        
                
                %Hkk
                %update the diagonals of diagonal super elements
        
                hessian(3*k-2,3*k-2)       = hessian(3*k-2,3*k-2)+2*K_theta/(1-G^2)*dGdXk*dGdXk;
                hessian(3*k-1,3*k-1)       = hessian(3*k-1,3*k-1)+2*K_theta/(1-G^2)*dGdYk*dGdYk;
                hessian(3*k,3*k)           = hessian(3*k,3*k)+2*K_theta/(1-G^2)*dGdZk*dGdZk;                        
            
        		% update the off-diagonals of diagonal super elements			
            	hessian(3*k-2,3*k-1)       = hessian(3*k-2,3*k-1)+2*K_theta/(1-G^2)*dGdXk*dGdYk;
                hessian(3*k-2,3*k)         = hessian(3*k-2,3*k)+2*K_theta/(1-G^2)*dGdXk*dGdZk;
       			hessian(3*k-1,3*k-2)       = hessian(3*k-1,3*k-2)+2*K_theta/(1-G^2)*dGdYk*dGdXk;
           		hessian(3*k-1,3*k)         = hessian(3*k-1,3*k)+2*K_theta/(1-G^2)*dGdYk*dGdZk;
               	hessian(3*k,3*k-2)         = hessian(3*k,3*k-2)+2*K_theta/(1-G^2)*dGdZk*dGdXk;
                hessian(3*k,3*k-1)         = hessian(3*k,3*k-1)+2*K_theta/(1-G^2)*dGdZk*dGdYk;                                                                                                                                                                    
    end
end


function hessian=thirdTerm(hessian,caArray,distance,numOfResidues,K_phi1,K_phi3)
K_phi=K_phi1/2+K_phi3*9/2;
for m=3:numOfResidues-1
    i=m-2;
    j=m-1;
    k=m;
    l=m+1;
        Xi=caArray(i,1);
        Yi=caArray(i,2);
        Zi=caArray(i,3);
        Xj=caArray(j,1);
        Yj=caArray(j,2);
        Zj=caArray(j,3);
        Xk=caArray(k,1);
        Yk=caArray(k,2);
        Zk=caArray(k,3);
        Xl=caArray(l,1);
        Yl=caArray(l,2);
        Zl=caArray(l,3);                                
        a=caArray(j,1:3)-caArray(i,1:3);
        b=caArray(k,1:3)-caArray(j,1:3);
        c=caArray(l,1:3)-caArray(k,1:3);
        v1=cross(a,b);
        v2=cross(b,c);
        lv1l=glength(v1);
        lv2l=glength(v2);
        G=dot(v1,v2)/(lv1l*lv2l);
        
        %dv1/dXi
        dv1dXi=[0 Zk-Zj Yj-Yk];
        %dv1/dYi
        dv1dYi=[Zj-Zk 0 Xk-Xj];
        %dv1/dZi
        dv1dZi=[Yk-Yj Xj-Xk 0];
        
        %dv1/dXj
        dv1dXj=[0 Zi-Zk Yk-Yi];
        %dv1/dYj
        dv1dYj=[Zk-Zi 0 Xi-Xk];
        %dv1/dZj
        dv1dZj=[Yi-Yk Xk-Xi 0];       
        
        
        %dv1/dXk
        dv1dXk=[0 Zj-Zi Yi-Yj];
        %dv2/dYk
        dv1dYk=[Zi-Zj 0 Xj-Xi];
        %dv2/dZk
        dv1dZk=[Yj-Yi Xi-Xj 0];

        %dv1/dXl
        dv1dXl=[0 0 0];
        %dv1/dYl
        dv1dYl=[0 0 0];
        %dv1/dZl
        dv1dZl=[0 0 0];
        
        
        %dv2/dXi
        dv2dXi=[0 0 0];
        %dv2/dYi
        dv2dYi=[0 0 0];
        %dv2/dZi
        dv2dZi=[0 0 0];        
                 
        %dv2/dXj
        dv2dXj=[0 Zl-Zk Yk-Yl];
        %dv2/dYj
        dv2dYj=[Zk-Zl 0 Xl-Xk];
        %dv2/dZj
        dv2dZj=[Yl-Yk Xk-Xl 0];
        
        %dv2/dXk
        dv2dXk=[0 Zj-Zl Yl-Yj];
        %dv2/dYk
        dv2dYk=[Zl-Zj 0 Xj-Xl];
        %dv2/dZk
        dv2dZk=[Yj-Yl Xl-Xj 0];
        
 
        %dv2/dXl
        dv2dXl=[0 Zk-Zj Yj-Yk];
        %dv2/dYl
        dv2dYl=[Zj-Zk 0 Xk-Xj];
        %dv2/dZl
        dv2dZl=[Yk-Yj Xj-Xk 0];
               
        K1=(Yj-Yi)*(Zk-Zj)-(Yk-Yj)*(Zj-Zi);
        K2=(Xk-Xj)*(Zj-Zi)-(Xj-Xi)*(Zk-Zj);
        K3=(Xj-Xi)*(Yk-Yj)-(Xk-Xj)*(Yj-Yi);
        
        %dlv1l/dXi
        dlv1ldXi=(2*K2*(Zk-Zj)+2*K3*(Yj-Yk))/(2*sqrt(K1^2+K2^2+K3^2));
        %dlv1l/dYi
        dlv1ldYi=(2*K1*(Zj-Zk)+2*K3*(Xk-Xj))/(2*sqrt(K1^2+K2^2+K3^2));
        %dlv1l/dZi
        dlv1ldZi=(2*K1*(Yk-Yj)+2*K2*(Xj-Xk))/(2*sqrt(K1^2+K2^2+K3^2));
                
        %dlv1ldXj        
        dlv1ldXj=(2*K2*(Zi-Zk)+2*K3*(Yk-Yi))/(2*sqrt(K1^2+K2^2+K3^2));
        %dlv1ldYj        
        dlv1ldYj=(2*K1*(Zk-Zi)+2*K3*(Xi-Xk))/(2*sqrt(K1^2+K2^2+K3^2));
        %dlv1ldZj        
        dlv1ldZj=(2*K1*(Yi-Yk)+2*K2*(Xk-Xi))/(2*sqrt(K1^2+K2^2+K3^2));
                
        %dlv1ldXk        
        dlv1ldXk=(2*K2*(Zj-Zi)+2*K3*(Yi-Yj))/(2*sqrt(K1^2+K2^2+K3^2));
        %dlv1ldYk      
        dlv1ldYk=(2*K1*(Zi-Zj)+2*K3*(Xj-Xi))/(2*sqrt(K1^2+K2^2+K3^2));
        %dlv1ldZk       
        dlv1ldZk=(2*K1*(Yj-Yi)+2*K2*(Xi-Xj))/(2*sqrt(K1^2+K2^2+K3^2));
                
        %dlv1ldXl       
        dlv1ldXl=0;
        %dlv1ldYl       
        dlv1ldYl=0;
        %dlv1ldZl       
        dlv1ldZl=0;
                                
        L1=(Yk-Yj)*(Zl-Zk)-(Yl-Yk)*(Zk-Zj);
        L2=(Xl-Xk)*(Zk-Zj)-(Xk-Xj)*(Zl-Zk);
        L3=(Xk-Xj)*(Yl-Yk)-(Xl-Xk)*(Yk-Yj);
        
        %dlv2l/dXi
        dlv2ldXi=0;
        %dlv2l/dYi
        dlv2ldYi=0;
        %dlv2l/dZi
        dlv2ldZi=0;
        
        %dlv2l/dXj
        dlv2ldXj=(2*L2*(Zl-Zk)+2*L3*(Yk-Yl))/(2*sqrt(L1^2+L2^2+L3^2));
        %dlv2l/dYj
        dlv2ldYj=(2*L1*(Zk-Zl)+2*L3*(Xl-Xk))/(2*sqrt(L1^2+L2^2+L3^2));
        %dlv2l/dZj
        dlv2ldZj=(2*L1*(Yl-Yk)+2*L2*(Xk-Xl))/(2*sqrt(L1^2+L2^2+L3^2));                        
        
        %dlv2l/dXk
        dlv2ldXk=(2*L2*(Zj-Zl)+2*L3*(Yl-Yj))/(2*sqrt(L1^2+L2^2+L3^2));
        %dlv2l/dYk
        dlv2ldYk=(2*L1*(Zl-Zj)+2*L3*(Xj-Xl))/(2*sqrt(L1^2+L2^2+L3^2));
        %dlv2l/dZk
        dlv2ldZk=(2*L1*(Yj-Yl)+2*L2*(Xl-Xj))/(2*sqrt(L1^2+L2^2+L3^2));

        %dlv2l/dXl
        dlv2ldXl=(2*L2*(Zk-Zj)+2*L3*(Yj-Yk))/(2*sqrt(L1^2+L2^2+L3^2));
        %dlv2l/dYl
        dlv2ldYl=(2*L1*(Zj-Zk)+2*L3*(Xk-Xj))/(2*sqrt(L1^2+L2^2+L3^2));
        %dlv2l/dZl
        dlv2ldZl=(2*L1*(Yk-Yj)+2*L2*(Xj-Xk))/(2*sqrt(L1^2+L2^2+L3^2));
      
        %dG/dXi
        dGdXi=((dot(dv1dXi,v2)+dot(dv2dXi,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldXi*lv2l+dlv2ldXi*lv1l))/(lv1l*lv2l)^2;
        %dG/dYi
        dGdYi=((dot(dv1dYi,v2)+dot(dv2dYi,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldYi*lv2l+dlv2ldYi*lv1l))/(lv1l*lv2l)^2;
        %dG/dZi
        dGdZi=((dot(dv1dZi,v2)+dot(dv2dZi,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldZi*lv2l+dlv2ldZi*lv1l))/(lv1l*lv2l)^2;
                
        %dG/dXj
        dGdXj=((dot(dv1dXj,v2)+dot(dv2dXj,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldXj*lv2l+dlv2ldXj*lv1l))/(lv1l*lv2l)^2;     
        %dG/dYj
        dGdYj=((dot(dv1dYj,v2)+dot(dv2dYj,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldYj*lv2l+dlv2ldYj*lv1l))/(lv1l*lv2l)^2;
        %dG/dZj
        dGdZj=((dot(dv1dZj,v2)+dot(dv2dZj,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldZj*lv2l+dlv2ldZj*lv1l))/(lv1l*lv2l)^2;

        %dG/dXk
        dGdXk=((dot(dv1dXk,v2)+dot(dv2dXk,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldXk*lv2l+dlv2ldXk*lv1l))/(lv1l*lv2l)^2;
        %dG/dYk
        dGdYk=((dot(dv1dYk,v2)+dot(dv2dYk,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldYk*lv2l+dlv2ldYk*lv1l))/(lv1l*lv2l)^2;
        %dG/dZk
        dGdZk=((dot(dv1dZk,v2)+dot(dv2dZk,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldZk*lv2l+dlv2ldZk*lv1l))/(lv1l*lv2l)^2;

         %dG/dXl
        dGdXl=((dot(dv1dXl,v2)+dot(dv2dXl,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldXl*lv2l+dlv2ldXl*lv1l))/(lv1l*lv2l)^2;
        %dG/dYl
        dGdYl=((dot(dv1dYl,v2)+dot(dv2dYl,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldYl*lv2l+dlv2ldYl*lv1l))/(lv1l*lv2l)^2;
        %dG/dZl
        dGdZl=((dot(dv1dZl,v2)+dot(dv2dZl,v1))*lv1l*lv2l-dot(v1,v2)*(dlv1ldZl*lv2l+dlv2ldZl*lv1l))/(lv1l*lv2l)^2;                       
                                               
        %Hij
        %d^2V/dXidXj  d^2V/dXidYj  d^2V/dXidZj
        %d^2V/dYidXj  d^2V/dYidYj  d^2V/dYidZj
        %d^2V/dZidXj  d^2V/dZidYj  d^2V/dZidZj               
        
        % diagonals of off-diagonal super elements             

        hessian(3*i-2,3*j-2)       = hessian(3*i-2,3*j-2)+(2*K_phi)/(1-G^2)*dGdXi*dGdXj;
    	hessian(3*i-1,3*j-1)       = hessian(3*i-1,3*j-1)+(2*K_phi)/(1-G^2)*dGdYi*dGdYj;
        hessian(3*i,3*j)           = hessian(3*i,3*j)+(2*K_phi)/(1-G^2)*dGdZi*dGdZj;

        % off-diagonals of off-diagonal super elements

    	hessian(3*i-2,3*j-1)       = hessian(3*i-2,3*j-1)+(2*K_phi)/(1-G^2)*dGdXi*dGdYj;
        hessian(3*i-2,3*j)         = hessian(3*i-2,3*j)+(2*K_phi)/(1-G^2)*dGdXi*dGdZj;
        hessian(3*i-1,3*j-2)       = hessian(3*i-1,3*j-2)+(2*K_phi)/(1-G^2)*dGdYi*dGdXj;
        hessian(3*i-1,3*j)         = hessian(3*i-1,3*j)+(2*K_phi)/(1-G^2)*dGdYi*dGdZj;
    	hessian(3*i,3*j-2)         = hessian(3*i,3*j-2)+(2*K_phi)/(1-G^2)*dGdZi*dGdXj;
        hessian(3*i,3*j-1)         = hessian(3*i,3*j-1)+(2*K_phi)/(1-G^2)*dGdZi*dGdYj;                                                                       
            
        %Hji
    	% diagonals of off-diagonal super elements            
        hessian(3*j-2,3*i-2)       = hessian(3*j-2,3*i-2)+(2*K_phi)/(1-G^2)*dGdXj*dGdXi;
        hessian(3*j-1,3*i-1)       = hessian(3*j-1,3*i-1)+(2*K_phi)/(1-G^2)*dGdYj*dGdYi;
    	hessian(3*j,3*i)           = hessian(3*j,3*i)+(2*K_phi)/(1-G^2)*dGdZj*dGdZi;

        % off-diagonals of off-diagonal super elements
    
        hessian(3*j-2,3*i-1)       = hessian(3*j-2,3*i-1)+(2*K_phi)/(1-G^2)*dGdXj*dGdYi;
    	hessian(3*j-2,3*i)         = hessian(3*j-2,3*i)+(2*K_phi)/(1-G^2)*dGdXj*dGdZi;
        hessian(3*j-1,3*i-2)       = hessian(3*j-1,3*i-2)+(2*K_phi)/(1-G^2)*dGdYj*dGdXi;
    	hessian(3*j-1,3*i)         = hessian(3*j-1,3*i)+(2*K_phi)/(1-G^2)*dGdYj*dGdZi;
        hessian(3*j,3*i-2)         = hessian(3*j,3*i-2)+(2*K_phi)/(1-G^2)*dGdZj*dGdXi;
    	hessian(3*j,3*i-1)         = hessian(3*j,3*i-1)+(2*K_phi)/(1-G^2)*dGdZj*dGdYi;                                                                                                   
                
        %Hil
            
        % diagonals of off-diagonal super elements

        hessian(3*i-2,3*l-2)       = hessian(3*i-2,3*l-2)+(2*K_phi)/(1-G^2)*dGdXi*dGdXl;
    	hessian(3*i-1,3*l-1)       = hessian(3*i-1,3*l-1)+(2*K_phi)/(1-G^2)*dGdYi*dGdYl;
        hessian(3*i,3*l)           = hessian(3*i,3*l)+(2*K_phi)/(1-G^2)*dGdZi*dGdZl;

        % off-diagonals of off-diagonal super elements

    	hessian(3*i-2,3*l-1)       = hessian(3*i-2,3*l-1)+(2*K_phi)/(1-G^2)*dGdXi*dGdYl;
        hessian(3*i-2,3*l)         = hessian(3*i-2,3*l)+(2*K_phi)/(1-G^2)*dGdXi*dGdZl;
        hessian(3*i-1,3*l-2)       = hessian(3*i-1,3*l-2)+(2*K_phi)/(1-G^2)*dGdYi*dGdXl;
        hessian(3*i-1,3*l)         = hessian(3*i-1,3*l)+(2*K_phi)/(1-G^2)*dGdYi*dGdZl;
    	hessian(3*i,3*l-2)         = hessian(3*i,3*l-2)+(2*K_phi)/(1-G^2)*dGdZi*dGdXl;
        hessian(3*i,3*l-1)         = hessian(3*i,3*l-1)+(2*K_phi)/(1-G^2)*dGdZi*dGdYl;                                                                       
            
        %Hli
    	% diagonals of off-diagonal super elements            
        hessian(3*l-2,3*i-2)       = hessian(3*l-2,3*i-2)+(2*K_phi)/(1-G^2)*dGdXl*dGdXi;
        hessian(3*l-1,3*i-1)       = hessian(3*l-1,3*i-1)+(2*K_phi)/(1-G^2)*dGdYl*dGdYi;
    	hessian(3*l,3*i)           = hessian(3*l,3*i)+(2*K_phi)/(1-G^2)*dGdZl*dGdZi;

        % off-diagonals of off-diagonal super elements
    
        hessian(3*l-2,3*i-1)       = hessian(3*l-2,3*i-1)+(2*K_phi)/(1-G^2)*dGdXl*dGdYi;
    	hessian(3*l-2,3*i)         = hessian(3*l-2,3*i)+(2*K_phi)/(1-G^2)*dGdXl*dGdZi;
        hessian(3*l-1,3*i-2)       = hessian(3*l-1,3*i-2)+(2*K_phi)/(1-G^2)*dGdYl*dGdXi;
    	hessian(3*l-1,3*i)         = hessian(3*l-1,3*i)+(2*K_phi)/(1-G^2)*dGdYl*dGdZi;
        hessian(3*l,3*i-2)         = hessian(3*l,3*i-2)+(2*K_phi)/(1-G^2)*dGdZl*dGdXi;
    	hessian(3*l,3*i-1)         = hessian(3*l,3*i-1)+(2*K_phi)/(1-G^2)*dGdZl*dGdYi;                                                                                                   
        
        %Hkj        
        % diagonals of off-diagonal super elements

        hessian(3*k-2,3*j-2)       = hessian(3*k-2,3*j-2)+(2*K_phi)/(1-G^2)*dGdXk*dGdXj;
    	hessian(3*k-1,3*j-1)       = hessian(3*k-1,3*j-1)+(2*K_phi)/(1-G^2)*dGdYk*dGdYj;
        hessian(3*k,3*j)           = hessian(3*k,3*j)+(2*K_phi)/(1-G^2)*dGdZk*dGdZj;

        % off-diagonals of off-diagonal super elements

    	hessian(3*k-2,3*j-1)       = hessian(3*k-2,3*j-1)+(2*K_phi)/(1-G^2)*dGdXk*dGdYj;
        hessian(3*k-2,3*j)         = hessian(3*k-2,3*j)+(2*K_phi)/(1-G^2)*dGdXk*dGdZj;
        hessian(3*k-1,3*j-2)       = hessian(3*k-1,3*j-2)+(2*K_phi)/(1-G^2)*dGdYk*dGdXj;
        hessian(3*k-1,3*j)         = hessian(3*k-1,3*j)+(2*K_phi)/(1-G^2)*dGdYk*dGdZj;
    	hessian(3*k,3*j-2)         = hessian(3*k,3*j-2)+(2*K_phi)/(1-G^2)*dGdZk*dGdXj;
        hessian(3*k,3*j-1)         = hessian(3*k,3*j-1)+(2*K_phi)/(1-G^2)*dGdZk*dGdYj;                                                                       
            
        %Hjk
    	% diagonals of off-diagonal super elements
        hessian(3*j-2,3*k-2)       = hessian(3*j-2,3*k-2)+(2*K_phi)/(1-G^2)*dGdXj*dGdXk;
        hessian(3*j-1,3*k-1)       = hessian(3*j-1,3*k-1)+(2*K_phi)/(1-G^2)*dGdYj*dGdYk;
    	hessian(3*j,3*k)           = hessian(3*j,3*k)+(2*K_phi)/(1-G^2)*dGdZj*dGdZk;

        % off-diagonals of off-diagonal super elements
    
        hessian(3*j-2,3*k-1)       = hessian(3*j-2,3*k-1)+(2*K_phi)/(1-G^2)*dGdXj*dGdYk;
    	hessian(3*j-2,3*k)         = hessian(3*j-2,3*k)+(2*K_phi)/(1-G^2)*dGdXj*dGdZk;
        hessian(3*j-1,3*k-2)       = hessian(3*j-1,3*k-2)+(2*K_phi)/(1-G^2)*dGdYj*dGdXk;
    	hessian(3*j-1,3*k)         = hessian(3*j-1,3*k)+(2*K_phi)/(1-G^2)*dGdYj*dGdZk;
        hessian(3*j,3*k-2)         = hessian(3*j,3*k-2)+(2*K_phi)/(1-G^2)*dGdZj*dGdXk;
    	hessian(3*j,3*k-1)         = hessian(3*j,3*k-1)+(2*K_phi)/(1-G^2)*dGdZj*dGdYk;                                                                                                   
        
        %Hik            
        % diagonals of off-diagonal super elements

        hessian(3*i-2,3*k-2)       = hessian(3*i-2,3*k-2)+(2*K_phi)/(1-G^2)*dGdXi*dGdXk;
    	hessian(3*i-1,3*k-1)       = hessian(3*i-1,3*k-1)+(2*K_phi)/(1-G^2)*dGdYi*dGdYk;
        hessian(3*i,3*k)           = hessian(3*i,3*k)+(2*K_phi)/(1-G^2)*dGdZi*dGdZk;

        % off-diagonals of off-diagonal super elements

    	hessian(3*i-2,3*k-1)       = hessian(3*i-2,3*k-1)+(2*K_phi)/(1-G^2)*dGdXi*dGdYk;
        hessian(3*i-2,3*k)         = hessian(3*i-2,3*k)+(2*K_phi)/(1-G^2)*dGdXi*dGdZk;
        hessian(3*i-1,3*k-2)       = hessian(3*i-1,3*k-2)+(2*K_phi)/(1-G^2)*dGdYi*dGdXk;
        hessian(3*i-1,3*k)         = hessian(3*i-1,3*k)+(2*K_phi)/(1-G^2)*dGdYi*dGdZk;
    	hessian(3*i,3*k-2)         = hessian(3*i,3*k-2)+(2*K_phi)/(1-G^2)*dGdZi*dGdXk;
        hessian(3*i,3*k-1)         = hessian(3*i,3*k-1)+(2*K_phi)/(1-G^2)*dGdZi*dGdYk;                                                                       
            
        %Hki
    	% diagonals of off-diagonal super elements            
        hessian(3*k-2,3*i-2)       = hessian(3*k-2,3*i-2)+(2*K_phi)/(1-G^2)*dGdXk*dGdXi;
        hessian(3*k-1,3*i-1)       = hessian(3*k-1,3*i-1)+(2*K_phi)/(1-G^2)*dGdYk*dGdYi;
    	hessian(3*k,3*i)           = hessian(3*k,3*i)+(2*K_phi)/(1-G^2)*dGdZk*dGdZi;

        % off-diagonals of off-diagonal super elements
    
        hessian(3*k-2,3*i-1)       = hessian(3*k-2,3*i-1)+(2*K_phi)/(1-G^2)*dGdXk*dGdYi;
    	hessian(3*k-2,3*i)         = hessian(3*k-2,3*i)+(2*K_phi)/(1-G^2)*dGdXk*dGdZi;
        hessian(3*k-1,3*i-2)       = hessian(3*k-1,3*i-2)+(2*K_phi)/(1-G^2)*dGdYk*dGdXi;
    	hessian(3*k-1,3*i)         = hessian(3*k-1,3*i)+(2*K_phi)/(1-G^2)*dGdYk*dGdZi;
        hessian(3*k,3*i-2)         = hessian(3*k,3*i-2)+(2*K_phi)/(1-G^2)*dGdZk*dGdXi;
    	hessian(3*k,3*i-1)         = hessian(3*k,3*i-1)+(2*K_phi)/(1-G^2)*dGdZk*dGdYi;                                                                                                   
        
        %Hlj
        % diagonals of off-diagonal super elements

        hessian(3*l-2,3*j-2)       = hessian(3*l-2,3*j-2)+(2*K_phi)/(1-G^2)*dGdXl*dGdXj;
    	hessian(3*l-1,3*j-1)       = hessian(3*l-1,3*j-1)+(2*K_phi)/(1-G^2)*dGdYl*dGdYj;
        hessian(3*l,3*j)           = hessian(3*l,3*j)+(2*K_phi)/(1-G^2)*dGdZl*dGdZj;

        % off-diagonals of off-diagonal super elements

    	hessian(3*l-2,3*j-1)       = hessian(3*l-2,3*j-1)+(2*K_phi)/(1-G^2)*dGdXl*dGdYj;
        hessian(3*l-2,3*j)         = hessian(3*l-2,3*j)+(2*K_phi)/(1-G^2)*dGdXl*dGdZj;
        hessian(3*l-1,3*j-2)       = hessian(3*l-1,3*j-2)+(2*K_phi)/(1-G^2)*dGdYl*dGdXj;
        hessian(3*l-1,3*j)         = hessian(3*l-1,3*j)+(2*K_phi)/(1-G^2)*dGdYl*dGdZj;
    	hessian(3*l,3*j-2)         = hessian(3*l,3*j-2)+(2*K_phi)/(1-G^2)*dGdZl*dGdXj;
        hessian(3*l,3*j-1)         = hessian(3*l,3*j-1)+(2*K_phi)/(1-G^2)*dGdZl*dGdYj;                                                                       
            
        %Hjl
    	% diagonals of off-diagonal super elements            
        hessian(3*j-2,3*l-2)       = hessian(3*j-2,3*l-2)+(2*K_phi)/(1-G^2)*dGdXj*dGdXl;
        hessian(3*j-1,3*l-1)       = hessian(3*j-1,3*l-1)+(2*K_phi)/(1-G^2)*dGdYj*dGdYl;
    	hessian(3*j,3*l)           = hessian(3*j,3*l)+(2*K_phi)/(1-G^2)*dGdZj*dGdZl;

        % off-diagonals of off-diagonal super elements
    
        hessian(3*j-2,3*l-1)       = hessian(3*j-2,3*l-1)+(2*K_phi)/(1-G^2)*dGdXj*dGdYl;
    	hessian(3*j-2,3*l)         = hessian(3*j-2,3*l)+(2*K_phi)/(1-G^2)*dGdXj*dGdZl;
        hessian(3*j-1,3*l-2)       = hessian(3*j-1,3*l-2)+(2*K_phi)/(1-G^2)*dGdYj*dGdXl;
    	hessian(3*j-1,3*l)         = hessian(3*j-1,3*l)+(2*K_phi)/(1-G^2)*dGdYj*dGdZl;
        hessian(3*j,3*l-2)         = hessian(3*j,3*l-2)+(2*K_phi)/(1-G^2)*dGdZj*dGdXl;
    	hessian(3*j,3*l-1)         = hessian(3*j,3*l-1)+(2*K_phi)/(1-G^2)*dGdZj*dGdYl;                                                                                                   
        

        %Hlk
        % diagonals of off-diagonal super elements

        hessian(3*l-2,3*k-2)       = hessian(3*l-2,3*k-2)+(2*K_phi)/(1-G^2)*dGdXl*dGdXk;
    	hessian(3*l-1,3*k-1)       = hessian(3*l-1,3*k-1)+(2*K_phi)/(1-G^2)*dGdYl*dGdYk;
        hessian(3*l,3*k)           = hessian(3*l,3*k)+(2*K_phi)/(1-G^2)*dGdZl*dGdZk;

        % off-diagonals of off-diagonal super elements

    	hessian(3*l-2,3*k-1)       = hessian(3*l-2,3*k-1)+(2*K_phi)/(1-G^2)*dGdXl*dGdYk;
        hessian(3*l-2,3*k)         = hessian(3*l-2,3*k)+(2*K_phi)/(1-G^2)*dGdXl*dGdZk;
        hessian(3*l-1,3*k-2)       = hessian(3*l-1,3*k-2)+(2*K_phi)/(1-G^2)*dGdYl*dGdXk;
        hessian(3*l-1,3*k)         = hessian(3*l-1,3*k)+(2*K_phi)/(1-G^2)*dGdYl*dGdZk;
    	hessian(3*l,3*k-2)         = hessian(3*l,3*k-2)+(2*K_phi)/(1-G^2)*dGdZl*dGdXk;
        hessian(3*l,3*k-1)         = hessian(3*l,3*k-1)+(2*K_phi)/(1-G^2)*dGdZl*dGdYk;                                                                       
            
        %Hkl
    	% diagonals of off-diagonal super elements            
        hessian(3*k-2,3*l-2)       = hessian(3*k-2,3*l-2)+(2*K_phi)/(1-G^2)*dGdXk*dGdXl;
        hessian(3*k-1,3*l-1)       = hessian(3*k-1,3*l-1)+(2*K_phi)/(1-G^2)*dGdYk*dGdYl;
    	hessian(3*k,3*l)           = hessian(3*k,3*l)+(2*K_phi)/(1-G^2)*dGdZk*dGdZl;

        % off-diagonals of off-diagonal super elements
    
        hessian(3*k-2,3*l-1)       = hessian(3*k-2,3*l-1)+(2*K_phi)/(1-G^2)*dGdXk*dGdYl;
    	hessian(3*k-2,3*l)         = hessian(3*k-2,3*l)+(2*K_phi)/(1-G^2)*dGdXk*dGdZl;
        hessian(3*k-1,3*l-2)       = hessian(3*k-1,3*l-2)+(2*K_phi)/(1-G^2)*dGdYk*dGdXl;
    	hessian(3*k-1,3*l)         = hessian(3*k-1,3*l)+(2*K_phi)/(1-G^2)*dGdYk*dGdZl;
        hessian(3*k,3*l-2)         = hessian(3*k,3*l-2)+(2*K_phi)/(1-G^2)*dGdZk*dGdXl;        
        hessian(3*k,3*l-1)         = hessian(3*k,3*l-1)+(2*K_phi)/(1-G^2)*dGdZk*dGdYl;
        
        %Hii
        %update the diagonals of diagonal super elements
        
        hessian(3*i-2,3*i-2)       = hessian(3*i-2,3*i-2)+2*K_phi/(1-G^2)*dGdXi*dGdXi;
        hessian(3*i-1,3*i-1)       = hessian(3*i-1,3*i-1)+2*K_phi/(1-G^2)*dGdYi*dGdYi;
        hessian(3*i,3*i)           = hessian(3*i,3*i)+2*K_phi/(1-G^2)*dGdZi*dGdZi;                       
            
        % update the off-diagonals of diagonal super elements			
        hessian(3*i-2,3*i-1)       = hessian(3*i-2,3*i-1)+2*K_phi/(1-G^2)*dGdXi*dGdYi;
        hessian(3*i-2,3*i)         = hessian(3*i-2,3*i)+2*K_phi/(1-G^2)*dGdXi*dGdZi;
       	hessian(3*i-1,3*i-2)       = hessian(3*i-1,3*i-2)+2*K_phi/(1-G^2)*dGdYi*dGdXi;
        hessian(3*i-1,3*i)         = hessian(3*i-1,3*i)+2*K_phi/(1-G^2)*dGdYi*dGdZi;
        hessian(3*i,3*i-2)         = hessian(3*i,3*i-2)+2*K_phi/(1-G^2)*dGdZi*dGdXi;
        hessian(3*i,3*i-1)         = hessian(3*i,3*i-1)+2*K_phi/(1-G^2)*dGdZi*dGdYi;                        
        
        %Hjj
        %update the diagonals of diagonal super elements
        
        hessian(3*j-2,3*j-2)       = hessian(3*j-2,3*j-2)+2*K_phi/(1-G^2)*dGdXj*dGdXj;
        hessian(3*j-1,3*j-1)       = hessian(3*j-1,3*j-1)+2*K_phi/(1-G^2)*dGdYj*dGdYj;
        hessian(3*j,3*j)           = hessian(3*j,3*j)+2*K_phi/(1-G^2)*dGdZj*dGdZj;                       
            
        % update the off-diagonals of diagonal super elements			
        hessian(3*j-2,3*j-1)       = hessian(3*j-2,3*j-1)+2*K_phi/(1-G^2)*dGdXj*dGdYj;
        hessian(3*j-2,3*j)         = hessian(3*j-2,3*j)+2*K_phi/(1-G^2)*dGdXj*dGdZj;
       	hessian(3*j-1,3*j-2)       = hessian(3*j-1,3*j-2)+2*K_phi/(1-G^2)*dGdYj*dGdXj;
        hessian(3*j-1,3*j)         = hessian(3*j-1,3*j)+2*K_phi/(1-G^2)*dGdYj*dGdZj;
        hessian(3*j,3*j-2)         = hessian(3*j,3*j-2)+2*K_phi/(1-G^2)*dGdZj*dGdXj;
        hessian(3*j,3*j-1)         = hessian(3*j,3*j-1)+2*K_phi/(1-G^2)*dGdZj*dGdYj;                        
        
        %Hkk
        %update the diagonals of diagonal super elements
        
        hessian(3*k-2,3*k-2)       = hessian(3*k-2,3*k-2)+2*K_phi/(1-G^2)*dGdXk*dGdXk;
        hessian(3*k-1,3*k-1)       = hessian(3*k-1,3*k-1)+2*K_phi/(1-G^2)*dGdYk*dGdYk;
        hessian(3*k,3*k)           = hessian(3*k,3*k)+2*K_phi/(1-G^2)*dGdZk*dGdZk;                       
            
        % update the off-diagonals of diagonal super elements			
        hessian(3*k-2,3*k-1)       = hessian(3*k-2,3*k-1)+2*K_phi/(1-G^2)*dGdXk*dGdYk;
        hessian(3*k-2,3*k)         = hessian(3*k-2,3*k)+2*K_phi/(1-G^2)*dGdXk*dGdZk;
       	hessian(3*k-1,3*k-2)       = hessian(3*k-1,3*k-2)+2*K_phi/(1-G^2)*dGdYk*dGdXk;
        hessian(3*k-1,3*k)         = hessian(3*k-1,3*k)+2*K_phi/(1-G^2)*dGdYk*dGdZk;
        hessian(3*k,3*k-2)         = hessian(3*k,3*k-2)+2*K_phi/(1-G^2)*dGdZk*dGdXk;
        hessian(3*k,3*k-1)         = hessian(3*k,3*k-1)+2*K_phi/(1-G^2)*dGdZk*dGdYk;                        
        
        %Hll
        %update the diagonals of diagonal super elements
        
        hessian(3*l-2,3*l-2)       = hessian(3*l-2,3*l-2)+2*K_phi/(1-G^2)*dGdXl*dGdXl;
        hessian(3*l-1,3*l-1)       = hessian(3*l-1,3*l-1)+2*K_phi/(1-G^2)*dGdYl*dGdYl;
        hessian(3*l,3*l)           = hessian(3*l,3*l)+2*K_phi/(1-G^2)*dGdZl*dGdZl;                       
            
        % update the off-diagonals of diagonal super elements			
        hessian(3*l-2,3*l-1)       = hessian(3*l-2,3*l-1)+2*K_phi/(1-G^2)*dGdXl*dGdYl;
        hessian(3*l-2,3*l)         = hessian(3*l-2,3*l)+2*K_phi/(1-G^2)*dGdXl*dGdZl;
       	hessian(3*l-1,3*l-2)       = hessian(3*l-1,3*l-2)+2*K_phi/(1-G^2)*dGdYl*dGdXl;
        hessian(3*l-1,3*l)         = hessian(3*l-1,3*l)+2*K_phi/(1-G^2)*dGdYl*dGdZl;
        hessian(3*l,3*l-2)         = hessian(3*l,3*l-2)+2*K_phi/(1-G^2)*dGdZl*dGdXl;
        hessian(3*l,3*l-1)         = hessian(3*l,3*l-1)+2*K_phi/(1-G^2)*dGdZl*dGdYl;                                                
end
end



function hessian=fourthTerm(hessian,caArray,distance,numOfResidues,Epsilon)
% derive the hessian of the first term (off diagonal)
for i=1:numOfResidues;
    for j=1:numOfResidues;
       if abs(i-j)>3   
           
          	bx=caArray(i,1) - caArray(j,1);
       		by=caArray(i,2) - caArray(j,2);
           	bz=caArray(i,3) - caArray(j,3);
            distijsqr=distance(i,j)^4;
            
			% diagonals of off-diagonal super elements (1st term)

			hessian(3*i-2,3*j-2)       = hessian(3*i-2,3*j-2)-120*Epsilon*bx*bx/distijsqr;
			hessian(3*i-1,3*j-1)       = hessian(3*i-1,3*j-1)-120*Epsilon*by*by/distijsqr;
			hessian(3*i,3*j)           = hessian(3*i,3*j)-120*Epsilon*bz*bz/distijsqr;

			% off-diagonals of off-diagonal super elements (1st term)

			hessian(3*i-2,3*j-1)       = hessian(3*i-2,3*j-1)-120*Epsilon*bx*by/distijsqr;
			hessian(3*i-2,3*j)         = hessian(3*i-2,3*j)-120*Epsilon*bx*bz/distijsqr;
			hessian(3*i-1,3*j-2)       = hessian(3*i-1,3*j-2)-120*Epsilon*by*bx/distijsqr;
			hessian(3*i-1,3*j)         = hessian(3*i-1,3*j)-120*Epsilon*by*bz/distijsqr;
			hessian(3*i,3*j-2)         = hessian(3*i,3*j-2)-120*Epsilon*bx*bz/distijsqr;
			hessian(3*i,3*j-1)         = hessian(3*i,3*j-1)-120*Epsilon*by*bz/distijsqr;                                                
            
           
            %Hii
            %update the diagonals of diagonal super elements
            
            hessian(3*i-2,3*i-2)       = hessian(3*i-2,3*i-2)+120*Epsilon*bx*bx/distijsqr;
            hessian(3*i-1,3*i-1)       = hessian(3*i-1,3*i-1)+120*Epsilon*by*by/distijsqr;
            hessian(3*i,3*i)           = hessian(3*i,3*i)+120*Epsilon*bz*bz/distijsqr;                        
            
    		% update the off-diagonals of diagonal super elements			
      		hessian(3*i-2,3*i-1)       = hessian(3*i-2,3*i-1)+120*Epsilon*bx*by/distijsqr;
           	hessian(3*i-2,3*i)         = hessian(3*i-2,3*i)+120*Epsilon*bx*bz/distijsqr;
   			hessian(3*i-1,3*i-2)       = hessian(3*i-1,3*i-2)+120*Epsilon*by*bx/distijsqr;
       		hessian(3*i-1,3*i)         = hessian(3*i-1,3*i)+120*Epsilon*by*bz/distijsqr;
           	hessian(3*i,3*i-2)         = hessian(3*i,3*i-2)+120*Epsilon*bz*bx/distijsqr;
   			hessian(3*i,3*i-1)         = hessian(3*i,3*i-1)+120*Epsilon*bz*by/distijsqr;                                                                
       end
    end
end

end
