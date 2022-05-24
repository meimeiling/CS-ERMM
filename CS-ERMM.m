function [W,fValue,Omega,Corr]=CS-ERMM(data,SNP_Exp,Z,beta,lambda,gamma,nu)


%% variance size of input matrix
    m=length(SNP_Exp); % m for task number, n for sample number
    d=size(data{1,1},2); % d for feature number 
    
%% Initialization
    W=zeros(d,m); % initialize the original W


%% Compute W
    [W,fValue,Omega,Corr]=myAPG(W,data,SNP_Exp,Z,beta,lambda,gamma,nu);   

%     if (nargin==2)%nargin - 输入变量个数的函数
%        predictions=W'*testData;
%     end
end

%% comput W through APG algorithm
function [W,fValue,Omega,m_Cor]=myAPG(W_initial,data,SNP_Exp,Z,beta,lambda,gamma,nu)
% warning off 
        %%W-->d*m  d is dimensional, m is task number
      m=length(SNP_Exp); % m for task number, n for sample number
      d=size(data{1,1},2); % d for feature number 
      epsilon=10^(-7);
      max_iteration=1000;
      L=1;t=0;tau=1; W=W_initial;
      Omega=eye(m)/m; % initialize the original Omega
      m_Cor=[];
 %     beta=0.1;
for iter=1:max_iteration

    %%%APG 优化W
    W_h=W;
    F_wh=0;
    gradient_f=zeros([d m]);
    for i=1:m
        gradient_f(:,i)= data{1,i}'* (data{1,i}* W_h(:,i)-SNP_Exp{1,i}')/(m)+beta*data{1,i}'* (data{1,i}* W_h(:,i)-Z{1,i}')/(m);
        F_wh = F_wh+norm(SNP_Exp{1,i}-W_h(:,i)'*data{1,i}',2)^2/(2*m)+beta*norm(Z{1,i}-W_h(:,i)'*data{1,i}',2)^2/(2*m);
    end
     G=[];
    for i=1:d
%        ;
        Vi=W_h(i,:)/max(nu,norm(W_h(i,:),2));
        G=[G;sum(abs(W_h(i,:)))*Vi];   % sum(i=1,...,d, abs(Wi)*Vi)
        F_wh =F_wh+lambda*sum( abs(W_h(i,:)) )^2/2;
    end

    G_f = gradient_f +lambda*G + gamma*W_h*Omega^(-1);%关系项不可导
    F_wh = F_wh + gamma * trace(W_h*Omega^(-1)*W_h')/2;

    while(true)
        W_h_new=W_h-G_f/L;
        h_new = 0;
        for i=1:m
            h_new = h_new+norm( SNP_Exp{1,i}-W_h_new(:,i)'*data{1,i}',2 )^2/(2*m)+beta*norm( Z{1,i}-W_h_new(:,i)'*data{1,i}',2 )^2/(2*m);%*size(data{1,i},1));;;
        end
        for i=1:d
            h_new = h_new+ lambda*sum( abs(W_h_new(i,:)) )^2/2;
        end
        h_new = h_new + gamma * trace(W_h_new*Omega^(-1)*W_h_new')/2;

        
        Q = F_wh+(W_h_new(:)-W_h(:))'*G_f(:)+L/2*sum((W_h_new(:)-W_h(:)).^2);

        if (h_new<=Q)
            break;
        else
            L=2*L;
        end
    end
    tau_new=2/(t+3);

     W=W_h_new+(1-tau)/tau*tau_new*(W_h_new-W_h);
%     L



    %%compute the Omega
    Omega = (W'*W)^(0.5)/trace((W'*W)^(0.5));

    Omega = real(Omega);

    % function value
    fValue(iter,1)=h_new;
        if (iter>10 && (abs(fValue(iter,1)-fValue(iter-1,1))/abs(fValue(iter-1,1))<epsilon))
%         flag=1;
            break;
        end
    tau = tau_new;
    t= t+1;

    

    
    
    



end
     
end