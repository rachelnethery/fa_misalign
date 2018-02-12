function [ avg_run_time ] = joint_spatial_fa(Y,Z,fipssb,fipslb,weights,R,burn_in,reps,M,varargin )
% Y contains the observed variables (in the columns of a matrix) recorded at
% small blocks
% Z contains the observed variables recorded at large blocks
% fipssb contains the fips codes for the small blocks (these codes correspond to
% large block level, so all small blocks in large block k would have value
% k for this variable)
% fipslb contains the fips codes for the large blocks
% weights contains the weights for each small block
% R is the adjacency matrix for the small blocks
% burn-in is the number of burn-in samples to be discarded
% reps is the number of Gibbs samples to take
% M is the number of latent factors to model
% varargin:
    % nsampfile is the number of samples to save in each output file
    % nkeepeta is the number of elements of eta to save in the output
    % (useful when the number of elements is very large)
    % sigmasq_lambda1 is the prior variance for elements of lambda1
    % sigmasq_lambda2 is the prior variance for elements of lambda2
    % alpha1 is the parameter in the IG prior distribution for sigma1
    % beta1 is the parameter in the IG prior distribution for sigma1
    % alpha2 is the parameter in the IG prior distribution for sigma2
    % beta2 is the parameter in the IG prior distribution for sigma2
    % sigma_m_phi is the tuning parameter in the Metropolis step for phi

% NT is the number of small blocks
NT=size(Y,1);
% N is the number of large blocks
N=size(Z,1);
% P1 is the number of observed variables recorded at small blocks
P1=size(Y,2);
% P2 is the number of observed variables recorded at large blocks
P2=size(Z,2);

% get the min and max eigenvalues of R
eigenvalsR=eig(R);
lb=1/min(eigenvalsR);
ub=1/max(eigenvalsR);

% set default constant values
nsampfile=1000;
nkeepeta=NT*M;
sigmasq_lambda1=10000;
sigmasq_lambda2=10000;
alpha1=1/1000;
beta1=1/1000;
alpha2=1/1000;
beta2=1/1000;
sigma_m_phi=.25;

% over-ride default constant values if they are specified by the user
for h=1:2:(length(varargin)-1)
    switch lower(varargin{h})
         case 'nsampfile'
             nsampfile=varargin{h+1};
         case 'nkeepeta'
             nkeepeta=varargin{h+1};
         case 'sigmasq_lambda1'
             sigmasq_lambda1= varargin{h+1};
         case 'sigmasq_lambda2'
             sigmasq_lambda2= varargin{h+1};
         case 'alpha1'
             alpha1=varargin{h+1};
         case 'beta1'
             beta1=varargin{h+1};
         case 'alpha2'
             alpha2=varargin{h+1};
         case 'beta2'
             beta2=varargin{h+1};
        case 'sigma_m_phi'
             sigma_m_phi=varargin{h+1};
     end
end
    
% set starting values for all parameters
lambda1=ones(P1,M);
lambda2=ones(P2,M);
for p1=1:P1
    for m=1:M
        if (p1<m) lambda1(p1,m)=0;
        end
    end
end
for p2=1:P2
    for m=1:M
        if (p2<m) lambda2(p2,m)=0;
        end
    end
end
sigma1=eye(P1);
sigma2=eye(P2);
eta=ones(M,NT);
eta_tilde=reshape(eta,[1 M*NT]);
eta_weight=bsxfun(@times,eta,weights');
eta_ws=[];
for i=1:N
    locs=find(fipssb(:,1)==fipslb(i,1));
    eta_ws=[eta_ws sum(eta_weight(:,locs),2)];
end
phi=mean([lb ub]);
accept_phi=0;

% initiate matrices used to store samples (nsampfile at a time)
lambda1_save=zeros(nsampfile,P1*M);
lambda2_save=zeros(nsampfile,P2*M);
sigma1_save=zeros(nsampfile,P1);
sigma2_save=zeros(nsampfile,P2);
eta_save=zeros(nsampfile,nkeepeta);
phi_save=zeros(nsampfile,1);
accept_phi_save=zeros(nsampfile,1);

% if not keeping all elements of eta, choose a random sample of them to
% keep of size nkeepeta
keepeta=randsample(NT*M,nkeepeta)';

% BEGIN THE GIBBS SAMPLER
run_times=[];
nsaves=0;

for g=1:(burn_in+reps)
    % start timer
    %tic;
    
    % start an indicator for saving sampled values
    indsave=g-burn_in-(nsaves*nsampfile);
    
    tic;
    % SAMPLE LAMBDA1
    for p1=1:P1
        for m=1:M
            if p1<m lambda1(p1,m)=0;
                
            elseif p1==m
                if M==1
                    Vlambda=(sigma1(p1,p1).*sigmasq_lambda1)./((sigmasq_lambda1.*sum(eta(m,:).^2))+sigma1(p1,p1));
                    Elambda=Vlambda.*(1./sigma1(p1,p1)).*sum(Y(:,p1)'.*eta(m,:));
                    lambda1(p1,m)=Elambda+(sqrt(Vlambda)*trandn(0-Elambda/sqrt(Vlambda),inf));
                else
                    lambdad=lambda1;
                    lambdad(:,m)=[];
                    etad=eta;
                    etad(m,:)=[];
                    Vlambda=(sigma1(p1,p1).*sigmasq_lambda1)./((sigmasq_lambda1.*sum(eta(m,:).^2))+sigma1(p1,p1));
                    Elambda=Vlambda.*(1./sigma1(p1,p1)).*sum((Y(:,p1)'-(lambdad(p1,:)*etad)).*eta(m,:));
                    lambda1(p1,m)=Elambda+(sqrt(Vlambda)*trandn(0-Elambda/sqrt(Vlambda),inf));
                end
                
            else
                if M==1
                    Vlambda=(sigma1(p1,p1).*sigmasq_lambda1)./((sigmasq_lambda1.*sum(eta(m,:).^2))+sigma1(p1,p1));
                    Elambda=Vlambda.*(1./sigma1(p1,p1)).*sum(Y(:,p1)'.*eta(m,:));
                    lambda1(p1,m)=normrnd(Elambda,sqrt(Vlambda),1,1);
                else
                    lambdad=lambda1;
                    lambdad(:,m)=[];
                    etad=eta;
                    etad(m,:)=[];
                    Vlambda=(sigma1(p1,p1).*sigmasq_lambda1)./((sigmasq_lambda1.*sum(eta(m,:).^2))+sigma1(p1,p1));
                    Elambda=Vlambda.*(1./sigma1(p1,p1)).*sum((Y(:,p1)'-(lambdad(p1,:)*etad)).*eta(m,:));
                    lambda1(p1,m)=normrnd(Elambda,sqrt(Vlambda),1,1);
                end
            end
        end
    end

    lambda1_vec=reshape(lambda1,[1 P1*M]);
    if (indsave>0)
    lambda1_save(indsave,:)=lambda1_vec;
    end
    
    % SAMPLE LAMBDA2
    for p2=1:P2
        for m=1:M
            if p2<m lambda2(p2,m)=0;
            
            elseif p2==m
                if M==1
                    Vlambda=(sigma2(p2,p2).*sigmasq_lambda2)./((sigmasq_lambda2.*sum(eta_ws(m,:).^2))+sigma2(p2,p2));
                    Elambda=Vlambda.*(1./sigma2(p2,p2)).*sum(Z(:,p2)'.*eta_ws(m,:));
                    lambda2(p2,m)=Elambda+(sqrt(Vlambda)*trandn(0-Elambda/sqrt(Vlambda),inf));
                else
                    lambdad=lambda2;
                    lambdad(:,m)=[];
                    etad=eta_ws;
                    etad(m,:)=[];
                    Vlambda=(sigma2(p2,p2).*sigmasq_lambda2)./((sigmasq_lambda2.*sum(eta_ws(m,:).^2))+sigma2(p2,p2));
                    Elambda=Vlambda.*(1./sigma2(p2,p2)).*sum((Z(:,p2)'-(lambdad(p2,:)*etad)).*eta_ws(m,:));
                    lambda2(p2,m)=Elambda+(sqrt(Vlambda)*trandn(0-Elambda/sqrt(Vlambda),inf));
                end
            else
                if M==1
                    Vlambda=(sigma2(p2,p2).*sigmasq_lambda2)./((sigmasq_lambda2.*sum(eta_ws(m,:).^2))+sigma2(p2,p2));
                    Elambda=Vlambda.*(1./sigma2(p2,p2)).*sum(Z(:,p2)'.*eta_ws(m,:));
                    lambda2(p2,m)=normrnd(Elambda,sqrt(Vlambda),1,1);
                else
                    lambdad=lambda2;
                    lambdad(:,m)=[];
                    etad=eta_ws;
                    etad(m,:)=[];
                    Vlambda=(sigma2(p2,p2).*sigmasq_lambda2)./((sigmasq_lambda2.*sum(eta_ws(m,:).^2))+sigma2(p2,p2));
                    Elambda=Vlambda.*(1./sigma2(p2,p2)).*sum((Z(:,p2)'-(lambdad(p2,:)*etad)).*eta_ws(m,:));
                    lambda2(p2,m)=normrnd(Elambda,sqrt(Vlambda),1,1);
                end
            end
        end
    end
    
    lambda2_vec=reshape(lambda2,[1 P2*M]);
    if (indsave>0)
    lambda2_save(indsave,:)=lambda2_vec;
    end
    
    runtimelambdas=toc
    
    tic;
    % SAMPLE SIGMA1
    for p1=1:P1
        betap1=beta1+(sum((Y(:,p1)'-(lambda1(p1,:)*eta)).^2)./2);
        sigma1(p1,p1)=1/gamrnd(alpha1+(NT/2),1/betap1);
    end
    
    if (indsave>0)
    sigma1_save(indsave,:)=diag(sigma1)';
    end
    
    % SAMPLE SIGMA2
    for p2=1:P2
        betap2=beta2+(sum((Z(:,p2)'-(lambda2(p2,:)*eta_ws)).^2)./2);
        sigma2(p2,p2)=1/gamrnd(alpha2+(N/2),1/betap2);
    end
    
    if (indsave>0)
    sigma2_save(indsave,:)=diag(sigma2)';
    end
    
    runtimesigmas=toc
    
    tic;
    % SAMPLE ETA
    for i=1:N
        locs=find(fipssb(:,1)==fipslb(i,1))';
        if size(locs,2)>1
            sumRatlocs=sum(R(locs,:));
            sumRatlocs(:,locs)=zeros(1,size(locs,2));
            neighbors=find(sumRatlocs>0);
        else
            sumRatlocs=R(locs,:);
            neighbors=find(sumRatlocs>0);
        end
        Rstar=kron(eye(size(locs,2)+size(neighbors,2))-(phi*R([locs neighbors],[locs neighbors])),eye(M));
        sigma11=Rstar(1:(size(locs,2)*M),1:(size(locs,2)*M));
        sigma12=Rstar(1:(size(locs,2)*M),(size(locs,2)*M+1):size(Rstar,2));
        sigma21=Rstar((size(locs,2)*M+1):size(Rstar,1),1:(size(locs,2)*M));
        sigma22=Rstar((size(locs,2)*M+1):size(Rstar,1),(size(locs,2)*M+1):size(Rstar,2));
        etasistar=reshape(eta(:,neighbors),[size(neighbors,2)*M 1]);
        
        mu_etasi=sigma12*inv(sigma22)*etasistar;
        sigma_inv_etasi=inv(sigma11-sigma12*inv(sigma22)*sigma21);
        
        lambda_tilde_si=[kron(eye(size(locs,2)),lambda1); kron(weights(locs,:)',lambda2)];
        sigma_tilde_inv_si=[kron(eye(size(locs,2)),diag(1./diag(sigma1))) zeros(P1*size(locs,2),P2); zeros(P2,P1*size(locs,2)) diag(1./diag(sigma2))];
        lxs=lambda_tilde_si'*sigma_tilde_inv_si;
        lxsxl=lxs*lambda_tilde_si;
        xsi=[reshape(Y(locs,:)',[size(locs,2)*P1 1]);reshape(Z(i,:),[P2 1])];
        
        Veta=inv(lxsxl+sigma_inv_etasi);
        Veta=(Veta+Veta.')/2;
        Eeta=Veta*(lxs*xsi+sigma_inv_etasi*mu_etasi);
        eta(:,locs)=reshape(mvnrnd(Eeta,Veta),[M size(locs,2)]);
    end
    
    %center and scale the factors (rows of eta)
    eta=(eta-repmat(mean(eta')',1,NT))./repmat(std(eta')',1,NT);
    eta_tilde=reshape(eta,[1 M*NT]);
    
    eta_weight=bsxfun(@times,eta,weights');
    eta_ws=[];
    for i=1:N
        locs=find(fipssb(:,1)==fipslb(i,1));
        eta_ws=[eta_ws sum(eta_weight(:,locs),2)];
    end
    
    if (indsave>0)
      if (nkeepeta==NT*M)
         eta_save(indsave,:)=eta_tilde;
      else
         eta_save(indsave,:)=eta_tilde(:,keepeta);
      end
    end
    
    runtimeeta=toc
    
    tic;
    % SAMPLE PHI (METROPOLIS STEP)
    phi_trans=log((phi-lb)/(ub-phi));
    phi_trans_prop=normrnd(phi_trans,sqrt(sigma_m_phi),1,1);
    phi_prop=(ub*exp(phi_trans_prop)+lb)/(1+exp(phi_trans_prop));
    [logh_prop_rvalue]=maup_log_h_speed(phi_trans_prop,phi_prop,NT,M,eta_tilde,R);
    [logh_old_rvalue]=maup_log_h_speed(phi_trans,phi,NT,M,eta_tilde,R);
    r=exp(logh_prop_rvalue-logh_old_rvalue);
    if (r>=1)
        phi=phi_prop;
        accept_phi=accept_phi+1;
    elseif (r>=rand(1))
        phi=phi_prop;
        accept_phi=accept_phi+1;
    end
    
    if (indsave>0)
        phi_save(indsave,:)=phi;
        accept_phi_save(indsave,:)=accept_phi/g;
    end
    
    runtimephi=toc
    
    % save output after collection of nsampfile samples
    if indsave==nsampfile
        nsaves=nsaves+1;
        fid = fopen(strcat('lambda1_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,P1*M),'\r\n'),lambda1_save');
        fclose(fid);
        fid = fopen(strcat('lambda2_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,P2*M),'\r\n'),lambda2_save');
        fclose(fid);
        fid = fopen(strcat('sigma1_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,P1),'\r\n'),sigma1_save');
        fclose(fid);
        fid = fopen(strcat('sigma2_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,P2),'\r\n'),sigma2_save');
        fclose(fid);
        fid = fopen(strcat('eta_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,NT*M),'\r\n'),eta_save');
        fclose(fid);
        fid = fopen(strcat('phi_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,'%.4f\r\n',phi_save);
        fclose(fid);
        fid = fopen(strcat('accept_phi_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,'%.4f\r\n',accept_phi_save);
        fclose(fid);
        
        lambda1_save=zeros(nsampfile,P1*M);
        lambda2_save=zeros(nsampfile,P2*M);
        sigma1_save=zeros(nsampfile,P1);
        sigma2_save=zeros(nsampfile,P2);
        eta_save=zeros(nsampfile,nkeepeta);
        phi_save=zeros(nsampfile,1);
        accept_phi_save=zeros(nsampfile,1);
        
    elseif g==burn_in+reps
        nsaves=nsaves+1;
        fid = fopen(strcat('lambda1_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,P1*M),'\r\n'),lambda1_save(1:indsave,:)');
        fclose(fid);
        fid = fopen(strcat('lambda2_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,P2*M),'\r\n'),lambda2_save(1:indsave,:)');
        fclose(fid);
        fid = fopen(strcat('sigma1_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,P1),'\r\n'),sigma1_save(1:indsave,:)');
        fclose(fid);
        fid = fopen(strcat('sigma2_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,P2),'\r\n'),sigma2_save(1:indsave,:)');
        fclose(fid);
        fid = fopen(strcat('eta_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,strcat(repmat('%.4f ',1,NT*M),'\r\n'),eta_save(1:indsave,:)');
        fclose(fid);
        fid = fopen(strcat('phi_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,'%.4f\r\n',phi_save(1:indsave,:));
        fclose(fid);
        fid = fopen(strcat('accept_phi_',int2str(nsaves),'.txt'),'w');
        fprintf(fid,'%.4f\r\n',accept_phi_save(1:indsave,:));
        fclose(fid);        
        
    end
    
    %end timer
    %run_times=[run_times toc];
end

%avg_run_time=mean(run_times);

end

