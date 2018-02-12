function [ r_val] = maup_log_h_speed(phitransnew,phinew,N,K,eta_vec,R)
%Computes the log likelihood for metropolis sampler in spatial model

psi_inv_new=inv(eye(N)-phinew.*R);
r_val=(-1/2)*(K*log(1/det(psi_inv_new)))-((1/2)*sum((eta_vec*kron(psi_inv_new,eye(K))).*eta_vec))+phitransnew-2*log(1+exp(phitransnew));

end

