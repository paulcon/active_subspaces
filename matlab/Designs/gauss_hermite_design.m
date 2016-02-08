function design = gauss_hermite_design(N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Tensor product Gauss-Hermite quadrature points
%
%   Inputs:
%          N: 1-by-m array containing the number of points in the design
%             along each dimension
%
%  Outputs:
%          design: N-by-m array containing the design points
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


J = diag(sqrt(1:N(1)-1), 1) + diag(sqrt(1:N(1)-1), -1);
design = sort(eig(J));
design(abs(design) < 1e-12) = 0;

for i = 2:length(N)
    J = diag(sqrt(1:N(i)-1), 1) + diag(sqrt(1:N(i)-1), -1);
    e = sort(eig(J));
    e(abs(e) < 1e-12) = 0;
    
    design = [kron(design, ones(size(e, 1), 1)),...
              kron(ones(size(design, 1), 1), e)];
end

end