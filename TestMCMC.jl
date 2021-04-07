using PyPlot
include("mcmc.jl")



#prop(mu,x) = pdf(MvNormal(mu, I(2)), x)
prop(mu) = MvNormal(mu,[3 0 0; 0 3 0; 0 0 3])

function target(x)
    #return pdf(MvNormal([-5, -5], I(2)), x) + pdf(MvNormal([5, 5], I(2)), x) 
    return log(pdf(MvNormal([-2, 2, 0], [1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1]), x))
end


samps, acc = MHsample(target, prop, [-2,2, 0], 2000, 200, islogged = true)

plt.scatter(samps[:, 1],samps[:,2])

f,ax = plt.subplots(3,3)

Ndims = size(samps,2)

for i =1:Ndims
    for j = 1:Ndims
    ax[i,j].scatter(samps[:, i],samps[:,j])
    end
end
