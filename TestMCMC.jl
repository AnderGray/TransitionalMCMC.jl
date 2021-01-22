include("mcmc.jl")



#prop(mu,x) = pdf(MvNormal(mu, I(2)), x)
proprnd(mu) = rand(MvNormal(mu,[2 0; 0 2]))

function target(x)
    #return pdf(MvNormal([-5, -5], I(2)), x) + pdf(MvNormal([5, 5], I(2)), x) 
    return pdf(MvNormal([-2, 2], [1 0.5; 0.5 1]), x)
end


samps, acc = MHsample(target, proprnd, [-2,2], 2000, 200)

plt.scatter(samps[:, 1],samps[:,2])