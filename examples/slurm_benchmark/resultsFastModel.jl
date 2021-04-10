using PyPlot

# 2000 samples with 0.1s per model eval

Nprocs = [1, 5, 10, 20, 50, 80, 100, 120, 150, 180 ]
t = [13287.663	2697.591854	1375.82991	709.99628 313.435787	218.478622	181.348302	161.611588 141.589949 127.132717]'

t = t./60

Rt = t ./ t[1]

fig = figure(figsize= [10,10])
plt.plot(Nprocs, t, linewidth = 2)

title("2000 samples with 0.1s per model eval", fontsize = 22)
xlabel("Num cpus", fontsize = 22)
ylabel("minutes", fontsize = 22)