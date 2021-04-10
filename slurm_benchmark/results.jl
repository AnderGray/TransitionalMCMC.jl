using PyPlot

# 2000 samples with 0.1s per model eval
Nprocs = [1, 5, 10, 20, 50, 80, 100, 120, 150, 180 ]
t = [13287.663	2697.591854	1375.82991	709.99628 313.435787	218.478622	181.348302	161.611588 141.589949 127.132717]'

# 2000 samples with 1s per model eval
Nprocs2 = [5, 10, 20, 50, 80, 100, 120, 150, 180 ]
t2 = [26119.82705	13085.11175	6567.842152	2655.510665	1679.504391	1353.719791	1157.073554	960.797982	830.116548]'



Rt = t[1] ./ t  * Nprocs[1]
t = t./60

Rt2 = t2[1] ./ t2 * Nprocs2[1]
t2 = t2./60

fig = figure(figsize= [10,10])
plt.plot(Nprocs, t, linewidth = 2)

title("2000 samples with 0.1s per model eval", fontsize = 22)
xlabel("Num cpus", fontsize = 22)
ylabel("minutes", fontsize = 22)

fig = figure(figsize= [10,10])
plt.plot(Nprocs, Rt, linewidth = 2, label = "TransitionalMCMC.jl", color = "blue")
plt.plot([0; Nprocs[end]], [0; Nprocs[end]], linewidth = 2, label = "linear", color = "red")

title("2000 samples with 0.1s per model eval", fontsize = 22)
xlabel("Num cpus", fontsize = 22)
ylabel("minutes", fontsize = 22)
legend(fontsize = 22)


fig = figure(figsize= [10,10])
plt.plot(Nprocs2, t2, linewidth = 2)

title("2000 samples with 1s per model eval", fontsize = 22)
xlabel("Num cpus", fontsize = 22)
ylabel("minutes", fontsize = 22)

fig = figure(figsize= [10,10])
plt.plot(Nprocs2, Rt2, linewidth = 2, label = "TransitionalMCMC.jl", color = "blue")
plt.plot([0; Nprocs[end]], [0; Nprocs[end]], linewidth = 2, label = "linear", color = "red")

title("2000 samples with 0.1s per model eval", fontsize = 22)
xlabel("Num cpus", fontsize = 22)
ylabel("minutes", fontsize = 22)
legend(fontsize = 22)