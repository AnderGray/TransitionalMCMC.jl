using PyPlot

# 1000 samples with ND gaussians
Ndims = [1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30]'
ts = [0.279603	0.637024	1.028485	1.003426	1.250441	1.374436	1.414937	1.588211	1.967722	2.024116	2.55868	2.589394	2.618068	2.919893	2.95502	3.258155	3.547371	3.728454	3.939705	4.212554	3.871222	4.072321	4.340777	4.284822	4.755864	4.718708	5.135071	5.260214	6.393843	6.390321]'
js = [2	3	5	5	6	7	7	8	9	9	10	11	11	12	12	14	14	14	14	15	14	15	13	14	15	15	15	15	17	17]'

slowdown = ts ./ts[1]


fig = figure(figsize= [10,10])
plt.plot(Ndims, js, linewidth = 2, color = "blue")

title("1000 samples of N-D gaussians", fontsize = 22)
xlabel("Num of dimension", fontsize = 22)
ylabel("Number of iterations", fontsize = 22)

fig = figure(figsize= [10,10])
plt.plot(Ndims, slowdown, linewidth = 2, color = "blue")

title("1000 samples of N-D gaussians", fontsize = 22)
xlabel("Num of dimension", fontsize = 22)
ylabel("slowdown", fontsize = 22)
