import numpy as np
import matplotlib.pyplot as plt
import os
import json

def getPressure(foldername):
    parameters = getParameters(foldername)

    data = np.load(os.path.join(foldername, "outGr.npz"))
    R_list = data["R"]
    gr_list = data["gr"]

    p = 0
    for R, gr in zip(R_list, gr_list):
        # Calculate other quantities from g(r)
        if 0 < R < parameters["rc"]:
            r6i = np.power(1.0 / R, 6)
            p += gr * 2 * np.pi * parameters["rho"] * parameters["rho"] * R * R * 48 * (r6i * r6i - 0.5 * r6i) * parameters["binwidth"] / 2

    data = np.loadtxt(os.path.join(foldername, "outAverages.txt"))

    T_mean = data[-1, 2]
    p = p + parameters["rho"] * T_mean
    return p

def getParameters(foldername):
    with open(os.path.join(foldername, "parameters.txt"), "r") as fp:
        return json.loads(fp.read())

def plotPositions(foldername, filename="outCoords_end.npz"):
    def add_circle(x, y, radius, **kwargs):
        from matplotlib.patches import Ellipse
        try:
            rx, ry = radius
        except TypeError:
            rx = radius
            ry = radius
        ellipse = Ellipse(xy=(x, y), width=rx*2, height=ry*2, **kwargs)
        plt.gca().add_patch(ellipse)
        #plt.plot([x], [y])
    def plotData(x, y, r, **kwargs):
        for i in range(len(x)):
            add_circle(x[i], y[i], r[i], **kwargs)
    print(os.path.join(foldername, filename))
    data = np.load(os.path.join(foldername, filename))
    radii = data["radius"]*(2**(1./6.))
    data = data["r"]
    #radii = np.loadtxt(os.path.join(foldername, "radii.txt"))*(2**(1./6.))
    parameters = getParameters(foldername)
    lx = parameters["box_size"][0]
    ly = parameters["box_size"][0]
    print(lx, ly)
    #for i in len(data):
    plotData(data[:, 0] % lx, data[:, 1] % ly, radii, color="b", alpha=0.5)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            plotData(data[:, 0] % lx + lx*i, data[:, 1] % ly + ly*j, radii, color="b", alpha=0.25)
            #plt.plot(data[:, 1] % lx + lx*i, data[:, 2] % ly + ly*j, 'bo', alpha=0.5)
    plt.axis("equal")
    plt.xlim(-lx*0.1, lx*1.1)
    plt.ylim(-ly*0.1, ly*1.1)

def loadData():
    folder = r"D:\FAUbox_NewStuff\aerial\clean\plots/"
    image_data = {"name": "Baie_du_Marin_2010_grouped"}
    output_density = "tmp/rdf_" + image_data["name"] + "_density.txt"
    output_x = "tmp/rdf_" + image_data["name"] + "_x.txt"
    output_y = "tmp/rdf_" + image_data["name"] + "_y.txt"

    y0 = np.loadtxt(folder + output_x)
    x0 = np.loadtxt(folder + output_y)
    density = np.loadtxt(folder + output_density)
    return x0, y0, density

def loadTrajectories(filename, filename_source=""):
    if os.path.exists(filename):
        return np.load(filename)["data"]
    xs = []
    ys = []
    N = 0
    with open(filename_source, "r") as fp:
        for lno, line in enumerate(fp):
            if lno == 0:
                continue
            variables = line.split()
            t, i, x, y, vx, vy = variables
            N = max(N, int(i)+1)
            xs.append(float(x))
            ys.append(float(y))
    data = np.vstack((xs, ys))
    data = data.reshape(2, int(data.shape[1]/N), N)
    data = data.transpose(2, 0, 1)
    #np.savez("outTrajectories3", data=data)
    np.savez_compressed(filename, data=data)
    return data

def getMSD(trajectories, parameters, magnitudes=6, skip=1):
    x = trajectories[:, 0, :]
    y = trajectories[:, 0, :]
    dt = parameters["dt"]*skip
    N = trajectories.shape[0]

    def displacement(x, y, t):
        return np.mean((x[0:-t] - x[t:]) ** 2 + (y[0:-t] - y[t:]) ** 2)

    def MSD(x, y, times):
        ds = np.zeros(len(times))
        for i, t in enumerate(times):
            ds[i] = displacement(x, y, t)
        return ds

    times = [1, 3]
    for i in range(magnitudes):
        times.extend([1 * int(10 ** i), 3 * int(10 ** i)])
    times = np.array(times)
    print("times", len(x[0, :]), np.log10(len(x[0, :])))
    print("times", len(x[0, :])*dt, np.log10(len(x[0, :])*dt))

    ds = None
    for i in range(N):
        msd = MSD(x[i, :], y[i, :], times)
        if ds is None:
            ds = msd
        else:
            ds += msd

    return times * dt, ds / N