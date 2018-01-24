import numpy as np
import json
import os
import tqdm


class Averager:
    sum = 0
    number = 0

    def add(self, value):
        self.sum += value
        self.number += 1

    def average(self):
        if self.number > 0:
            return self.sum / self.number
        return self.sum


class LennardJonesSimulation:
    """ simulation parameters """
    # number of particles
    N = 100
    # particle size
    sigma = 1
    # density
    rho = 0.8
    # temperature
    T = 1.0
    # cut-off radius
    rc = 5.0
    # dispersity
    sigma_deviation = 0.1

    """ time parameters """
    # time step
    dt = 0.01
    # time steps in equilibrium phase
    nequil = 2000
    # time steps in main phase
    nproduct = 6000

    """ rdf parameters """
    # rdf width of bins
    binwidth = 0.010
    # Maximum radius for sampling g(r)
    grmax = 5
    # Sample every nsamp timestaps
    nsamp = 10

    trajectory_export = 0

    def __init__(self, folder=None, seed=1234, **kwargs):
        # set arguments obtained from kwargs
        for name in kwargs:
            # try to get the attribute (raises an Exception if the attribute does not exist)
            getattr(self, name)
            # print
            print(name, getattr(self, name), kwargs[name])
            # set the attribute
            setattr(self, name, kwargs[name])

        # round N to the next quadratic number
        self.N = int(np.sqrt(self.N)) ** 2

        # calculate rc**2
        self.rc2 = self.rc * self.rc

        # total timesteps
        self.nt = self.nequil + self.nproduct

        # times for the two phases
        self.tequil = self.nequil * self.dt
        self.tproduct = self.nproduct + self.dt

        # maximum time
        self.tmax = self.nt * self.dt

        # the simulation box size
        l = np.sqrt(self.N / self.rho)
        self.box_size = [l, l]

        # the cut-off has to be smaller than the box size
        self.rc = min(self.box_size[0], self.rc)

        # the seed
        self.seed = seed

        parameters = json.dumps(
            {a: getattr(self, a) for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))},
            indent=4)

        # initialize the random generator
        self.random = np.random.RandomState(self.seed)

        # generate a random distribution of radii
        self.radius = self.random.normal(self.sigma, self.sigma_deviation, self.N) * 0.5

        # store the output folder
        self.folder = folder

        # create an array for the mutual forces of each particle on all other particles
        self.forces_square = np.zeros((self.N, self.N, 2))
        # create list of all possible pairs
        self.pair_i, self.pair_j = np.array([[i, j] for i in range(self.N) for j in range(i + 1, self.N)]).T

        # calculate the repulsion radii for each pair
        self.pair_radii = self.radius[self.pair_i] + self.radius[self.pair_j]

        # if an export folder is given
        if self.folder:
            # create the output folder
            if not os.path.exists(folder):
                os.mkdir(folder)

            # store the attributes
            with open(os.path.join(folder, "parameters.txt"), "w") as fp:
                fp.write(parameters)

            # open the file for the averages and the trajecotires
            self.outAverages = open(os.path.join(folder, "outAverages.txt"), "w")
            if self.trajectory_export:
                self.outTrajectories = open(os.path.join(folder, "outTrajectories.txt"), "w")

        # initialize the particles
        self.initialize()

    def initialize(self):
        # Initialize arrays for particle positions & velocities
        self.r = np.zeros((self.N, 2))
        self.r_next = np.zeros((self.N, 2))
        self.v = self.random.rand(self.N, 2) * 2 - 1
        self.v_next = np.zeros((self.N, 2))
        self.F = np.zeros((self.N, 2))

        # place particles in a regular grid
        nrows = np.sqrt(self.N)
        dlx = self.box_size[0] / nrows
        dly = self.box_size[1] / nrows
        self.r[:, 0] = np.arange(self.N) % nrows * dlx + 0.5
        self.r[:, 1] = np.floor(np.arange(self.N) / nrows) * dly + 0.5

        # set the mean velocity to zero
        self.v -= np.mean(self.v, axis=0)
        # scale the velocities to the desired temperature
        self.scale_velocities()

        self.print_coords("outCoords_start.npz")

        # define the offsets for the potentials
        self.offset_potential = self.potential(self.rc2, self.pair_radii)
        self.offset_gradient = self.potential_gradient(1, self.rc2, self.pair_radii)[:, 0]

        # initialize the averagers
        self.avg_temp = Averager()
        self.avg_epot = Averager()
        self.avg_ekin = Averager()
        self.avg_etot = Averager()

        # Histogram for pair correlation function
        self.nbins = int(np.ceil(
            self.grmax / self.binwidth))  # Maximum correlation length to be measured should be L/2 due to periodic BC
        self.bincount = 0  # Number of counts done on the histogram
        self.hist = np.zeros(int(self.nbins))

    def run(self):
        if self.folder:
            if self.trajectory_export:
                self.outTrajectories.write("#t\tn\tr_x\t\tr_y\t\tv_x\t\tv_y\n")
            self.outAverages.write("#t\tT(t)\t\t<T(t)>\t\tE_tot(T)\t\t<E_tot(T)>\n")
        for n in tqdm.tqdm(range(self.nt + 1)):
            # Current time
            t = self.dt * n
            # Calculate all forces
            self.forces()

            # perform leap-frog-integration
            self.v_next[:] = self.v + self.F * self.dt
            self.r_next[:] = self.r + self.v_next * self.dt

            # Write trajectories to a file
            if self.trajectory_export:
                if n % self.trajectory_export == 0:
                    for i in range(self.N):
                        self.outTrajectories.write("%6.3f\t%6d\t%e\t%e\t%e\t%e\n" % (
                        t, i, self.r[i, 0], self.r[i, 1], self.v[i, 0], self.v[i, 1]))

            # update particle coordinates
            self.v[:] = self.v_next
            self.r[:] = self.r_next

            # Equilibration phase, scale velocities to keep temperature
            if n < self.nequil:
                # Rescale velocities every 10 timesteps
                if n % 10 == 0:
                    self.scale_velocities()
            elif n % self.nsamp == 0:
                # Sum velocities
                self.vsum = np.sum(self.v_next)
                self.vsum2 = np.sum(np.power((self.v_next + self.v) / 2, 2))  # (v averaged between t and t+1)

                # calculate temperature
                Tt = self.vsum2 / (2.0 * self.N)

                # check if simulation is not running into numerical errors
                if Tt > self.T * 10:
                    raise ValueError("Numerical precession error, increase dt")
                self.avg_temp.add(Tt)

                self.avg_epot.add(self.epot)

                ekin = 0.5 * self.vsum2
                self.avg_ekin.add(ekin)

                etot = (self.epot + ekin)
                self.avg_etot.add(etot)

                self.update_histogram()

                if self.folder:
                    self.outAverages.write(
                        "%6.3f\t%e\t%e\t%e\t%e\n" % (t, Tt, self.avg_temp.average(), etot, self.avg_etot.average()))

        # finally evaluate and export the results
        self.save_results()

    def save_results(self):
        self.print_coords("outCoords_end.npz")

        R_list = []
        gr_list = []
        for i in range(1, self.nbins):
            R = i * self.binwidth
            area = 2.0 * np.pi * R * self.binwidth
            # Multiply g(r) by two, since in the histogram we only counted each pair once, but each pair
            # gives two contributions to g(r)
            gr = 2.0 * self.hist[i] / (self.rho * area * self.bincount * self.N)
            R_list.append(R)
            gr_list.append(gr)

        self.gr = np.array(gr_list)
        self.R = np.array(R_list)
        if self.folder:
            np.savez(os.path.join(self.folder, "outGr.npz"), R=self.R, gr=self.gr)

        if self.folder:
            self.outAverages.close()
            if self.trajectory_export:
                self.outTrajectories.close()

    def potential(self, r2, sigma=1, offset=0, cut_off=1000):
        # Calculate potential energy between two LJ-particles
        r2i = sigma / (r2 + 0.001)  # 1 / r ^ 2
        r6i = r2i * r2i * r2i  # 1 / r ^ 6
        return (r2 < cut_off) * (4.0 * r6i * (r6i - 1.0) - offset)

    def potential_gradient(self, dr, r2, sigma=1, offset=0, cut_off=1000):
        # Calculate the gradient of the LJ-potential
        r2i = sigma / (r2 + 0.001)  # 1 / r ^ 2
        r6i = r2i * r2i * r2i  # 1 / r ^ 6
        return ((r2 < cut_off) * (-48.0 * r2i * r6i * (r6i - 0.5) - offset))[:, None] * dr

    def distance(self, p1, p2):
        # the difference between the points
        d = p1 - p2
        # periodic boundary conditions
        d = d - self.box_size * np.round(d / self.box_size)

        # calculate the quadratic distance r^2
        r2 = np.sum(np.power(d, 2), axis=1)
        return d, r2

    def forces(self):
        # get the distances between all particles
        d_vec, r2 = self.distance(self.r[self.pair_i], self.r[self.pair_j])
        # the forces are the negative gradients
        flj = -1.0 * self.potential_gradient(d_vec, r2, self.pair_radii, self.offset_gradient, self.rc2)

        # sum over the forces of particle i
        self.forces_square[self.pair_i, self.pair_j, :] = flj
        self.F = np.sum(self.forces_square, axis=1) - np.sum(self.forces_square, axis=0)

        # sum the total potential energy
        self.epot = np.sum(self.potential(r2, self.pair_radii, self.offset_potential, self.rc2))

    def scale_velocities(self):
        # sum of squared velocities
        v2 = np.sum(np.power(self.v, 2))
        # scaling factor
        fs = np.sqrt(2.0 * self.N * self.T / v2)
        # scale velocities
        self.v *= fs

    def update_histogram(self):
        _, r2 = self.distance(self.r[self.pair_i], self.r[self.pair_j])
        d = np.sqrt(r2)
        # Find correct bin of histogram
        bin = np.floor(d / self.binwidth).astype("int")
        bin = bin[bin < self.nbins]
        bin = bin[bin >= 0]
        # add one to each of the bins
        np.add.at(self.hist, bin, 1)
        # Number of times averaging is done
        self.bincount += 1

    def print_coords(self, filename):
        if self.folder:
            np.savez(os.path.join(self.folder, filename), r=self.r, v=self.v, radius=self.radius)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import helpers
    sim = LennardJonesSimulation()
    sim.run()
    plt.plot(sim.R, sim.gr)
    plt.show()
