import numpy as np

from blankets.utils import get_img, get_signal_img


class BlanketSimulation(object):
    def __init__(self, cf):
        self.cf = cf
        self.init_particles = cf.init_particles
        self.n_dims = cf.n_dims
        self.n_signals = cf.n_signals
        self.n_parts = cf.n_parts
        self.x_size = cf.x_size
        self.y_size = cf.y_size

        self.apply_stim = cf.apply_stim
        self.stimuli = cf.stimuli

        self.loc_path = cf.loc_path
        self.vel_path = cf.vel_path
        self.acc_path = cf.acc_path

        self.kappa = cf.kappa
        self.dt = cf.dt
        self.vec_prec = cf.vec_prec
        self.vec_omega = np.random.normal(0, cf.vec_omega_std, self.n_signals)
        self.y_omega = np.random.normal(0, cf.y_omega_std, self.n_signals)
        self.mu_prec_scale = cf.mu_prec_scale
        self.eps = cf.eps
        self.phi = cf.phi

        self._init_vars()

    def step(self, t):
        for p in range(self.n_parts):
            self.vel[p, :] = 0
            self.acc[p, :] = 0
            self.mu_dot[p, :] = 0

            self._update_expectations(p)
            signal, signal_dot = self._get_signal(p)
            for s in range(self.n_signals):
                self.vec_field[p, s, :] = signal_dot[s, :] * signal[s] ** -1
                self.vec_mag[p, s] = np.abs(
                    np.sqrt(np.sum(self.vec_field[p, s, :] ** 2)) + self.vec_omega[s]
                )
                self.y[p, s] = np.abs(self.psi[p, s] + self.y_omega[s])

            self._update_mu_dot(p)
            for s in range(self.n_signals):
                self.vel[p, :] += self._get_vel(p, s)
                self.acc[p, :] += self._get_acc(p, s)

        self.loc = self.loc + self.dt * self.vel + 0.5 * self.dt ** 2 * self.acc
        self.mu = self.mu + self.dt * self.mu_dot
        if self.apply_stim:
            self.mu, self.psi = self.stimuli.apply(self.loc, self.mu)
        self.psi = np.transpose((np.transpose(np.exp(self.mu))) / np.sum(np.exp(self.mu), axis=1))

    def get_img(self, t, save=False):
        img = get_img(
            self.loc, self.psi, self.cf, vec_field=None, title=f"Step {t}", save=save, step=t
        )
        return img

    def get_stim_field(self, t):
        img = get_signal_img(self.loc, self.psi, self.cf, title=f"Step {t}", save=True, step=t)
        return img

    def get_evidence(self):
        evidence = np.zeros(self.n_signals)
        for p in range(self.n_parts):
            evidence += self.psi[p, :]
        return evidence / self.n_parts

    def save_particles(self):
        np.save(self.loc_path, self.loc)
        np.save(self.vel_path, self.vel)
        np.save(self.acc_path, self.acc)

    def _update_expectations(self, idx):
        self.vec_exp[idx, :] = 0
        for s in range(self.n_signals):
            self.vec_exp[idx, :] += self.psi[idx, s] * self.vec_prior[s, :]

    def _get_signal(self, idx):
        signal = np.zeros(self.n_signals)
        signal_dot = np.zeros((self.n_signals, self.n_dims))
        for p in range(self.n_parts):
            if p != idx:
                dx = self.loc[p, :] - self.loc[idx, :]
                dxq = np.sum(dx ** 2)
                for s in range(self.n_signals):
                    signal[s] += self.psi[p, s] * np.exp(-0.5 * dxq)
                    for d in range(self.n_dims):
                        signal_dot[s, d] += self.psi[p, s] * dx[d] * np.exp(-0.5 * dxq)
        return signal, signal_dot

    def _update_mu_dot(self, idx):
        for i in range(self.n_signals):
            for j in range(self.n_signals):
                self.mu_dot[idx, j] += (
                    -1
                    * self.vec_prec[i]
                    * (self.vec_prior[j, i] - self.vec_mag[idx, i])
                    * self.psi[idx, j]
                )
                self.mu_dot[idx, j] += (
                    -1 * self.y_prec[i] * (self.y_prior[j, i] - self.y[idx, i]) * self.psi[idx, j]
                )

                for s in range(self.n_signals):
                    self.mu_dot[idx, i] += (
                        self.vec_prec[i]
                        * (self.vec_prior[s, i] - self.vec_mag[idx, i])
                        * self.psi[idx, j]
                        * self.psi[idx, s]
                    )
                    self.mu_dot[idx, i] += (
                        self.y_prec[i]
                        * (self.y_prior[s, i] - self.y[idx, i])
                        * self.psi[idx, j]
                        * self.psi[idx, s]
                    )

        for s in range(self.n_signals):
            self.mu_dot[idx, s] += -1 * self.mu_prec[s] * self.mu[idx, s]

    def _get_vel(self, idx, s):
        return (
            -1
            * self.psi[idx, s]
            * self.vec_field[idx, s, :]
            / np.linalg.norm(self.vec_field[idx, s, :])
            * self.vec_prec[s]
            * (self.vec_exp[idx, s] - self.vec_mag[idx, s])
        )

    def _get_acc(self, idx, s):
        return (
            self.vec_field[idx, s, :]
            / np.linalg.norm(self.vec_field[idx, s, :])
            * self.vec_prec[s]
            * (
                (self.vec_exp[idx, s] - self.vec_mag[idx, s])
                * self.psi[idx, s]
                * (1 - self.psi[idx, s])
                * self.mu_dot[idx, s]
                + self.vec_omega[s] * self.eps ** -1
            )
        )

    def _init_vars(self):
        if self.init_particles:
            self.loc = self._init_locations(self.x_size, self.y_size)
            self.vel = np.zeros((self.n_parts, self.n_dims))
            self.acc = np.zeros((self.n_parts, self.n_dims))
        else:
            self.loc = np.load(self.loc_path)
            self.vel = np.load(self.vel_path)
            self.acc = np.load(self.acc_path)

        self.psi = np.zeros((self.n_parts, self.n_signals)) + self.n_signals ** -1
        self.mu = np.log(self.psi)
        self.mu_dot = np.zeros((self.n_parts, self.n_signals))
        self.mu_prec = np.ones(self.n_signals) * self.mu_prec_scale

        self.vec_field = np.zeros((self.n_parts, self.n_signals, self.n_dims))
        self.vec_mag = np.zeros((self.n_parts, self.n_signals))
        self.vec_exp = np.zeros((self.n_parts, self.n_signals))
        self.vec_prec = np.array(self.vec_prec)
        self.vec_prior = np.zeros((self.n_signals, self.n_signals)) + self.kappa
        self.vec_prior[0, :] = [self.kappa, self.kappa, self.kappa * 2, self.kappa * 2]
        self.vec_prior[1, :] = [self.kappa, self.kappa, self.kappa, self.kappa * 2]
        self.vec_prior[2, :] = [self.kappa * 2, self.kappa, self.kappa, self.kappa]
        self.vec_prior[3, :] = [self.kappa * 2, self.kappa * 2, self.kappa, self.kappa]

        self.y = np.zeros((self.n_parts, self.n_signals))
        self.y_prior = np.eye(self.n_signals) + self.phi
        self.y_prec = np.ones(self.n_signals)

    @staticmethod
    def _init_locations(x_size, y_size):
        size = x_size * y_size
        cx = np.linspace(-(x_size - 1), (x_size - 1), x_size) * 2
        cy = np.linspace(-(y_size - 1), (y_size - 1), y_size) * 2
        loc = np.transpose(np.reshape(np.meshgrid(cx, cy), (2, size)))
        for i in range(x_size)[::2]:
            loc[(i * y_size) : (i * y_size + x_size), 0] += 2
        return loc

