import os
import pprint
import numpy as np

from blankets.utils import AttrDict, save_gif, save_evidence
from blankets.stimuli import Stimuli
from blankets.simulation import BlanketSimulation


def main(cf):
    os.makedirs(cf.logdir, exist_ok=True)
    pprint.pprint(f"config: {cf}")

    evidence = np.zeros((cf.n_steps, cf.n_signals))
    imgs = []

    sim = BlanketSimulation(cf)

    for t in range(cf.n_steps):
        print(f"step {t}/{cf.n_steps}")
        sim.step(t)

        if cf.record_gif:
            img = sim.get_img(t, save=(t % cf.save_every == 0))
            imgs.append(img)

        if t % cf.record_stim_field_every == 0:
            sim.get_stim_field(t)

        if cf.record_evidence:
            evidence[t, :] = sim.get_evidence()

    if cf.save_particles:
        sim.save_particles()

    if cf.record_gif:
        save_gif(imgs, cf)

    if cf.record_evidence:
        save_evidence(evidence, cf)


if __name__ == "__main__":

    cf = AttrDict()
    cf.n_steps = 200
    cf.save_every = 10
    cf.init_particles = True
    cf.save_particles = True
    cf.apply_stim = False
    cf.record_gif = True
    cf.record_evidence = True
    cf.record_stim_field_every = 10

    cf.logdir = "log_init"
    cf.gif_path = "gifs/init_particles.gif"
    cf.loc_path = "data/loc.npy"
    cf.vel_path = "data/vel.npy"
    cf.acc_path = "data/acc.npy"

    cf.x_size = 16
    cf.y_size = 16
    cf.n_parts = 256
    cf.plot_size = 60
    cf.n_dims = 2
    cf.n_signals = 4

    cf.mu_prec_scale = 0.4
    cf.vec_prec = [4.0, 0.5, 0.5, 4.0]
    cf.vec_omega_std = 10 ** -2
    cf.y_omega_std = 10 ** -3

    cf.kappa = 3.6
    cf.eps = 10
    cf.dt = 0.05
    cf.phi = 1e-7

    cf.stimuli = None

    main(cf)
