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
    cf.stimuli.modulate_stimuli(apply_fn_1=True, apply_fn_2=False)
    
    for t in range(cf.n_steps):
        print(f"step {t}/{cf.n_steps}")

        if cf.swap_stimuli is not None and t is cf.swap_stimuli:
            cf.stimuli.modulate_stimuli(apply_fn_1=False, apply_fn_2=True)

        sim.step(t)

        if cf.record_gif:
            img = sim.get_img(t, save=(t % cf.save_every == 0))
            imgs.append(img)

        if cf.record_stim_field_every is not None and  t % cf.record_stim_field_every == 0:
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
    cf.logdir = "log_bar_trans"
    cf.gif_path = "gifs/bars_transient.gif"

    cf.n_steps = 400
    cf.save_every = 10
    cf.init_particles = False
    cf.save_particles = False
    cf.apply_stim = True
    cf.record_gif = True
    cf.record_evidence = True
    cf.record_stim_field_every = None

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

    # cf.kappa = 4.0
    cf.kappa = 3.6
    cf.eps = 10
    # cf.dt = 0.05
    cf.dt = 0.01
    cf.phi = 1e-7

    cf.stim_weight = 1.0
    cf.stim_factor = 10.0
    cf.swap_stimuli = 100
    cf.apply_stim_to_blanket = True
    cf.stim_type = Stimuli.BARS
    cf.apply_type = Stimuli.DUAL
    cf.stim_weight = 1.0
    cf.stimuli = Stimuli(cf)

    main(cf)
