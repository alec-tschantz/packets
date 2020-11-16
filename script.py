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

        if cf.record_stim_field_every is not None and t % cf.record_stim_field_every == 0:
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
    cf.save_every = 5
    cf.init_particles = False
    cf.save_particles = False
    cf.apply_stim = True
    cf.record_gif = True
    cf.record_evidence = True
    cf.record_stim_field_every = 20

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
    # TODO
    cf.vec_prec = [4.0, 0.5, 0.5, 4.0]
    # cf.vec_prec = [4.0, 2.0, 2.0, 4.0]
    cf.vec_omega_std = 10 ** -2
    cf.y_omega_std = 10 ** -3

    # cf.kappa = 4.0
    cf.kappa = 3.6
    cf.eps = 10
    # cf.dt = 0.05
    cf.dt = 0.01
    cf.phi = 1e-7

    cf.stim_weight = 1.0
    cf.blanket_weight = 1.0

    cf.n_steps = 300

    """
    CIRCLE 
    """
    cf.logdir = "log_circle_test"
    cf.gif_path = "gifs/circle_test.gif"

    cf.apply_stim_to_blanket = True
    cf.stim_type = Stimuli.CIRCLE
    cf.apply_type = Stimuli.THRESHOLD
    cf.stimuli = Stimuli(cf)
    main(cf)

    """
    CROSS DUAL
    cf.logdir = "log_cross_dual"
    cf.gif_path = "gifs/cross_dual.gif"

    cf.apply_stim_to_blanket = False
    cf.stim_type = Stimuli.CROSS
    cf.apply_type = Stimuli.DUAL
    cf.stimuli = Stimuli(cf)
    main(cf)
    """

    """
    CROSS SINGLE
    cf.logdir = "log_cross_single"
    cf.gif_path = "gifs/cross_single.gif"

    cf.apply_stim_to_blanket = False
    cf.stim_type = Stimuli.CROSS
    cf.apply_type = Stimuli.SINGLE
    cf.stimuli = Stimuli(cf)
    main(cf)
    """

    """
    CIRCLE SINGLE
    cf.logdir = "log_circle_single"
    cf.gif_path = "gifs/circle_single.gif"
    cf.apply_stim_to_blanket = False
    cf.stim_type = Stimuli.CIRCLE
    cf.apply_type = Stimuli.SINGLE
    cf.stimuli = Stimuli(cf)
    main(cf)
    """

    """
    CIRCLE THRESHOLD
    cf.logdir = "log_circle_threshold"
    cf.gif_path = "gifs/circle_threshold.gif"

    cf.apply_stim_to_blanket = False
    cf.stim_type = Stimuli.CIRCLE
    cf.apply_type = Stimuli.THRESHOLD
    cf.stimuli = Stimuli(cf)
    main(cf)
    """

    """
    BAR DUAL BLANKET
    cf.logdir = "log_bar_dual_blank"
    cf.gif_path = "gifs/bar_dual_blank.gif"

    cf.apply_stim_to_blanket = True
    cf.stim_type = Stimuli.BARS
    cf.apply_type = Stimuli.DUAL
    cf.stimuli = Stimuli(cf)
    main(cf)
    """

    """
    CROSS DUAL BLANKET
    cf.logdir = "log_cross_dual_blank"
    cf.gif_path = "gifs/cross_dual_blank.gif"

    cf.apply_stim_to_blanket = True
    cf.stim_type = Stimuli.CROSS
    cf.apply_type = Stimuli.DUAL
    cf.stimuli = Stimuli(cf)
    main(cf)
    """

    """
    CROSS SINGLE BLANKET
    cf.logdir = "log_cross_single_blank"
    cf.gif_path = "gifs/cross_single_blank.gif"

    cf.apply_stim_to_blanket = True
    cf.stim_type = Stimuli.CROSS
    cf.apply_type = Stimuli.SINGLE
    cf.stimuli = Stimuli(cf)
    main(cf)
    """

    """
    CIRCLE SINGLE BLANKET
    cf.logdir = "log_circle_single_blank"
    cf.gif_path = "gifs/circle_single_blank.gif"
    cf.apply_stim_to_blanket = True
    cf.stim_type = Stimuli.CIRCLE
    cf.apply_type = Stimuli.SINGLE
    cf.stimuli = Stimuli(cf)
    main(cf)
    """

    """
    CIRCLE THRESHOLD BLANKET
    cf.logdir = "log_circle_threshold_blank"
    cf.gif_path = "gifs/circle_threshold_blank.gif"

    cf.apply_stim_to_blanket = True
    cf.stim_type = Stimuli.CIRCLE
    cf.apply_type = Stimuli.THRESHOLD
    cf.stimuli = Stimuli(cf)
    main(cf)
    """
