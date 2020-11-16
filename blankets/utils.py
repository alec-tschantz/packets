import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio

matplotlib.use("TkAgg")
matplotlib.rcParams["axes.linewidth"] = 1.6

__colors__ = ["red", "green", "yellow", "blue"]
__names__ = ["Internal (A)", "Blanket (A)", "Blanket (B)", "Internal (B)"]


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def save_gif(imgs, cf, fps=32):
    imageio.mimsave(cf.gif_path, imgs, fps=fps)


def save_evidence(evidence, cf):
    _, ax = plt.subplots()
    for s in range(cf.n_signals):
        ax.plot(evidence[:, s], color=__colors__[s], label=__names__[s])
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Evidence")
    plt.legend()
    plt.savefig(cf.logdir + "/evidence.png")
    np.save(cf.logdir + "/evidence", evidence)


def get_img(loc, psi, cf, plot_beliefs=True, vec_field=None, title=None, save=False, step=None):
    colors = _get_colors(psi, cf)
    edges = _get_edges(psi, cf)

    fig, ax = plt.subplots()
    ax.scatter(loc[:, 0], loc[:, 1], edgecolors=colors, linewidth=3, s=100, facecolor=edges)

    if vec_field is not None:
        _draw_vec_field(ax, loc, vec_field, cf)

    for p in range(cf.n_parts):
        belief = np.argmax(psi[p, :])
        ax.text(loc[p, 0] + 1.5, loc[p, 1], str(belief), fontsize=8)

    ax.set_xlim([-1 * cf.plot_size, cf.plot_size])
    ax.set_ylim([-1 * cf.plot_size, cf.plot_size])
    if title is not None:
        ax.set_title(title)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save:
        plt.savefig(cf.logdir + "/step_{}.png".format(step))
    plt.close()
    return img


def get_signal_img(loc, psi, cf, title=None, save=False, step=None):
    fig, ax = plt.subplots()
    imgs = np.zeros((cf.n_signals, cf.plot_size * 2, cf.plot_size * 2))
    for xi, x in enumerate(range(-cf.plot_size, cf.plot_size)):
        for yi, y in enumerate(range(-cf.plot_size, cf.plot_size)):
            grid_loc = np.array([x, y])
            signal = np.zeros(cf.n_signals)
            for p in range(cf.n_parts):
                dx = loc[p, :] - grid_loc
                dxq = np.sum(dx ** 2)
                for s in range(cf.n_signals):
                    signal[s] += psi[p, s] * np.exp(-0.5 * dxq)
            imgs[:, yi, xi] = signal

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(imgs[0, :, :])
    ax[0, 0].set_title(__names__[0])

    ax[0, 1].imshow(imgs[1, :, :])
    ax[0, 1].set_title(__names__[1])

    ax[1, 0].imshow(imgs[2, :, :])
    ax[1, 0].set_title(__names__[2])

    ax[1, 1].imshow(imgs[3, :, :])
    ax[1, 1].set_title(__names__[3])

    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save:
        plt.savefig(cf.logdir + "/signal_step_{}.png".format(step))
    plt.close()
    return img


def _get_colors(psi, cf):
    colors = np.zeros((cf.n_parts, 3))
    colors[:, 0] = psi[:, 0]
    colors[:, 1] = psi[:, 1] + psi[:, 2]
    colors[:, 2] = psi[:, 3]
    colors = np.clip(colors, 0.0001, 0.9999)
    colors[np.isnan(colors)] = 0.001
    return colors


def _get_edges(psi, cf):
    edges = np.zeros((cf.n_parts, 3))
    edges[:, :] += np.reshape(np.sum(psi[:, 2:4], axis=1) / (np.sum(psi[:, 0:4], axis=1)), (cf.n_parts, 1))
    edges = np.clip(edges, 0.0001, 0.9999)
    edges[np.isnan(edges)] = 0.001
    return edges


def _draw_vec_field(ax, loc, vec_field, cf):
    for s in range(cf.n_signals):
        ax.quiver(
            loc[:, 0],
            loc[:, 1],
            vec_field[:, s, 0],
            vec_field[:, s, 1],
            color=__colors__[s],
            alpha=0.4,
            headwidth=2,
            headlength=3,
        )
