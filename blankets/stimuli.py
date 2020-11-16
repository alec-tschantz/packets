import numpy as np
import scipy.interpolate as interp
import scipy.ndimage as ndimage


class Stimuli(object):
    __possible_stimuli__ = ["CROSS", "CIRCLE", "BARS"]
    __possible_applications__ = ["SINGLE", "DUAL", "THRESHOLD"]

    CROSS = 0
    CIRCLE = 1
    BARS = 2

    SINGLE = 0
    DUAL = 1
    THRESHOLD = 2

    INTERNAL_A = 0
    BLANKET_A = 1
    BLANKET_B = 2
    INTERNAL_B = 3

    def __init__(self, cf, n_points=100):
        self.stim_type = cf.stim_type
        self.apply_type = cf.apply_type
        self.apply_stim_to_blanket = cf.apply_stim_to_blanket
        self.stim_weight = cf.stim_weight
        self.blanket_weight = cf.blanket_weight
        self.plot_size = cf.plot_size
        self.n_parts = cf.n_parts
        self.n_points = n_points

        self.apply_fn_1 = True
        self.apply_fn_2 = True

        self.fn_1 = None
        self.fn_1 = None
        self.raw_1 = None
        self.raw_2 = None

        if self.stim_type is self.CROSS:
            self.fn_1, self.fn_2, self.raw_1, self.raw_2 = self._get_cross_stim()
        elif self.stim_type is self.CIRCLE:
            self.fn_1, self.fn_2, self.raw_1, self.raw_2 = self._get_circle_stim()
        elif self.stim_type is self.BARS:
            self.fn_1, self.fn_2, self.raw_1, self.raw_2 = self._get_bars_stim()
        else:
            raise ValueError(f"{self.stim_type} is not a supported stimuli")

    def apply(self, loc, mu):
        if self.apply_type is self.SINGLE:
            return self._apply_single(loc, mu)
        elif self.apply_type is self.DUAL:
            return self._apply_dual(loc, mu)
        elif self.apply_type is self.THRESHOLD:
            return self._apply_threshold(loc, mu)
        else:
            raise ValueError(f"{self.apply_type} not a supported apply type")

    def modulate_stimuli(self, apply_fn_1, apply_fn_2):
        self.apply_fn_1 = apply_fn_1
        self.apply_fn_2 = apply_fn_2

    def get_img(self):
        img = np.zeros((self.plot_size * 2, self.plot_size * 2, 3))
        for xi, x in enumerate(range(-self.plot_size, self.plot_size)):
            for yi, y in enumerate(range(-self.plot_size, self.plot_size)):
                img[yi, xi, 0] += self.fn_1(x, y)
                if self.fn_2 is not None:
                    if self.apply_type is self.DUAL:
                        img[yi, xi, 2] += self.fn_2(x, y)
                    else:
                        img[yi, xi, 0] += self.fn_2(x, y)

                if self.apply_type is self.THRESHOLD:
                    if np.any(img[yi, xi, :] < 255.00):
                        img[yi, xi, 2] = 1.0
                    else:
                        img[yi, xi, 0] = 1.0
        return img

    def _apply_single(self, loc, mu):
        stim_1 = np.zeros(self.n_parts)
        stim_2 = np.zeros(self.n_parts)
        for p in range(self.n_parts):
            if self.fn_1 is not None and self.apply_fn_1:
                stim_1[p] = self.fn_1(loc[p, 0], loc[p, 1])
            if self.fn_2 is not None and self.apply_fn_2:
                stim_2[p] = self.fn_2(loc[p, 0], loc[p, 1])

        mu[:, self.INTERNAL_A] += self.stim_weight * stim_1
        mu[:, self.INTERNAL_A] += self.stim_weight * stim_2
        if self.apply_stim_to_blanket:
            mu[:, self.BLANKET_A] += self.blanket_weight * stim_1
            mu[:, self.BLANKET_A] += self.blanket_weight * stim_2
        psi = np.exp(mu) / np.sum(np.exp(mu))
        return mu, psi

    def _apply_dual(self, loc, mu):
        assert self.fn_2 is not None
        stim_1 = np.zeros(self.n_parts)
        stim_2 = np.zeros(self.n_parts)
        for p in range(self.n_parts):
            if self.fn_1 is not None and self.apply_fn_1:
                stim_1[p] = self.fn_1(loc[p, 0], loc[p, 1])
            if self.fn_2 is not None and self.apply_fn_2:
                stim_2[p] = self.fn_2(loc[p, 0], loc[p, 1])

        mu[:, self.INTERNAL_A] += self.stim_weight * stim_1
        mu[:, self.INTERNAL_B] += self.stim_weight * stim_2
        if self.apply_stim_to_blanket:
            mu[:, self.BLANKET_A] += self.blanket_weight * stim_1
            mu[:, self.BLANKET_B] += self.blanket_weight * stim_2
        psi = np.exp(mu) / np.sum(np.exp(mu))
        return mu, psi

    def _apply_threshold(self, loc, mu, threshold=0.001):
        stim_1 = np.zeros(self.n_parts)
        stim_2 = np.zeros(self.n_parts)
        for p in range(self.n_parts):
            if self.fn_1 is not None and self.apply_fn_1:
                stim_1[p] = self.fn_1(loc[p, 0], loc[p, 1])
            if self.fn_2 is not None and self.apply_fn_2:
                stim_2[p] = self.fn_2(loc[p, 0], loc[p, 1])

        stim = stim_1 + stim_2
        #  pos = np.where(self.stim_weight * stim >= threshold)
        #  neg = np.where(self.stim_weight * stim < threshold)

        #  treat positive and negative two stimuli,
        mu[:, self.INTERNAL_A] += self.stim_weight * stim
        mu[:, self.BLANKET_A] += self.blanket_weight * stim
        mu[:, self.INTERNAL_B] += self.stim_weight * (1 - stim)
        mu[:, self.BLANKET_B] += self.blanket_weight * (1 - stim)

        """
        if self.apply_stim_to_blanket:
            mu[pos, self.BLANKET_A] += (self.stim_weight / self.stim_factor) * stim[pos]
            mu[neg, self.BLANKET_B] += (self.stim_weight / self.stim_factor) * stim[neg]
        """
        psi = np.exp(mu) / np.sum(np.exp(mu))
        return mu, psi

    def _get_cross_stim(self):
        increment = (self.plot_size - (-self.plot_size)) / self.n_points
        grid = np.arange(-self.plot_size, self.plot_size, increment)
        center = self.n_points // 2
        x_dis = self.n_points // 15
        y_dis = (self.n_points // 3) + 10

        img = np.zeros((self.n_points, self.n_points)) + (10 ** -4)
        img[:, center - x_dis : center + x_dis] = 1
        img[0 : center - y_dis, :] = 0
        img[center + y_dis : self.n_points, :] = 0

        # TODO try to get signal gradient
        img = ndimage.gaussian_filter(img, 4)
        img_1 = ndimage.rotate(img, 45, mode="constant", reshape=False)
        img_2 = ndimage.rotate(img, 135, mode="constant", reshape=False)
        f_1 = interp.interp2d(grid, grid, img_1)
        f_2 = interp.interp2d(grid, grid, img_2)
        return f_1, f_2, img_1, img_2

    def _get_circle_stim(self):
        increment = (self.plot_size - (-self.plot_size)) / self.n_points
        grid = np.arange(-self.plot_size, self.plot_size, increment)
        img = np.zeros((self.n_points, self.n_points)) + (10 ** -4)
        x, y = self._get_points_in_circle(20)
        x += 30
        y += 30
        img[x, y] = 1
        # img = ndimage.gaussian_filter(img, 4)
        f_1 = interp.interp2d(grid, grid, img)
        return f_1, None, img, None

    def _get_bars_stim(self):
        center = self.n_points // 2
        left = self.n_points // 3
        right = self.n_points - (self.n_points // 3)
        x_dis = self.n_points // 10
        y_dis = self.n_points // 4

        increment = (self.plot_size - (-self.plot_size)) / self.n_points
        grid = np.arange(-self.plot_size, self.plot_size, increment)
        img_1 = np.zeros((self.n_points, self.n_points)) + (10 ** -4)
        img_2 = np.zeros((self.n_points, self.n_points)) + (10 ** -4)

        img_1[:, left - x_dis : left + x_dis] = 1
        img_1[0 : center - y_dis, :] = 0
        img_1[center + y_dis : self.n_points, :] = 0

        img_2[:, right - x_dis : right + x_dis] = 1
        img_2[0 : center - y_dis, :] = 0
        img_2[center + y_dis : self.n_points, :] = 0

        img_1 = ndimage.gaussian_filter(img_1, 4)
        img_2 = ndimage.gaussian_filter(img_2, 4)
        f_1 = interp.interp2d(grid, grid, img_1)
        f_2 = interp.interp2d(grid, grid, img_2)
        return f_1, f_2, img_1, img_2

    @staticmethod
    def _get_points_in_circle(radius, mid_x=0, mid_y=0):
        x_ = np.arange(mid_x - radius - 1, mid_x + radius + 1, dtype=int)
        y_ = np.arange(mid_y - radius - 1, mid_y + radius + 1, dtype=int)
        x, y = np.where((x_[:, np.newaxis] - mid_x) ** 2 + (y_ - mid_y) ** 2 <= radius ** 2)
        return x, y

    def __repr__(self):
        stim_name = self.__possible_stimuli__[self.stim_type]
        apply_name = self.__possible_applications__[self.apply_type]
        return "<{} {}>".format(stim_name, apply_name)
