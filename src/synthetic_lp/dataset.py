import tarfile
import torch
import pathlib
import numpy as np

from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

SyntheticLPDatapoint = namedtuple("SyntheticLPDatapoint", "u c A b x f")

class SyntheticLPDataset(torch.utils.data.Dataset):
    def __init__(self, tarfile_path: pathlib.Path, dataset: str, include_slack: bool=False, norm: bool=True):
        dir = pathlib.Path(dataset)

        with tarfile.open(tarfile_path, "r:gz") as tarball:
            u = self._load_tensor_from_tarball(dir / "u.pt", tarball)
            c = self._load_tensor_from_tarball(dir / "c.pt", tarball).squeeze()
            A = self._load_tensor_from_tarball(dir / "A_ub.pt", tarball)
            b = self._load_tensor_from_tarball(dir / "b_ub.pt", tarball).squeeze()
            x = self._load_tensor_from_tarball(dir / "x.pt", tarball).squeeze()
            f = self._load_tensor_from_tarball(dir / "f.pt", tarball)

            # for an equality constraint, add the slack variables
            if include_slack:
                slack = self._load_tensor_from_tarball(dir / "slack.pt", tarball).squeeze()
                # append slack variables to x and adjust c, A appropriately
                x = torch.cat([x, slack], dim=1)
                c = torch.cat([c, torch.zeros_like(slack)], dim=1)
                I = torch.eye(slack.size(1)).expand(slack.size(0), -1, -1).double()
                A = torch.cat([A, I], dim=2)

        assert u.size(0) == c.size(0) == A.size(0) == b.size(0) == x.size(0) == f.size(0), \
            f'expected the same number of samples in u, c, A, b, x, f but got ({u.size(0), c.size(0), A.size(0), b.size(0), x.size(0), f.size(0)})'

        self.is_normed = norm
        if self.is_normed:
            self.means = SyntheticLPDatapoint(
                u.mean(dim=0, keepdim=True),
                c.mean(dim=0, keepdim=True),
                A.mean(dim=0, keepdim=True),
                b.mean(dim=0, keepdim=True),
                x.mean(dim=0, keepdim=True),
                f.mean(dim=0, keepdim=True),
            )
            self.scales = SyntheticLPDatapoint(
                u.std(dim=0, correction=0, keepdim=True),
                c.std(dim=0, correction=0, keepdim=True),
                A.std(dim=0, correction=0, keepdim=True),
                b.std(dim=0, correction=0, keepdim=True),
                x.std(dim=0, correction=0, keepdim=True),
                f.std(dim=0, correction=0, keepdim=True),
            )

            # prevent division by 0, just like sklearn.preprocessing.StandardScaler
            for scale in self.scales:
                scale[torch.isclose(scale, torch.zeros_like(scale))] = 1.

            for field, mean in self.means._asdict().items():
                assert not mean.requires_grad, f'{field} mean requires grad'

            for field, scale in self.scales._asdict().items():
                assert not scale.requires_grad, f'{field} scale requires grad'

            self.u, self.c, self.A, self.b, self.x, self.f = self.norm(u, c, A, b, x, f)

        else:
            self.u, self.c, self.A, self.b, self.x, self.f = u, c, A, b, x, f

    def norm(self, u, c, A, b, x, f):
        assert self.is_normed, "can't call norm on an unnormed dataset"
        u_normed = (u - self.means.u.to(u.device))/self.scales.u.to(u.device)
        c_normed = (c - self.means.c.to(c.device))/self.scales.c.to(c.device)
        A_normed = (A - self.means.A.to(A.device))/self.scales.A.to(A.device)
        b_normed = (b - self.means.b.to(b.device))/self.scales.b.to(b.device)
        x_normed = (x - self.means.x.to(x.device))/self.scales.x.to(x.device)
        f_normed = (f - self.means.f.to(f.device))/self.scales.f.to(f.device)

        return u_normed, c_normed, A_normed, b_normed, x_normed, f_normed

    def unnorm(self, u_normed, c_normed, A_normed, b_normed, x_normed, f_normed):
        assert self.is_normed, "can't call unnorm on an unnormed dataset"
        u = self.means.u.to(u_normed.device) + self.scales.u.to(u_normed.device)*u_normed
        c = self.means.c.to(c_normed.device) + self.scales.c.to(c_normed.device)*c_normed
        A = self.means.A.to(A_normed.device) + self.scales.A.to(A_normed.device)*A_normed
        b = self.means.b.to(b_normed.device) + self.scales.b.to(b_normed.device)*b_normed
        x = self.means.x.to(x_normed.device) + self.scales.x.to(x_normed.device)*x_normed
        f = self.means.f.to(f_normed.device) + self.scales.f.to(f_normed.device)*f_normed

        return u, c, A, b, x, f

    def _load_tensor_from_tarball(self, filepath: pathlib.Path, tarball: tarfile.TarFile) -> torch.Tensor:
        info = tarball.getmember(str(filepath))
        data = tarball.extractfile(info)
        tensor = torch.load(data)
        return tensor

    def __len__(self) -> int:
        return self.A.size(0)

    def __getitem__(self, index: int) -> SyntheticLPDatapoint:
        return SyntheticLPDatapoint(
            self.u[index],
            self.c[index],
            self.A[index],
            self.b[index],
            self.x[index],
            self.f[index]
        )

    @classmethod
    def render(cls, u, c, A, b, x_opt, f_opt, representation='polytope', mode='rgb_array'):
        if representation == 'polytope':
            return cls._render_polytope(u, c, A, b, x_opt, f_opt, mode=mode)
        elif representation == 'extended_matrix':
            return cls._render_extended_matrix(u, c, A, b, x_opt, f_opt, mode=mode)
        else:
            raise ValueError(f'representation={representation} is not implemented.')

    # assemble a problem into
    # [0 c^T 0]
    # [0 I   x]
    # [1 -A  b]
    @classmethod
    def _render_extended_matrix(cls, u, c, A, b, x_opt, f_opt, mode='rgb_array'):
        assert mode == 'rgb_array', f'mode={mode} is not implemented.'

        first_row = torch.cat([torch.zeros(1, 1, device=c.device), c, torch.zeros(1, 1, device=c.device)], dim=1)
        secnd_row = torch.cat([torch.zeros_like(x_opt), torch.eye(x_opt.size(0), device=x_opt.device), x_opt], dim=1)
        third_row = torch.cat([torch.ones_like(b), -A, b], dim=1)
        array = torch.cat([first_row, secnd_row, third_row])

        rgb_array = torch.stack([array, array, array])
        return rgb_array

    # render the polytope (for 2D problems)
    @classmethod
    def _render_polytope(cls, u, c, A, b, x_opt, f_opt, mode='rgb_array'):
        fig = plt.gcf()
        fig.clear()

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        cls._draw_polytope(u, c, A, b, x_opt, f_opt, ax)

        if mode == 'human':
            plt.ion()
            fig.canvas.draw()
            fig.show()

        elif mode == 'rgb_array':
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            img = data.reshape((int(h), int(w), -1))
            return img

        else:
            raise ValueError(f'mode={mode} is not implemented.')

    @classmethod
    def _draw_polytope(cls, u, c, A, b, x_opt, f_opt, ax, x_lim=None, y_lim=None):
        if x_lim is None:
            x_lim = ax.get_xlim()
        if y_lim is None:
            y_lim = ax.get_ylim()

        x = np.linspace(*x_lim, 1000)
        y = np.linspace(*y_lim, 1000)
        xx, yy = np.meshgrid(x, y)

        ff = c[0]*xx + c[1]*yy

        feasible_mask = np.ones_like(ff, dtype=bool)
        for coeffs, intercept in zip(A, b):
            constr_mask = coeffs[0]*xx + coeffs[1]*yy <= intercept
            feasible_mask &= constr_mask.bool().numpy()

            ax.plot(x, (intercept - coeffs[0]*x) / coeffs[1], ls='--', linewidth=1, alpha=0.5, color='k')

        ff[~feasible_mask] = np.nan
        ax.contourf(xx, yy, ff)

        ax.plot(*x_opt, marker='^', color='k')
