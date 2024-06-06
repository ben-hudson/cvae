import tarfile
import torch
import pathlib
import typing

from collections import namedtuple
from sklearn.preprocessing import StandardScaler

SyntheticLPDatapoint = namedtuple("SyntheticLPDatapoint", "u c A b x f")

class SyntheticLPDataset(torch.utils.data.Dataset):
    def __init__(self, tarfile_path: pathlib.Path, slack: bool=True, norm: bool=True):
        with tarfile.open(tarfile_path, "r:gz") as tarball:
            A = self._load_tensor_from_tarball("A_eq.pt" if slack else "A_ub.pt", tarball)
            b = self._load_tensor_from_tarball("b_eq.pt" if slack else "b_ub.pt", tarball)
            c = self._load_tensor_from_tarball("c.pt", tarball)
            f = self._load_tensor_from_tarball("f.pt", tarball)
            u = self._load_tensor_from_tarball("u.pt", tarball)
            x = self._load_tensor_from_tarball("x.pt", tarball)

            self.is_normed = norm
            if self.is_normed:
                self.A_scaler = StandardScaler()
                self.A = torch.Tensor(self.A_scaler.fit_transform(A.flatten(1)).reshape(A.shape))

                self.b_scaler = StandardScaler()
                self.b = torch.Tensor(self.b_scaler.fit_transform(b))

                self.c_scaler = StandardScaler()
                self.c = torch.Tensor(self.c_scaler.fit_transform(c))

                self.f_scaler = StandardScaler()
                self.f = torch.Tensor(self.f_scaler.fit_transform(f))

                self.u_scaler = StandardScaler()
                self.u = torch.Tensor(self.u_scaler.fit_transform(u))

                self.x_scaler = StandardScaler()
                self.x = torch.Tensor(self.x_scaler.fit_transform(x))

    def _load_tensor_from_tarball(self, filename: str, tarball: tarfile.TarFile) -> torch.Tensor:
        info = tarball.getmember(filename)
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
