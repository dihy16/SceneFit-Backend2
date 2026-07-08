"""
CLIP subspace projection utilities.
Canonical location: app/models/embedding/clip_subspace.py
(Backward-compat shim remains at app/services/clip_subspace.py)
"""
import torch


class ClipSubspace:
    def __init__(self, basis: torch.Tensor):
        """
        basis: [d, k] orthonormal columns
        """
        self.basis = basis

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x into the subspace.
        x: [n, d] or [d]
        """
        basis = self.basis.to(dtype=x.dtype, device=x.device)
        if x.dim() == 1:
            return basis @ (basis.T @ x)
        return x @ basis @ basis.T

    def remove(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the subspace component from x."""
        return x - self.project(x)


def orthogonalize_subspaces(
    primary: ClipSubspace,
    secondary: ClipSubspace,
) -> ClipSubspace:
    """Make secondary orthogonal to primary via QR decomposition."""
    B = primary.basis
    E = secondary.basis
    E_ortho = E - B @ (B.T @ E)
    Q, _ = torch.linalg.qr(E_ortho)
    return ClipSubspace(Q)
