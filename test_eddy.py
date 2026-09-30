from functools import partial

import torch

from alg_eddy import EDDYGuidance

torch.set_default_dtype(torch.float64)


_GAMMA = 0.1


class _FnExtractor:
    def __init__(self, fn):
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def identity_extractor():
    return _FnExtractor(lambda x: x)


def projective_extractor():
    return _FnExtractor(lambda x: x / torch.clamp_min(torch.linalg.norm(x, dim=-1, keepdim=True), 1e-6))


def random_nonlinear_extractor(d_in, d_out, seed):
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn(d_in, d_out, generator=g)
    W2 = torch.randn(d_out, d_out, generator=g)
    return _FnExtractor(lambda x: torch.tanh(x @ W1) @ W2)


def eddy_s_rbf(X: torch.Tensor, V: torch.Tensor, score: torch.Tensor, gamma: float = _GAMMA) -> torch.Tensor:
    # x_hat_i = x_i / ||x_i||
    Xnorm = torch.linalg.norm(X, dim=1, keepdim=True)
    Xhat = X / Xnorm

    # z_ij = <x_hat_i, x_hat_j>
    z = Xhat @ Xhat.T

    # w_ij = x_hat_j - z_ij * x_hat_i
    w = Xhat[None, :, :] - z[:, :, None] * Xhat[:, None, :]

    # k_ij = exp(-||x_hat_i - x_hat_j||^2 / gamma) = exp(2 (z_ij - 1) / gamma)
    k = torch.exp(2 * (z - 1) / gamma)

    d = X.shape[1]
    n = X.shape[0]

    # C_ij^w = <(2/(gamma*||x_i||)) w_ij - (1/||x_i||) x_hat_i + s_i, v_j>
    Xnorm3 = Xnorm[:, :, None]
    term_w = (2 / gamma) * w / Xnorm3 - Xhat[:, None, :] / Xnorm3 + score[:, None, :]
    C_w = torch.einsum('ijd,jd->ij', term_w, V)

    # C_ij^v = (1/||x_i||)((d-2) z_ij - (2/gamma)(1-z_ij^2)) - <w_ij, s_i>
    C_v = (1 / Xnorm) * ((d - 2) * z - (2 / gamma) * (1 - z ** 2)) - torch.einsum('ijd,id->ij', w, score)

    # C_ij^xhat = (1/||x_i||)(z_ij <x_hat_i, v_j> - <w_ij, v_j>)
    C_xhat = (1 / Xnorm) * (z * torch.einsum('id,jd->ij', Xhat, V) - torch.einsum('ijd,jd->ij', w, V))

    # psi_i = (2/(gamma*||x_i||*(n-1))) sum_{i!=j} k_ij (C_ij^w w_ij + C_ij^v v_j + C_ij^xhat x_hat_i)
    psi_mat = k[:, :, None] * (
        C_w[:, :, None] * w + C_v[:, :, None] * V[None, :, :] + C_xhat[:, :, None] * Xhat[:, None, :]
    )
    psi_mask = torch.ones((n, n), dtype=X.dtype) - torch.eye(n, dtype=X.dtype)
    psi = 2 * (psi_mask[:, :, None] * psi_mat).sum(dim=1) / (gamma * Xnorm * (n - 1))

    return psi


def eddy_rbf(X: torch.Tensor, V: torch.Tensor, score: torch.Tensor, gamma: float = _GAMMA) -> torch.Tensor:
    # delta_ij = x_i - x_j
    delta = X[:, None, :] - X[None, :, :]

    # cdist square
    cdist_sq = (delta ** 2).sum(dim=2)

    # k_ij = exp(-||delta_ij||^2/gamma)
    k = torch.exp(-cdist_sq / gamma)

    # C_ij^delta = (2/gamma) * <delta_ij, v_j> - <s_i, v_j>
    C_delta = 2 * torch.einsum('ijd,jd->ij', delta, V) / gamma - score @ V.T

    # C_ij^v = d - 1 - (2/gamma)||delta_ij||^2 + <delta_ij, s_i>
    d = X.shape[1]
    C_v = d - 1 - 2 * cdist_sq / gamma + torch.einsum('ijd,id->ij', delta, score)

    # psi_i = (2/gamma*(n-1)) * sum_{i!=j} k_ij (C_ij^delta * delta_ij + C_ij^v * v_j)
    n = X.shape[0]
    psi_mat = k[:, :, None] * (C_delta[:, :, None] * delta + C_v[:, :, None] * V[None, :, :])
    psi_mask = torch.ones((n, n), dtype=X.dtype) - torch.eye(n, dtype=X.dtype)
    psi = 2 * (psi_mask[:, :, None] * psi_mat).sum(dim=1) / (gamma * (n - 1))

    return psi


def eddy_autograd(
    X: torch.Tensor,
    V: torch.Tensor,
    S: torch.Tensor,
    feature_extractor,
    gamma: float = _GAMMA) -> torch.Tensor:

    def _h(x):
        return feature_extractor.forward(x.unsqueeze(0))[0]

    def _k(x, y):
        return torch.exp(-(_h(x) - _h(y)).pow(2).sum() / gamma)

    def _grad_k(x, y):
        # create_graph=True: _A (used below) differentiates through this
        # jacobian again, so its graph must stay live for the outer call.
        return torch.autograd.functional.jacobian(lambda x_: _k(x_, y), x, create_graph=True)

    def _A(x, y, v):
        r = -_grad_k(x, y)
        return torch.outer(v, r) - torch.outer(r, v)

    def _div_A(x, y, v):
        jac = torch.autograd.functional.jacobian(lambda x_: _A(x_, y, v), x)  # jac[i, j, e] = d A[i,j] / d x[e]
        return torch.einsum('ijj->i', jac)  # sum_j d A[i,j] / d x[j]

    n, d = X.shape
    psi = torch.zeros((n, d), dtype=X.dtype)
    for i in range(n):
        acc = torch.zeros(d, dtype=X.dtype)
        x_i = X[i].detach()
        for j in range(n):
            if i == j:
                continue
            x_j = X[j].detach()
            A = _A(x_i, x_j, V[j])
            acc += _div_A(x_i, x_j, V[j]) + A @ S[i]
        psi[i] = acc / (n - 1)
    return psi


def self_test():
    header = f"{'name':<32} {'max rel diff':^14} {'result':^10}"
    print(header)

    def _assert(name, actual, reference, tol, eps=1e-12):
        # Relative error, taken per sample so that draws where psi is tiny are
        # judged against their own scale instead of being masked by draws where
        # it is O(10). The eps floor covers draws whose reference underflows to
        # ~0: both sides are then ~0, so the ratio stays ~0 rather than blowing
        # up, while a genuinely wrong actual still registers as huge.
        diff = (actual - reference).abs()
        ref = reference.abs()
        if actual.ndim == 3:
            num = diff.flatten(1).max(dim=1).values
            den = ref.flatten(1).max(dim=1).values
        else:
            num, den = diff.max().reshape(1), ref.max().reshape(1)
        max_rel_diff = (num / den.clamp_min(eps)).max()
        is_pass = max_rel_diff < tol
        print(f"{name:<32} {max_rel_diff:^14.2e} {'PASS' if is_pass else 'FAIL':^10}")
        return is_pass

    all_pass = True
    SAMPLES = 20
    TOL = 0.2
    N = 4
    FD_EPS = 1e-4
    
    def _assert_match(name, target, blueprint, dim):
        # randomize X, V and scores
        X = torch.randn(SAMPLES, N, dim)
        V = torch.randn(SAMPLES, N, dim)
        score = torch.randn(SAMPLES, N, dim)
        
        # calculate vector fields
        actual = torch.stack([
            target(X[i,:,:], V[i,:,:], score[i,:,:])
            for i in range(SAMPLES)
        ])
        reference = torch.stack([
            blueprint(X[i,:,:], V[i,:,:], score[i,:,:])
            for i in range(SAMPLES)
        ])
        # assert
        return _assert(name, actual, reference, TOL)

    # consistency with identity
    target = partial(eddy_autograd, feature_extractor=identity_extractor())
    all_pass &= _assert_match('eddy-rbf formula verified 1', target, eddy_rbf, 2)
    all_pass &= _assert_match('eddy-rbf formula verified 2', target, eddy_rbf, 3)
    all_pass &= _assert_match('eddy-rbf formula verified 3', target, eddy_rbf, 10)
    
    # consistency with projective
    target = partial(eddy_autograd, feature_extractor=projective_extractor())
    all_pass &= _assert_match('eddy-s-rbf formula verified 1', target, eddy_s_rbf, 2)
    all_pass &= _assert_match('eddy-s-rbf formula verified 2', target, eddy_s_rbf, 3)
    all_pass &= _assert_match('eddy-s-rbf formula verified 3', target, eddy_s_rbf, 10)
    
    # identity
    target = EDDYGuidance(
        identity_extractor(),
        gamma=_GAMMA,
        fd_eps=FD_EPS,
        hutchinson_samples=1_000
    )
    all_pass &= _assert_match('eddy approx agrees with eddy-rbf 1', target.compute_guidance, eddy_rbf, 3)
    all_pass &= _assert_match('eddy approx agrees with eddy-rbf 2', target.compute_guidance, eddy_rbf, 4)
    all_pass &= _assert_match('eddy approx agrees with eddy-rbf 3', target.compute_guidance, eddy_rbf, 50)
    
    # projective
    target = EDDYGuidance(
        projective_extractor(),
        gamma=_GAMMA,
        fd_eps=FD_EPS,
        hutchinson_samples=1_000
    )
    all_pass &= _assert_match('eddy approx agrees with eddy-s-rbf 1', target.compute_guidance, eddy_s_rbf, 3)
    all_pass &= _assert_match('eddy approx agrees with eddy-s-rbf 2', target.compute_guidance, eddy_s_rbf, 4)
    target.hutchinson_samples = 5_000
    all_pass &= _assert_match('eddy approx agrees with eddy-s-rbf 3', target.compute_guidance, eddy_s_rbf, 50)
    
    # random
    fe = random_nonlinear_extractor(3, 3, 42)
    target = EDDYGuidance(
        fe,
        gamma=_GAMMA,
        fd_eps=FD_EPS,
        hutchinson_samples=1_000
    )
    blueprint = partial(eddy_autograd, feature_extractor=fe)
    all_pass &= _assert_match('eddy approx agrees with eddy autograd 1', target.compute_guidance, blueprint, 3)
    
    fe = random_nonlinear_extractor(4, 4, 42)
    target.feature_extractor = fe
    blueprint = partial(eddy_autograd, feature_extractor=fe)
    all_pass &= _assert_match('eddy approx agrees with eddy autograd 2', target.compute_guidance, blueprint, 4)
    
    fe = random_nonlinear_extractor(50, 10, 42)
    target.feature_extractor = fe
    blueprint = partial(eddy_autograd, feature_extractor=fe)
    all_pass &= _assert_match('eddy approx agrees with eddy autograd 3', target.compute_guidance, blueprint, 50)

    print("ALL PASS" if all_pass else "SOME FAILED")
    return all_pass


if __name__ == "__main__":
    self_test()
