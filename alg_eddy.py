import torch


class EDDYGuidance:
    def __init__(
        self, 
        feature_extractor, 
        gamma: float, 
        fd_eps: float,
        hutchinson_samples: int,
    ):
        self.feature_extractor = feature_extractor
        self.gamma = gamma
        self.fd_eps = fd_eps
        self.hutchinson_samples = hutchinson_samples

    def _neg_kernel_grads(self, x_i, Y, gamma, exclude_idx):
        x_ig = x_i.detach().clone().requires_grad_(True)
        y_i  = self.feature_extractor.forward(x_ig)
        r_js = {}
        for j, y_j in enumerate(Y.unbind(0)):
            if j == exclude_idx:
                continue
            neg_k_ij = -torch.exp(-(y_i - y_j).pow(2).sum() / gamma)
            (r_j,) = torch.autograd.grad(neg_k_ij, x_ig, retain_graph=True)
            r_js[j] = r_j.detach()
        del y_i
        return r_js

    def _hutchinson_traces(self, x_i, Y, gamma, exclude_idx):
        eps = self.fd_eps
        m   = self.hutchinson_samples
        rads = torch.randint(0, 2, (m, *x_i.shape[1:]), device=x_i.device, dtype=x_i.dtype) * 2 - 1

        # central fd probes for double differentiation
        x_i_probs = torch.cat(
            [x_i.detach()]
            + [x_i.detach() + eps * rads[k:k+1] for k in range(m)]
            + [x_i.detach() - eps * rads[k:k+1] for k in range(m)],
            dim=0,
        )  # (2m+1, ...)
        with torch.no_grad():
            y_i_probs = self.feature_extractor.forward(x_i_probs)  # (2m+1, D)

        y_i = y_i_probs[0:1]

        trace_js = {}
        for j, y_j in enumerate(Y.unbind(0)):
            if j == exclude_idx:
                continue
            
            # hutchinson estimator over central fd per probe
            f0 = torch.exp(-(y_i - y_j).pow(2).sum() / gamma).item()
            trace_js[j] = sum(
                (
                    torch.exp(-(y_i_probs[1+a:2+a] - y_j).pow(2).sum() / gamma).item()
                    + torch.exp(-(y_i_probs[1+m+a:2+m+a] - y_j).pow(2).sum() / gamma).item()
                    - 2 * f0
                ) / eps ** 2
                for a in range(m)
            ) / m

        return trace_js

    def _hvp_terms(self, x_i, V, Y, gamma, exclude_idx):
        eps    = self.fd_eps
        j_list = [j for j in range(Y.shape[0]) if j != exclude_idx]

        # central fd probes for single differentiation
        x_i_probes = torch.cat(
            [x_i.detach() + eps * V[j:j+1] for j in j_list]
            + [x_i.detach() - eps * V[j:j+1] for j in j_list],
            dim=0,
        ).requires_grad_(True)
        y_j_doubled = torch.cat([Y[j:j+1] for j in j_list] * 2, dim=0).detach()

        # gradient of probes
        y_i_probes    = self.feature_extractor.forward(x_i_probes)
        loss     = torch.exp(-(y_i_probes - y_j_doubled).pow(2).sum(dim=-1) / gamma).sum()
        (g_hvp,) = torch.autograd.grad(loss, x_i_probes)

        # central fd estimator for hvp
        n_j = len(j_list)
        return {
            j: ((g_hvp[k:k+1] - g_hvp[n_j+k:n_j+k+1]) / (2 * eps)).detach()
            for k, j in enumerate(j_list)
        }

    def _accumulate_particle_guidance(self, i, j_list, r_js, trace_js, hvp_js, V, S):
        s_i = S[i:i+1]
        acc = torch.zeros_like(s_i)
        for j in j_list:
            r_dot_si  = (r_js[j].reshape(-1) * s_i.reshape(-1)).sum()
            si_dot_vj = (s_i.reshape(-1) * V[j:j+1].reshape(-1)).sum()
            acc += hvp_js[j] - (trace_js[j] * V[j:j+1]) + V[j:j+1] * r_dot_si - r_js[j] * si_dot_vj
        return acc / (len(j_list))

    @torch.enable_grad()
    def compute_guidance(self, X: torch.Tensor, V: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        B  = X.shape[0]
        gamma = torch.tensor(self.gamma, dtype=X.dtype, device=X.device)
        guidance = torch.zeros_like(X)

        with torch.no_grad():
            Y = self.feature_extractor.forward(X)  # (B, D)

        j_lists = [[j for j in range(B) if j != i] for i in range(B)]

        for i in range(B):
            x_i    = X[i:i+1]
            j_list = j_lists[i]

            r_js     = self._neg_kernel_grads(x_i, Y, gamma, exclude_idx=i)
            trace_js = self._hutchinson_traces(x_i, Y, gamma, exclude_idx=i)
            hvp_js   = self._hvp_terms(x_i, V, Y, gamma, exclude_idx=i)

            guidance[i:i+1] = self._accumulate_particle_guidance(i, j_list, r_js, trace_js, hvp_js, V, S)

        return guidance
