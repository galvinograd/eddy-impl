from dataclasses import dataclass

from alg_eddy import EDDYGuidance
from embedding import VisionExtractor


def _clamp_norm(V):
    d = V[0].numel()
    max_norm = d ** 0.5
    flat = V.reshape(V.shape[0], -1)
    norms = flat.norm(dim=1)
    scale = (max_norm / norms.clamp(min=max_norm)).view(-1, *([1] * (V.dim() - 1)))
    return V * scale


@dataclass
class EDDYConfig:
    wg: float
    stop_step_ratio: float
    hutchinson_samples: int
    fd_eps: float
    gamma: float


class EDDYCallback:
    def __init__(self, config: EDDYConfig, num_steps: int, feature_extractor: VisionExtractor, model: str = 'flux'):
        if model not in ('flux', 'sdxl'):
            raise ValueError(f"Unknown model: {model!r}")
        self.config = config
        self.num_steps = num_steps
        self.feature_extractor = feature_extractor
        self.model = model
        self.guidance = EDDYGuidance(
            feature_extractor,
            gamma=config.gamma,
            fd_eps=config.fd_eps,
            hutchinson_samples=config.hutchinson_samples,
        )

    def set_config(self, config: EDDYConfig):
        self.config = config
        self.guidance.gamma = config.gamma
        self.guidance.fd_eps = config.fd_eps
        self.guidance.hutchinson_samples = config.hutchinson_samples

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        X = callback_kwargs["latents"]
        B = X.shape[0]

        if B <= 1 or step_index == 0:
            return callback_kwargs
        if step_index / self.num_steps > self.config.stop_step_ratio:
            return callback_kwargs

        sigmas = pipeline.scheduler.sigmas
        sigma_curr = sigmas[step_index].item()
        sigma_next = sigmas[step_index + 1].item() if step_index + 1 < len(sigmas) else 0.0
        dt = abs(sigma_next - sigma_curr)

        if sigma_curr < 1e-5:
            return callback_kwargs

        V = callback_kwargs["noise_pred"]
        V, S = self._score_and_velocity(callback_kwargs["latents"], V, sigma_curr, sigma_next)
        V = _clamp_norm(V)

        guidance = self.guidance.compute_guidance(X, V, S)
        callback_kwargs["latents"] = X + guidance.to(X.dtype) * (self.config.wg * dt * B / X.numel())
        return callback_kwargs

    def _score_and_velocity(self, Y_post, V, sigma_curr, sigma_next):
        if self.model == 'sdxl':
            epsilon = V.float()
            S = -epsilon / sigma_curr
            velocity = -epsilon
            return velocity, S

        Y_pre   = Y_post - (sigma_next - sigma_curr) * V
        eps_hat = Y_pre.float() + (1.0 - sigma_curr) * V.float()
        S   = -eps_hat / sigma_curr
        return V, S
