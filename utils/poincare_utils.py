import torch


# ############Functions for Poincare space#################################################
def artanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def poincare_exp_map_c(v, c):
    normv = torch.clamp(
        torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10
    )  # we need clamp here because we need to divide the norm.
    normv_c = torch.clamp(torch.sqrt(c) * normv, min=1e-10)
    return torch.tanh(normv_c) * v / (normv_c)


def poincare_log_map_c(v, c):
    normv = torch.clamp(
        torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1 - 1e-5
    )  # we need clamp here because we need to divide the norm.
    sqrt_c = torch.sqrt(c)
    if sqrt_c.detach().item() < 1e-10:
        raise ValueError("sqrt of curvature smaller than 1e-10")
    normv_c = torch.clamp(sqrt_c * normv, max=1 - 1e-5)

    return 1 / sqrt_c * artanh(normv_c) * v / normv


def poincare_sum_c(x, y, c):
    sqxnorm_c = torch.clamp(c * torch.sum(x * x, dim=-1, keepdim=True), 0, 1 - 1e-5)
    sqynorm_c = torch.clamp(c * torch.sum(y * y, dim=-1, keepdim=True), 0, 1 - 1e-5)
    dotxy = torch.sum(x * y, dim=-1, keepdim=True)
    numerator = (1 + 2 * c * dotxy + sqynorm_c) * x + (1 - sqxnorm_c) * y
    denominator = 1 + 2 * c * dotxy + sqxnorm_c * sqynorm_c
    return numerator / denominator


def full_poincare_exp_map_c(x, v, c):  # tangent space of an reference point x
    normv_c = torch.clamp(
        torch.sqrt(c) * torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10
    )
    sqxnorm_c = torch.clamp(
        c * torch.sum(x * x, dim=-1, keepdim=True), 0, 1 - 1e-5
    )  # we need clamp here because we need to divide the norm.
    y = torch.tanh(normv_c / (1 - sqxnorm_c)) * v / (normv_c)
    return poincare_sum_c(x, y, c)


def poincare_sqdist(
    p1, p2, c
):  # c is positive in projected hypersphere space, negative in poincare ball.
    sqrt_c = torch.sqrt(torch.abs(c))
    dist = artanh(
        torch.clamp(
            sqrt_c * torch.norm(poincare_sum_c(-p1, p2, c), 2, dim=-1), 1e-10, 1 - 1e-5
        )
    )
    sqdist = (2.0 * dist / sqrt_c) ** 2
    return sqdist


def poincare_cosh_sqdist(
    p1, p2, c
):  # c is positive in projected hypersphere space, negative in poincare ball.
    sqrt_c = torch.sqrt(torch.abs(c))
    dist = artanh(
        torch.clamp(
            sqrt_c * torch.norm(poincare_sum_c(-p1, p2, c), 2, dim=-1), 1e-10, 1 - 1e-5
        )
    )
    cosh_sqdist = torch.cosh(2.0 * dist / sqrt_c) ** 2
    return cosh_sqdist
