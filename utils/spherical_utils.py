import torch


# ##################Functions for spherical space ###########################################################
def sphere_sum_c(x, y, c):
    if c < 0:
        raise ValueError("error in sphere_sum_c")
    else:
        sqxnorm_c = c * torch.sum(x * x, dim=-1, keepdim=True)
        sqynorm_c = c * torch.sum(y * y, dim=-1, keepdim=True)

    dotxy = torch.sum(x * y, dim=-1, keepdim=True)
    numerator = (1 - 2 * c * dotxy - sqynorm_c) * x + (1 + sqxnorm_c) * y
    denominator = 1 - 2 * c * dotxy + sqxnorm_c * sqynorm_c
    return numerator / denominator


def sphere_exp_map_c(v, c):
    normv_c = torch.clamp(
        torch.sqrt(torch.abs(c)) * torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10
    )
    return torch.tan(normv_c) * v / (normv_c)


def sphere_sqdist(
    p1, p2, c
):  # c is positive in projected hypersphere space, negative in poincare ball.
    sqrt_c = torch.sqrt(torch.abs(c))
    dist = torch.atan(sqrt_c * torch.norm(sphere_sum_c(-p1, p2, c), 2, dim=-1))
    sqdist = ((dist * 2 / sqrt_c) ** 2).clamp(max=75)
    return sqdist


def full_sphere_exp_map_c(x, v, c):  # tangent space of an reference point x
    normv_c = torch.clamp(
        torch.sqrt(torch.abs(c)) * torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10
    )
    if c < 0:
        raise ValueError("error in full_sphere_exp_map_c")
    else:
        sqxnorm_c = c * torch.sum(x * x, dim=-1, keepdim=True)
    y = torch.tan(normv_c / (1 + sqxnorm_c)) * v / (normv_c)
    return sphere_sum_c(x, y, c)


def sphere_log_map_c(v, c):
    normv = torch.clamp(
        torch.norm(v, 2, dim=-1, keepdim=True), 1e-10
    )  # we need clamp here because we need to divide the norm.
    sqrt_c = torch.sqrt(torch.abs(c))
    normv_c = sqrt_c * normv

    return 1.0 / sqrt_c * torch.atan(normv_c) * (v / normv)
