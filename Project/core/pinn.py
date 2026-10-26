import torch
import math


def poisson_residual(model, x):  # Residual για την Poisson: -(u_xx + u_yy) - f = 0
    x.requires_grad_(True)
    u = model(x)

    grads = torch.autograd.grad(  # Πρώτες παράγωγοι u_x και u_y
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True  # Διατήρηση του γράφου για τις δεύτερες παραγώγους
    )[0]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, x,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 1:2]

    f = 2 * math.pi**2 * torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])

    return -(u_xx + u_yy) - f


def poisson_residual_inverse(model, xy, alpha):  # Αντίστροφο πρόβλημα για τον προσδιορισμό του alpha
    u = model(xy)

    grads = torch.autograd.grad(
        u, xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, xy,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 1:2]

    laplace_u = u_xx + u_yy

    f = 2 * math.pi**2 * torch.sin(math.pi * xy[:, 0:1]) * torch.sin(math.pi * xy[:, 1:2])

    return -alpha * laplace_u - f


def poisson_residual_gauss(model, x, sigma):
    x.requires_grad_(True)
    u = model(x)

    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]  # N_f × 1
    u_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    # Για -Δu = f, το δεξί μέλος είναι το αρνητικό Laplacian της Gaussian
    x_c = x[:, 0:1] - 0.5
    y_c = x[:, 1:2] - 0.5
    r_squared = x_c**2 + y_c**2
    u_exact = torch.exp(-r_squared / (2 * sigma**2))
    f = u_exact * ((2.0 / sigma**2) - r_squared / (sigma**4))

    return -(u_xx + u_yy) - f


def poisson_residual_3d(model, x):
    x.requires_grad_(True)
    u = model(x)

    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_z = grads[:, 2:3]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    u_zz = torch.autograd.grad(u_z, x, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, 2:3]

    f = 3 * math.pi**2 * (
        torch.sin(math.pi * x[:, 0:1]) *
        torch.sin(math.pi * x[:, 1:2]) *
        torch.sin(math.pi * x[:, 2:3])
    )

    return -(u_xx + u_yy + u_zz) - f
