import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
# imported by lib.model.joint

def compute_gradient_penalty(model, X):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Random weight term for interpolation
    alpha = torch.Tensor(np.random.random(size=X.shape)).to(X.device)

    interpolates = alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size()).to(X.device)))
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = model(interpolates)

    fake = Variable(torch.Tensor(X.shape[0], 1).fill_(1.0), requires_grad=False).to(X.device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty