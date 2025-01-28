import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import jacrev, vmap

# Define MLP layer
def mlp(sizes, activation, output_activation=nn.Identity):
    # declare network
    layers = []
    for j in range(0, len(sizes) - 2):
        layers += [nn.Linear(sizes[j], sizes[j+1]),
                    activation()]
    layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
    net = nn.Sequential(*layers)
    
    # init weight
    for i in range(len(net)):
            if isinstance(net[i], nn.Linear):
                if isinstance(net[i+1], nn.ReLU):
                    nn.init.kaiming_normal_(net[i].weight, nonlinearity='relu')
                elif isinstance(net[i+1], nn.LeakyReLU):
                    nn.init.kaiming_normal_(net[i].weight, nonlinearity='leaky_relu')
                else:
                    nn.init.xavier_normal_(net[i].weight)
    return net

# Define LipsNet++
class LipsNet_v2(nn.Module):
    def __init__(self, sizes, obs_dim, obs_len, activation, output_activation=nn.Tanh,
                 lambda_t=0.1, lambda_k=0., kernel_scale=0.02, enable_fft_2d=True, norm_layer_type="none") -> None:
        super().__init__()

        # declare network
        self.enable_fft_2d = enable_fft_2d
        self.norm_layer_type = norm_layer_type
        if enable_fft_2d:
            if norm_layer_type == "batch_norm":
                self.norm_layer = nn.BatchNorm1d(obs_dim)
            elif norm_layer_type == "layer_norm":
                self.norm_layer = nn.LayerNorm(obs_dim)
            self.filter_kernel = nn.Parameter(torch.cat([
                torch.ones(obs_len, obs_dim//2 + 1, 1, dtype=torch.float32),
                torch.randn(obs_len, obs_dim//2 + 1, 1, dtype=torch.float32) * kernel_scale],
                dim=2
            ))
        else:
            self.filter_kernel = nn.Parameter(torch.cat([
                torch.ones(obs_len//2 + 1, obs_dim, 1, dtype=torch.float32),
                torch.randn(obs_len//2 + 1, obs_dim, 1, dtype=torch.float32) * kernel_scale],
                dim=2
            ))
        self.mlp = mlp(sizes, activation, output_activation)

        # loss weight
        self.lambda_t = lambda_t
        self.lambda_k = lambda_k
        # observation info
        self.obs_len = obs_len
        self.obs_dim = obs_dim
        
        # auto_adjust lips init
        self.regularization_loss = 0
        self.register_full_backward_pre_hook(backward_hook)

    def fft_2d(self, x):
        if self.norm_layer_type == "batch_norm":
            x = self.norm_layer(x.reshape(-1,self.obs_dim)).reshape(x.shape)
        elif self.norm_layer_type == "layer_norm":
            x = self.norm_layer(x)
        x_f = torch.fft.rfft2(x, s=(self.obs_len, self.obs_dim), dim=(1, 2), norm='ortho')
        kernal = torch.view_as_complex(self.filter_kernel)
        x_f = x_f * kernal
        x_f = torch.fft.irfft2(x_f, s=(self.obs_len, self.obs_dim), dim=(1, 2), norm='ortho')
        return x_f

    def fft_1d(self, x):
        x_f = torch.fft.rfft(x, n=self.obs_len, dim=1, norm='ortho')
        kernal = torch.view_as_complex(self.filter_kernel)
        x_f = x_f * kernal
        x_f = torch.fft.irfft(x_f, n=self.obs_len, dim=1, norm='ortho')
        return x_f

    def forward(self, x):
        # forward
        if self.enable_fft_2d:
            x_f = self.fft_2d(x)
        else:
            x_f = self.fft_1d(x)
        x_result = x_f[..., 0, :]
        out = self.mlp(x_result)
        
        # if need store regularization loss
        if self.training and out.requires_grad:
            # calcute jac matrix
            jacobi = vmap(jacrev(self.mlp))(x_result.detach())
            # jacobi.dim: (x.shape[0], f_out.shape[1], x.shape[1])
            #             (batch     , f output dim  , x intput dim)
            # calcute jac norm
            norm = torch.norm(jacobi, 2, dim=(1,2)).unsqueeze(1)
            self.regularization_loss += self.lambda_t * (self.filter_kernel ** 2).sum() \
                                      + self.lambda_k * norm.mean()
            # self.regularization_loss += self.loss_lambda * self.filter_kernel.abs().sum()
        return out
    
def backward_hook(module, gout):
    if not isinstance(module.regularization_loss, int):
        module.regularization_loss.backward(retain_graph=True)
        module.regularization_loss = 0
    return gout


if __name__ == "__main__":
    net = LipsNet_v2(sizes=[2,2], obs_dim=10, obs_len=10, activation=nn.GELU)

    print(net.parameters())

    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())

    exit()