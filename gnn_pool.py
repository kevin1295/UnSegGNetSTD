import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GNNpool(nn.Module):
    def __init__(self, input_dim, conv_hidden, mlp_hidden, num_clusters, device,
                 activ="silu", loss_type="DMON", conv_type="GCN",
                 alpha=2.0, lam=1.0, entropy_epsilon=3.0, nb_iterations=3):
        super(GNNpool, self).__init__()
        self.device = device
        self.activ = activ
        self.num_clusters = num_clusters
        self.mlp_hidden = mlp_hidden
        self.nb_iterations = nb_iterations

        # ========== STD 可学习参数（论文标准初始化）==========
        if activ == 'STD':
            self.alpha = nn.Parameter(torch.tensor([alpha], device=device), requires_grad=True)
            self.lam = nn.Parameter(torch.tensor([lam], device=device), requires_grad=True)
            self.entropy_epsilon = nn.Parameter(torch.tensor([entropy_epsilon], device=device), requires_grad=True)

        if loss_type not in ["DMON", "NCUT"]:
            raise ValueError(f'Loss type: {loss_type} is not supported')
        self.loss_type = loss_type

        # 激活函数
        if activ == "deepcut_activation":
            act = 'relu'
            nn_activ = nn.ReLU()
            self.f_act = F.elu
        elif activ == "relu":
            act = 'relu'
            nn_activ = nn.ReLU()
            self.f_act = F.relu
        elif activ == "silu":
            act = 'silu'
            nn_activ = nn.SiLU()
            self.f_act = F.silu
        elif activ == "gelu":
            act = 'gelu'
            nn_activ = nn.GELU()
            self.f_act = F.gelu
        elif activ == "selu":
            act = 'selu'
            nn_activ = nn.SELU()
            self.f_act = F.selu
        elif activ == 'STD':
            act = 'selu'
            nn_activ = nn.SELU()
            self.f_act = F.selu
        else:
            raise ValueError("Activation function not supported")

        # GCN
        if conv_type == "GCN":
            self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act=act)
        else:
            raise ValueError("Conv type not supported")

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn_activ, nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters)
        )

    # ========== 拉普拉斯 对称归一化 L_sym ==========
    def get_laplacian(self, A):
        degree = A.sum(dim=1).clamp(min=1e-8)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree))
        L_sym = torch.eye(A.size(0), device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        return L_sym

    def forward(self, data, A):
        x, edge_index, edge_atrr = data.x, data.edge_index, data.edge_attr
        x = self.convs(x, edge_index, edge_atrr)
        x = self.f_act(x)

        # MLP
        H = self.mlp(x)
        
        # ========== STD 迭代 ==========
        if self.activ == 'STD':
            L_sym = self.get_laplacian(A)
            u = torch.softmax(H, dim=-1)

            for _ in range(self.nb_iterations):
                # 论文公式：p = λ L (1 - 2u) + 2α u
                p = self.lam * torch.matmul(L_sym, (1.0 - 2.0 * u))
                p = p + 2 * self.alpha * u
                u = torch.softmax((H - p) / self.entropy_epsilon, dim=-1)

            S = u
        else:
            S = F.softmax(H, dim=-1)
            
        return A, S

    def loss(self, A, S):
        if self.loss_type == "NCUT":
            A_pool = torch.matmul(torch.matmul(A, S).t(), S)
            num = torch.trace(A_pool)
            D = torch.diag(torch.sum(A, dim=-1))
            D_pooled = torch.matmul(torch.matmul(D, S).t(), S)
            den = torch.trace(D_pooled)
            mincut_loss = -(num / den)
            St_S = torch.matmul(S.t(), S)
            I_S = torch.eye(self.num_clusters, device=self.device)
            ortho_loss = torch.norm(St_S / torch.norm(St_S) - I_S / torch.norm(I_S))
            return mincut_loss + ortho_loss

        elif self.loss_type == "DMON":
            C = S
            d = torch.sum(A, dim=1)
            m = torch.sum(A)
            B = A - torch.ger(d, d) / (2 * m)
            I_S = torch.eye(self.num_clusters, device=self.device)
            k = torch.norm(I_S)
            n = S.shape[0]
            modularity_term = (-1/(2*m)) * torch.trace(C.t() @ B @ C)
            collapse_reg_term = (torch.sqrt(k)/n) * torch.norm(torch.sum(C.t(), dim=0), p='fro') - 1
            return modularity_term + collapse_reg_term