import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from .resnet18 import resnet18


# from utils import weights_init

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

########## CNN #############################
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class ConvVAE(nn.Module):
    def __init__(self, configs, latent_size=8):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=configs.input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * (configs.sequence_len // 4), latent_size * 2)  # *2 for mean and variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128 * (configs.sequence_len // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (128, configs.sequence_len // 4)),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=configs.input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Assuming input data is normalized between 0 and 1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        # Encode
        encoding_params = self.encoder(x)
        mu, log_var = encoding_params[:, :self.latent_size], encoding_params[:, self.latent_size:]

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var

    def projection(self, x):
        encoding_params = self.encoder(x)
        mu, log_var = encoding_params[:, :self.latent_size], encoding_params[:, self.latent_size:]
        return self.reparameterize(mu, log_var)

    def loss(self, x, x_dec, mu, logvar):
        criterion = lambda x1, x2: torch.sum((x1 - x2) ** 2,1).mean()  # nn.BCELoss(reduction='sum')  # Binary cross-entropy loss for reconstruction
        kl_divergence = lambda mu, log_var: -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        bce = criterion(x, x_dec)
        kl = kl_divergence(mu, logvar)
        return bce, kl


class CNN_VAE(nn.Module):
    def __init__(self, configs):
        super(CNN_VAE, self).__init__()

        self.configs = configs
        print(configs.kernel_size)
        self.encConv1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 8, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.encConv2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.enc_muFC1 = nn.Linear(16*(configs.sequence_len//2//2+1+1), configs.sequence_len)
        self.enc_muFC2 = nn.Linear(configs.sequence_len, 16)

        self.flatten = nn.Flatten()

        self.enc_varFC1 = nn.Linear(16*(configs.sequence_len//2//2+1+1), configs.sequence_len)
        self.enc_varFC2 = nn.Linear(configs.sequence_len, 16)

        self.decFC1 = nn.Linear(16, configs.sequence_len)
        self.decFC2 = nn.Linear(configs.sequence_len, 16*(configs.sequence_len//2//2+1+1))

        self.decConv1 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=(configs.kernel_size // 2), stride=2, bias=False,
                      padding=(configs.kernel_size // 4)),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.decConv2 = nn.Sequential(
            nn.ConvTranspose1d(8, configs.input_channels, kernel_size=8,
                      stride=2, bias=False, padding=4),

        )

        self.decConvF = nn.Sequential(
            nn.ConvTranspose1d(configs.input_channels, configs.input_channels, kernel_size=8,
                               stride=2, bias=False, padding=5),
        )

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.encConv1(x)
        x = self.encConv2(x)
        x = self.flatten(x)
        mu = F.relu(self.enc_muFC1(x))
        mu = self.enc_muFC2(mu)
        logVar = F.relu(self.enc_varFC1(x))
        logVar = self.enc_varFC2(logVar)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = F.relu(self.decFC2(x))
        x = x.view(-1, 16, (self.configs.sequence_len//2//2+1+1))
        x = self.decConv1(x)
        x = self.decConv2(x)
        x = torch.sigmoid(self.decConvF(x))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

    def loss(self, x, x_dec, mu, logvar):
        rec_loss = torch.sum(torch.pow(x.reshape(x.shape[0], -1) - x_dec.reshape(x.shape[0], -1), 2), 1).sqrt()
        #rec_loss = torch.sum(torch.pow(x.reshape(x.shape[0], -1) - x_dec.reshape(x.shape[0], -1), 2))
        rec_loss = rec_loss.mean()
        kl_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #torch.sum(0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1))
        return rec_loss, kl_loss

class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)
        self.configs = configs

    def forward(self, x):

        predictions = self.logits(x)

        return predictions



########## TCN #############################
torch.backends.cudnn.benchmark = True  # might be required to fasten TCN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    def __init__(self, configs):
        super(TCN, self).__init__()

        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)

        out = out_1[:, :, -1]
        return out


######## RESNET ##############################################

class RESNET18(nn.Module):
    def __init__(self, configs):
        super(RESNET18, self).__init__()
        self.resnet = resnet18(configs)
    def forward(self, x_in):
        x = self.resnet(x_in)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


##################################################
##########  OTHER NETWORKS  ######################
##################################################

class codats_classifier(nn.Module):
    def __init__(self, configs):
        super(codats_classifier, self).__init__()
        model_output_dim = configs.features_len
        self.hidden_dim = configs.hidden_dim
        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.final_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, configs.num_classes))

    def forward(self, x_in):
        predictions = self.logits(x_in)
        return predictions


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_CDAN(nn.Module):
    """Discriminator model for CDAN ."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels * configs.num_classes, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by AdvSKM ##############
class Cosine_act(nn.Module):
    def __init__(self):
        super(Cosine_act, self).__init__()

    def forward(self, input):
        return torch.cos(input)


cos_act = Cosine_act()

class AdvSKM_Disc(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(AdvSKM_Disc, self).__init__()

        self.input_dim = configs.features_len * configs.final_out_channels
        self.hid_dim = configs.DSKN_disc_hid
        self.branch_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            cos_act,
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.Linear(self.hid_dim // 2, self.hid_dim // 2),
            nn.BatchNorm1d(self.hid_dim // 2),
            cos_act
        )
        self.branch_2 = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.BatchNorm1d(configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim // 2),
            nn.Linear(configs.disc_hid_dim // 2, configs.disc_hid_dim // 2),
            nn.BatchNorm1d(configs.disc_hid_dim // 2),
            nn.ReLU())

    def forward(self, input):
        """Forward the discriminator."""
        out_cos = self.branch_1(input)
        out_rel = self.branch_2(input)
        total_out = torch.cat((out_cos, out_rel), dim=1)
        return total_out
    
  # SASA model
class CNN_ATTN(nn.Module):
    def __init__(self, configs):
        super(CNN_ATTN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)
        self.attn_network = attn_network(configs)
        self.sparse_max = Sparsemax(dim=-1)
        self.feat_len = configs.features_len

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        attentive_feat = self.calculate_attentive_feat(x_flat)
        return attentive_feat

    def self_attention(self, Q, K, scale=True, sparse=True, k=3):

        attention_weight = torch.bmm(Q.view(Q.shape[0], self.feat_len, -1), K.view(K.shape[0], -1, self.feat_len))

        attention_weight = torch.mean(attention_weight, dim=2, keepdim=True)

        if scale:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, self.feat_len]))
            attention_weight = torch.reshape(attention_weight_sparse, [-1, attention_weight.shape[1],
                                                                       attention_weight.shape[2]])
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def attention_fn(self, Q, K, scaled=False, sparse=True, k=1):

        attention_weight = torch.matmul(F.normalize(Q, p=2, dim=-1),
                                        F.normalize(K, p=2, dim=-1).view(K.shape[0], K.shape[1], -1, self.feat_len))

        if scaled:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
            attention_weight = k * torch.log(torch.tensor(self.feat_len, dtype=torch.float32)) * attention_weight

        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, self.feat_len]))

            attention_weight = torch.reshape(attention_weight_sparse, attention_weight.shape)
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def calculate_attentive_feat(self, candidate_representation_xi):
        Q_xi, K_xi, V_xi = self.attn_network(candidate_representation_xi)
        intra_attention_weight_xi = self.self_attention(Q=Q_xi, K=K_xi, sparse=True)
        Z_i = torch.bmm(intra_attention_weight_xi.view(intra_attention_weight_xi.shape[0], 1, -1),
                        V_xi.view(V_xi.shape[0], self.feat_len, -1))
        final_feature = F.normalize(Z_i, dim=-1).view(Z_i.shape[0],-1)

        return final_feature

class attn_network(nn.Module):
    def __init__(self, configs):
        super(attn_network, self).__init__()

        self.h_dim = configs.features_len * configs.final_out_channels
        self.self_attn_Q = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.ELU()
                                         )
        self.self_attn_K = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.LeakyReLU()
                                         )
        self.self_attn_V = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.LeakyReLU()
                                         )

    def forward(self, x):
        Q = self.self_attn_Q(x)
        K = self.self_attn_K(x)
        V = self.self_attn_V(x)

        return Q, K, V


    # Sparse max
class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1,
                                                                                                                     -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
