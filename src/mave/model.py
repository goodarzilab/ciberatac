import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from typing import Dict, Tuple
from typing import Iterable, Optional
from utils import Encoder, FCLayers
from utils import NegativeBinomial, ZeroInflatedNegativeBinomial


try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type("Literal_", (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass


class DecoderSCVI(nn.Module):
    """
    Decodes data from latent space of ``n_input``
    dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer,
        or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.n_cat_list = n_cat_list

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, cat_list
    ):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value
         for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is
            constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ
            between different batches
            * ``'gene-label'`` - dispersion can differ
            between different labels
            * ``'gene-cell'`` - dispersion can differ
            for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters
        # of the ZINB distribution
        px = self.px_decoder(z, [0])
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to
        # avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout, px


def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def identity(x):
    return x


def loss_function(
        qz_m, qz_v, x, px_rate, px_r, px_dropout,
        ql_m, ql_v, use_observed_lib_size,
        local_l_mean, local_l_var):
    mean = torch.zeros_like(qz_m)
    scale = torch.ones_like(qz_v)
    kl_divergence_z = kl(
        Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1)
    if not use_observed_lib_size:
        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)
    else:
        kl_divergence_l = 0.0
    kl_divergence = kl_divergence_z
    reconst_loss = get_reconstruction_loss(
        x, px_rate, px_r, px_dropout)
    return reconst_loss + kl_divergence_l, kl_divergence


def get_reconstruction_loss(
        x, px_rate, px_r, px_dropout, gene_likelihood="zinb"):
    # Reconstruction Loss
    if gene_likelihood == "zinb":
        reconst_loss = (
            -ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            )
            .log_prob(x)
            .sum(dim=-1)
        )
    elif gene_likelihood == "nb":
        reconst_loss = (
            -NegativeBinomial(
                mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
        )
    elif gene_likelihood == "poisson":
        reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
    return reconst_loss


class VAE(nn.Module):
    def __init__(
        self,
        n_input: int,
        connections=None,
        n_celltypes: int = 10,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_observed_lib_size: bool = True,
    ):
        super().__init__()
        self.n_celltypes = n_celltypes
        self.connections = connections
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or\
            use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or\
            use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or\
            use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or\
            use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list(
            [] if n_cats_per_cov is None else n_cats_per_cov)
        self.cat_list = cat_list
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_cat_list = encoder_cat_list
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            self.connections,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        self.ctpred_linear = nn.Linear(n_latent, n_celltypes)
        self.ctpred_activation = nn.ReLU()

        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            None,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def get_latents(self, x, y=None) -> torch.Tensor:
        """
        Returns the result of ``sample_from_posterior_z`` inside a list.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, inputsize)``
        y
            tensor of cell-types labels with shape
            ``(batch_size, n_labels)`` (Default value = None)
        Returns
        -------
        type
            one element list of tensor
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(
        self, x, batch_index=None, y=None, give_mean=False, n_samples=5000
    ) -> torch.Tensor:
        """
        Samples the tensor of latent values from the posterior.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, inputsize)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
            (Default value = None)
        give_mean
            is True when we want the mean of the posterior
            distribution rather than sampling (Default value = False)
        n_samples
            how many MC samples to average over for
            transformed mean (Default value = 5000)
        Returns
        -------
        type
            tensor of shape ``(batch_size, lvsize)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(
            x)  # y only used in VAEC

        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.z_encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return z

    def sample_from_posterior_l(
        self, x, batch_index=None, give_mean=True
    ) -> torch.Tensor:
        """
        Samples the tensor of library sizes from the posterior.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, inputsize)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        give_mean
            Return mean or sample
        Returns
        -------
        type
            tensor of shape ``(batch_size, 1)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        if give_mean is False:
            library = library
        else:
            library = torch.distributions.LogNormal(ql_m, ql_v.sqrt()).mean
        return library

    def get_sample_scale(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> torch.Tensor:
        """
        Returns the tensor of predicted frequencies of expression.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, inputsize)``
        batch_index
            array that indicates which batch the cells
            belong to with shape ``batch_size`` (Default value = None)
        y
            tensor of cell-types labels with shape ``(batch_size,
            n_labels)`` (Default value = None)
        n_samples
            number of samples (Default value = 1)
        transform_batch
            int of batch to transform samples into (Default value = None)
        Returns
        -------
        type
            tensor of predicted frequencies of expression
            with shape ``(batch_size, inputsize)``
        """
        return self.inference(
            x,
            batch_index=batch_index,
            y=y,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_scale"]

    def get_sample_rate(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> torch.Tensor:
        """
        Returns the tensor of means of the negative binomial distribution.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, inputsize)``
        y
            tensor of cell-types labels with shape
            ``(batch_size, n_labels)`` (Default value = None)
        batch_index
            array that indicates which batch the cells belong to with
            shape ``batch_size`` (Default value = None)
        n_samples
            number of samples (Default value = 1)
        transform_batch
            int of batch to transform samples into (Default value = None)
        Returns
        -------
        type
            tensor of means of the negative binomial distribution with
            shape ``(batch_size, inputsize)``
        """
        return self.inference(
            x,
            batch_index=batch_index,
            y=y,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_rate"]

    def inference(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library_encoded = self.l_encoder(x_)
        if not self.use_observed_lib_size:
            library = library_encoded

        # Predict celltypes using z
        ctpred = self.ctpred_activation(
            self.ctpred_linear(qz_m))

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(
                0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(
                0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            ql_m = ql_m.unsqueeze(
                0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(
                0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = Normal(ql_m, ql_v.sqrt()).sample()

        if transform_batch is not None:
            dec_batch_index = transform_batch * torch.ones_like(batch_index)
        else:
            dec_batch_index = batch_index

        px_scale, px_r, px_rate, px_dropout, px = self.decoder(
            self.dispersion, z, library, [0]
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(dec_batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
            px=px,
            ctpred=ctpred
        )

    def forward(
        self, x, batch_index=None, y=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Parameters for z latent distribution
        outputs = self.inference(x, batch_index, y)
        return outputs


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    connections = torch.ones(950, 15000).long().to(device)
    vae = VAE(15000, connections,
              0, 0, 950, 10).to(device)
    train1 = torch.rand(10, 15000).to(device)
    outdict = vae(train1)
