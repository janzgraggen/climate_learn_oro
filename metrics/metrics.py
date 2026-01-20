# Standard Library
from typing import Callable, Optional, Union, Iterable, Dict, Any, Literal

# Local application
from .utils import MetricsMetaInfo, register, register_lagg, LAGG_REGISTRY
from .functional import *

# Third party
import os
import numpy as np
import torch
from functools import partial

from ..models.hub.domain_mapping import dH_to_dT_conv, H_to_dT_conv_PE


class Metric:
    """Parent class for all ClimateLearn metrics."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None, loss_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        .. highlight:: python

        :param aggregate_only: If false, returns both the aggregate and
            per-channel metrics. Otherwise, returns only the aggregate metric.
            Default is `False`.
        :type aggregate_only: bool
        :param metainfo: Optional meta-information used by some metrics.
        :type metainfo: MetricsMetaInfo|None
        """
        self.aggregate_only = aggregate_only
        self.metainfo = metainfo
        self.loss_kwargs = loss_kwargs

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param pred: The predicted value(s).
        :type pred: torch.Tensor
        :param target: The ground truth target value(s).
        :type target: torch.Tensor

        :return: A tensor. See child classes for specifics.
        :rtype: torch.Tensor
        """
        raise NotImplementedError()


class LatitudeWeightedMetric(Metric):
    """Parent class for latitude-weighted metrics."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
    ):
        super().__init__(aggregate_only, metainfo)
        lat_weights = np.cos(np.deg2rad(self.metainfo.lat))
        lat_weights = lat_weights / lat_weights.mean()
        lat_weights = torch.from_numpy(lat_weights).view(1, 1, -1, 1)
        self.lat_weights = lat_weights

    def cast_to_device(
        self, pred: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> None:
        r"""
        .. highlight:: python

        Casts latitude weights to the same device as `pred`.
        """
        self.lat_weights = self.lat_weights.to(device=pred.device)


class ClimatologyBasedMetric(Metric):
    """Parent class for metrics that use climatology."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
    ):
        super().__init__(aggregate_only, metainfo)
        climatology = self.metainfo.climatology
        climatology = climatology.unsqueeze(0)
        self.climatology = climatology

    def cast_to_device(
        self, pred: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> None:
        r"""
        .. highlight:: python

        Casts climatology to the same device as `pred`.
        """
        self.climatology = self.climatology.to(device=pred.device)


class TransformedMetric:
    """Class which composes a transform and a metric."""

    def __init__(self, transform: Callable, metric: Metric):
        self.transform = transform
        self.metric = metric
        self.name = metric.name

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> None:
        pred = self.transform(pred)
        target = self.transform(target)
        return self.metric(pred, target)


@register("mse")
class MSE(Metric):
    """Computes standard mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return mse(pred, target, self.aggregate_only)


@register("lat_mse")
class LatWeightedMSE(LatitudeWeightedMetric):
    """Computes latitude-weighted mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        return mse(pred, target, self.aggregate_only, self.lat_weights)


@register("rmse")
class RMSE(Metric):
    """Computes standard root mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            RMSE, and the preceding elements are the channel-wise RMSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        if mask is not None:
            return rmse(pred, target, self.aggregate_only, mask)
        return rmse(pred, target, self.aggregate_only)


@register("lat_rmse")
class LatWeightedRMSE(LatitudeWeightedMetric):
    """Computes latitude-weighted root mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            RMSE, and the preceding elements are the channel-wise RMSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        if mask is not None:
            return rmse(pred, target, self.aggregate_only, self.lat_weights, mask)
        return rmse(pred, target, self.aggregate_only, self.lat_weights)


@register("acc")
class ACC(ClimatologyBasedMetric):
    """
    Computes standard anomaly correlation coefficient.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            ACC, and the preceding elements are the channel-wise ACCs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        if mask is not None:
            return acc(pred, target, self.climatology, self.aggregate_only, mask)
        return acc(pred, target, self.climatology, self.aggregate_only)


@register("lat_acc")
class LatWeightedACC(LatitudeWeightedMetric, ClimatologyBasedMetric):
    """
    Computes latitude-weighted anomaly correlation coefficient.
    """

    def __init__(self, *args, **kwargs):
        LatitudeWeightedMetric.__init__(self, *args, **kwargs)
        ClimatologyBasedMetric.__init__(self, *args, **kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            ACC, and the preceding elements are the channel-wise ACCs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        LatitudeWeightedMetric.cast_to_device(self, pred)
        ClimatologyBasedMetric.cast_to_device(self, pred)
        if mask is not None:
            return acc(
                pred,
                target,
                self.climatology,
                self.aggregate_only,
                self.lat_weights,
                mask,
            )
        return acc(
            pred, target, self.climatology, self.aggregate_only, self.lat_weights
        )


@register("pearson")
class Pearson(Metric):
    """
    Computes the Pearson correlation coefficient, based on
    https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/10
    """

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            Pearson correlation coefficient, and the preceding elements are the
            channel-wise Pearson correlation coefficients.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return pearson(pred, target, self.aggregate_only)


@register("mean_bias")
class MeanBias(Metric):
    """Computes the standard mean bias."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate mean
            bias, and the preceding elements are the channel-wise mean bias.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return mean_bias(pred, target, self.aggregate_only)


## LOAD H (Static) –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
oro_path = "dataset/CERRA-534/orography.npz"
oro = np.load(oro_path)["orography"]
oro = np.squeeze(oro).astype(np.float32)
H = torch.tensor(oro).to("cuda")  # shape (1,534, 534)

class DH_TO_DT_CONV_LOADER: 
    def __init__(self, model_spec: Literal["conv","conv_PE"]):
        self.spec = model_spec
        self.path = "climate_learn_oro/metrics/eecr_models/dH_to_dT_"+model_spec+".pt" #or location of cl (climate_learn_oro) package
        self.model = False

    def _load_model(self):
        ## Model 
        if self.spec == "conv":
            self.model = dH_to_dT_conv()
        elif self.spec == "conv_PE":
            self.model = H_to_dT_conv_PE()
        else:
            raise ValueError(f"Unknown model spec: {self.spec}")
        self.model.cuda()
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Model weights not found at {self.path}. Please train the model first.")
        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[LAGG]: Total number of parameters: {param_count / 1e3:.1f}K")

    def __call__(self, dH):
        if not self.model:
            self._load_model()
        return self.model(dH)

DH_TO_DT_CONV = {
    "conv": DH_TO_DT_CONV_LOADER("conv"),
    "conv_PE": DH_TO_DT_CONV_LOADER("conv_PE")
}

def DM_empirical_linear(dH):
    """Linear expected topological change"""
    return 0.0005816 * dH + 0.01518

def DM_learned_conv(dH, model_spec: Literal["conv", "conv_PE"]="conv"):
    return(DH_TO_DT_CONV[model_spec](dH))

def SO_abs_horizontal_vertical_differences(M, output: Literal["tuple", "flat", "concat"]= "tuple",soft = False):
    """Compute horizontal and vertical absolute differences of map M.
    input:
        M: shape (B, H, W)
    Returns: 
     if flat: 
        dM of shape (B, 2*H*W -H -W) i.e. 
     if concat:
        dM of shape (B, 2, H, W)
     if tuple:
        dM_x of shape (B, H, W-1), dM_y of shape (B, H-1, W)
    """

    def softabs(x, eps=1e-6):
        return torch.sqrt(x**2 + eps)
    B, H, W = M.shape
    dM_x = (M[:, :, 1:] - M[:, :, :-1])  # shape (B, H, W-1)
    dM_y = (M[:, 1:, :] - M[:, :-1, :])  # shape (B, H-1, W)
    if output == "flat":
        flat = torch.cat([dM_x.reshape(B, -1), dM_y.reshape(B, -1)], dim=1) # shape (B, 2*H*W - H - W)
        print(flat.shape)
        if soft:
            return softabs(flat).cuda()
        else:
            return torch.abs(flat).cuda()
    if output == "concat":
        dMx_pad = torch.nn.functional.pad(dM_x, (1,0), mode='replicate') #(B,H,W-1+1)
        dMy_pad = torch.nn.functional.pad(dM_y.unsqueeze(0), (0, 0, 1, 0), mode='replicate').squeeze(0) # (B,H-1+1,W)
        pad_concat = torch.stack([dMx_pad, dMy_pad], dim=1).float() # (B,2,H,W)
        if soft:
            return softabs(pad_concat).cuda()
        else:
            return torch.abs(pad_concat).cuda()
    if output == "tuple":
        if soft:
            return softabs(dM_x).cuda(), softabs(dM_y).cuda()
        else:
            return torch.abs(dM_x).cuda(), torch.abs(dM_y).cuda()

@register_lagg("empirical_linear")
def LAGG_empirical_linear(T):
    """
    T: predictions, shape (B, H, W)

    calculates the LAGG using 
        SIDM: empirical_linear
        DISA: abs horizontal and vertical differences, flat: (B,H,W)-> (B,2*H*W)
        weighting: sigmoid weight based on dH (sigmoid to adapt to linear regime of the empirical relation)
    """

    def _w(dH):
        """Sigmoid weight: 0 for small dH, ~1 at dH ~ 130"""
        return torch.sigmoid((dH - 130) / 20.0)  # adjust slope if needed

    B, _, _ = T.shape

    # Make H batched
    H_batch = H.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # Horizontal and vertical differences
    dT = SO_abs_horizontal_vertical_differences(T, output="flat",soft=True).cuda()
    dH = SO_abs_horizontal_vertical_differences(H_batch, output="flat",soft=False).cuda()

    return mse(dT, DM_empirical_linear(dH), lat_weights=_w(dH),aggregate_only=True) 

def LAGG_conv(T, model_spec: Literal["conv", "conv_PE"]="conv",metric_gate_weights: Optional[torch.nn.Parameter]=None): 
    """
    T: predictions, shape (B, H, W)

    calculates the LAGG using 
        SIDM: dH_to_dT_conv
        DISA: abs horizontal and vertical differences, concatenated: (B,H,W)-> (B,2,H,W)
        weighting: none 
    """
    # compute dH
    B, _, _ = T.shape
    H_batch = H.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    
    if model_spec == "conv":
        dH_stack = SO_abs_horizontal_vertical_differences(H_batch, output="concat",soft=False).cuda()
    elif model_spec == "conv_PE": # H derrivatives naturally computed in the GeoINR positional encoding)
        dH_stack = H_batch.unsqueeze(1).float().cuda()  # shape (B,1,H,W)
    dT_stack = SO_abs_horizontal_vertical_differences(T, output="concat",soft=True).cuda()
    return  mse(dT_stack, DM_learned_conv(dH_stack, model_spec=model_spec), aggregate_only=True,lat_weights=metric_gate_weights)


for name in DH_TO_DT_CONV.keys():
    register_lagg(name)(partial(LAGG_conv, model_spec=name))

@register("mse_oro")
class MSE_ORO(Metric):
    """Computes standard mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        if self.loss_kwargs.get("lambda_oro") is None:
            raise ValueError("lambda_oro must be provided for MSE_ORO metric.\n give lambda_oro as loss_kwargs parameter when loading the loss.")
        else :
            lambda_oro = self.loss_kwargs["lambda_oro"]

        if self.loss_kwargs.get("lagg") is None or self.loss_kwargs["lagg"] not in LAGG_REGISTRY.keys():
            raise ValueError(f"lagg must be (correctly) provided for MSE_ORO metric. Choose from {list(LAGG_REGISTRY.keys())}.\n info: Give lagg as loss_kwargs parameter when loading the loss.")
        else: 
            LAGG_fn = LAGG_REGISTRY[self.loss_kwargs["lagg"]]
        mse_loss = mse(pred, target, self.aggregate_only)
        oro_loss = LAGG_fn(pred.squeeze())
        return mse_loss + lambda_oro * oro_loss  # lambda_oro = 0.1s
