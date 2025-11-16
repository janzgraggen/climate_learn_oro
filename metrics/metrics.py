# Standard Library
from typing import Callable, Optional, Union

# Local application
from .utils import MetricsMetaInfo, register
from .functional import *

# Third party
import numpy as np
import torch


class Metric:
    """Parent class for all ClimateLearn metrics."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
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


# -------------------------
oro_path = "dataset/CERRA-534/orography.npz"
oro = np.load(oro_path)["orography"]
oro = np.squeeze(oro).astype(np.float32)
H = torch.tensor(oro)  # shape (534, 534)

# -------------------------
# Differentiable helper functions
def softabs(x, eps=1e-6):
    return torch.sqrt(x**2 + eps)

def f(dH):
    """Linear expected topological change"""
    return 0.0005816 * dH + 0.01518

def weight(dH):
    """Sigmoid weight: 0 for small dH, ~1 at dH ~ 130"""
    return torch.sigmoid((dH - 130) / 20.0)  # adjust slope if needed

def empirical_oro_loss(T, H, use_weight=True):
    """
    T: predictions, shape (B, H, W)
    H: reference map, shape (H, W)
    """
    B, _, _ = T.shape

    # Make H batched
    H_batch = H.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # Horizontal and vertical differences
    dT_x = torch.abs(T[:, :, 1:] - T[:, :, :-1])
    dT_y = torch.abs(T[:, 1:, :] - T[:, :-1, :])
    dT = torch.cat([dT_x.reshape(B, -1), dT_y.reshape(B, -1)], dim=1)

    dH_x = torch.abs(H_batch[:, :, 1:] - H_batch[:, :, :-1])
    dH_y = torch.abs(H_batch[:, 1:, :] - H_batch[:, :-1, :])
    dH = torch.cat([dH_x.reshape(B, -1), dH_y.reshape(B, -1)], dim=1)

    deltaT_soft = softabs(dT)
    f_val = f(dH)

    w = weight(dH) if use_weight else 1.0
    print(deltaT_soft.shape, f_val.shape, w.shape)
    loss = w * (deltaT_soft - f_val) ** 2
    return loss.mean()

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
        mse_loss = mse(pred, target, self.aggregate_only)
        oro_loss = empirical_oro_loss(pred.squeeze(), H.to(device=pred.device))
        return mse_loss + 0.5 * oro_loss  # lambda_topo = 0.1