import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm
from ..data.processing.era5_constants import VAR_TO_UNIT as ERA5_VAR_TO_UNIT
from ..data.processing.cmip6_constants import VAR_TO_UNIT as CMIP6_VAR_TO_UNIT
from pathlib import Path
from matplotlib.colors import Normalize, LogNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import os
from matplotlib import colors
from scipy.interpolate import griddata
import glob
import torch
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def visualize_sphere_at_index_save(mm, dm, in_transform, out_transform, variable, src, out_path, index=0, is_global=True):
    print("@ visualize_sphere_at_index_save")

    lat, lon = dm.get_lat_lon() 

    print(lat.shape)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    channel = dm.hparams.out_vars.index(variable)
    history = dm.hparams.history
    # if src == "era5":
    #     variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    # elif src == "cmip6":
    #     variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    # elif src == "prism":
    #     variable_with_units = f"Daily Max Temperature (C)"
    if src == 'cerra':   # 'ALLWAYS cerra'     
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
        extent = [-58.0, 74.0, lat.min(), lat.max()]
    else:
        raise NotImplementedError(f"{src} is not a supported source")

    counter = 0
    adj_index = None
    for batch in tqdm(dm.test_dataloader()):
        x, y = batch[:2]
        batch_size = x.shape[0]
        if index in range(counter, counter + batch_size):
            adj_index = index - counter
            x = x.to(mm.device)
            pred = mm.forward(x)
            break
        counter += batch_size

    if adj_index is None:
        raise RuntimeError("Given index could not be found")
    xx = x[adj_index]
    #always direct
    # if dm.hparams.task == "continuous-forecasting":
    #     xx = xx[:, :-1]

    # Create animation/plot of the input sequence
    if history > 1:
        in_fig, in_ax = plt.subplots()
        in_ax.set_title(f"Input Sequence: {variable_with_units}")
        in_ax.set_xlabel("Longitude")
        in_ax.set_ylabel("Latitude")
        imgs = []
        for time_step in range(history):
            img = in_transform(xx[time_step])[channel].detach().cpu().numpy()
            if src == "era5":
                img = np.flip(img, 0)
            img = in_ax.imshow(img, cmap=plt.cm.coolwarm, animated=True, extent=extent)
            imgs.append([img])
        cax = in_fig.add_axes(
            [
                in_ax.get_position().x1 + 0.02,
                in_ax.get_position().y0,
                0.02,
                in_ax.get_position().y1 - in_ax.get_position().y0,
            ]
        )
        in_fig.colorbar(in_ax.get_images()[0], cax=cax)
        anim = animation.ArtistAnimation(in_fig, imgs, interval=1000, repeat_delay=2000)
        plt.close()
    else:
        if dm.hparams.task == "downscaling":
            img = in_transform(xx)[channel].detach().cpu().numpy()
        else:
            img = in_transform(xx[0])[channel].detach().cpu().numpy()
        if src == "era5" or "cerra":
            img = np.flip(img, 0)
        if src == "cerra":
            visualize_sample_save_sphere_cerra(img, out_path, land_mask=None, extent=extent, index=index, title=f"Input: {variable_with_units}", content='in')
        else: #Allways cerra
            raise NotImplementedError(f"{src} is not supported for input visualization.")
            # visualize_sample_save_sphere(img, out_path, extent, index, f"Input: {variable_with_units}",'in',is_global=is_global)
        anim = None
        plt.show()

    # Plot the ground truth
    yy = out_transform(y[adj_index])
    yy = yy[channel].detach().cpu().numpy()
    if src == "era5" or 'cerra':
        yy = np.flip(yy, 0)

    land_mask = np.load('/home/janz/dataset/CERRA-534/landsea.npz')['lsm']

    if src == "cerra":
        visualize_sample_save_sphere_cerra(yy, out_path, land_mask, extent, index, f"Ground truth: {variable_with_units}", 'gt')
    else: #Allways cerra
        raise NotImplementedError(f"{src} is not supported for ground truth visualization.")
        #visualize_sample_save_sphere(yy, out_path, extent, index, f"Ground truth: {variable_with_units}", 'gt',is_global=is_global)
    plt.show()

    # Plot the prediction
    ppred = out_transform(pred[adj_index])
    ppred = ppred[channel].detach().cpu().numpy()
    if src == "era5" or 'cerra':
        ppred = np.flip(ppred, 0)
    if src == "cerra":
        visualize_sample_save_sphere_cerra(ppred, out_path, land_mask, extent, index, f"Prediction: {variable_with_units}", 'pred')
    else: #Allways cerra
        raise NotImplementedError(f"{src} is not supported for prediction visualization.")
        #visualize_sample_save_sphere(ppred, out_path, extent, index, f"Prediction: {variable_with_units}", 'pred',is_global=is_global)
    plt.show()

    # Plot the bias
    bias = ppred - yy
    if src == "cerra":
        visualize_sample_save_sphere_cerra(bias, out_path, land_mask, extent, index, f"Bias: {variable_with_units}", 'bias')
    else: #Allways cerra
        raise NotImplementedError(f"{src} is not supported for bias visualization.")
        #visualize_sample_save_sphere(bias, out_path, extent, index, f"Bias: {variable_with_units}", 'bias',is_global=is_global)
    plt.show()

    return anim

def _load_mask_any(land_mask, index: int):
    print("@ _load_mask_any")
    """
    """
    if isinstance(land_mask, np.ndarray):
        mask = land_mask
    else:
        if os.path.isdir(land_mask):
            files = sorted(glob.glob(os.path.join(land_mask, "*.npz")))
            if not files:
                raise FileNotFoundError(f"No .npz found in dir: {land_mask}")
            fp = files[index % len(files)]
        else:
            fp = land_mask
        with np.load(fp) as npz:
            key = list(npz.keys())[0]
            mask = npz[key]
    # squeeze 到 2D
    mask = np.asarray(mask)
    if mask.ndim == 3 and 1 in mask.shape:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Loaded mask must be 2D, got shape {mask.shape}")
    return mask

def visualize_sample_save_sphere_cerra(img, out_path, land_mask, extent, index, title, content,abs_bias=False):
    print("@ visualize_sample_save_sphere_cerra")
    """
    img:    2D 数组（与 land-sea mask 对齐）
    out_path: 保存图片的完整路径（含文件名 .png/.jpg）
    land_mask: 掩码数组 或 .npz 文件路径 或 含 .npz 的目录
    extent: [lon_min, lon_max, lat_min, lat_max]  与 img/mask 对齐
    index:  当 land_mask 是目录时，用于选择第 index 个 .npz
    title:  图标题
    content: 底部说明文字
    """
    font_size=12
    filename = f"{str(index)}_{content}.png"
    if content == "bias":
        if abs_bias:
            norm = Normalize(vmin=0, vmax=4)
        else:
            norm = Normalize(vmin=-4, vmax=4)
    else:
        norm = Normalize(vmin=220, vmax=320)

    # ---------- make sure target folder exists ----------
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename

    # get land-sea mask
    mask = _load_mask_any(land_mask, index)
    if mask.shape != img.shape:
        raise ValueError(f"mask shape {mask.shape} != img shape {img.shape}")

    lon_min, lon_max, lat_min, lat_max = extent
    x = np.linspace(lon_min, lon_max, mask.shape[1])
    y = np.linspace(lat_min, lat_max, mask.shape[0])
    cmap = "coolwarm" if not abs_bias else "plasma_r"
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        img,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",              
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    
    # coastlines
    cs = ax.contour(
        x, y, mask,
        levels=[0.5],
        colors="k",
        linewidths=0.6
    )

    #################################
    ##############################
    ###############################
    # Add zoom inset if coordinates are provided–––––––––
    zoom_coords=[172, 202, 41, 46]  # x1, x2, y1, y2
    zoom_size=(2.5, 2.5)
    if zoom_coords is not None:
        x1, x2, y1, y2 = zoom_coords
        axins = inset_axes(ax, width=zoom_size[0], height=zoom_size[1], loc='upper right')
        axins.imshow(img, extent=extent, origin="upper",
                     cmap=cmap, norm=norm, interpolation="nearest", aspect='auto')
        axins.contour(
            np.linspace(extent[0], extent[1], mask.shape[1]),
            np.linspace(extent[2], extent[3], mask.shape[0]),
            mask,
            levels=[0.5],
            colors="k",
            linewidths=1
        )

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        # Draw rectangle + lines connecting to inset
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="whitesmoke", lw=0.7)
    # Add zoom inset if coordinates are provided–––––––––

    #############################
    ###############################
    #######################################


    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect((lon_max - lon_min) / (lat_max - lat_min))

    ax.set_title(title, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, pad=0.02,label="Abs Bias (K)" if abs_bias else "Temperature (K)")
    cbar.ax.tick_params(labelsize=font_size)
    #fig.text(0.5, 0.01, content, ha="center", va="bottom", fontsize=font_size)

    

    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")
    return save_path

@torch.no_grad()
def compute_pixelwise_mean_abs_bias(
    mm,
    dm,
    out_transform,
    variable,
    indices=None,
    src="cerra",
):
    mm.eval()
    device = mm.device
    channel = dm.hparams.out_vars.index(variable)

    bias_sum = None
    count = 0
    counter = 0

    for batch in tqdm(dm.test_dataloader(), desc="Pixelwise mean abs bias"):
        x, y = batch[:2]
        x = x.to(device)
        y = y.to(device)

        pred = mm(x)
        batch_size = x.shape[0]

        for bi in range(batch_size):
            gi = counter + bi
            if indices is not None and gi not in indices:
                continue

            yy = out_transform(y[bi])[channel]
            pp = out_transform(pred[bi])[channel]

            if src in ("era5", "cerra"):
                yy = torch.flip(yy, dims=[0])
                pp = torch.flip(pp, dims=[0])

            abs_bias = torch.abs(pp - yy)

            if bias_sum is None:
                bias_sum = abs_bias.clone()
            else:
                bias_sum += abs_bias

            count += 1

        counter += batch_size

    if count == 0:
        raise RuntimeError("No samples processed")

    return (bias_sum / count).cpu().numpy()


def visualize_pixelwise_mean_abs_bias(
    mm,
    dm,
    land_mask,
    out_transform,
    variable,
    out_path,
    indices=None,
    src="cerra",
    save_data_path=None,
):
    mean_abs_bias = compute_pixelwise_mean_abs_bias(
        mm=mm,
        dm=dm,
        out_transform=out_transform,
        variable=variable,
        indices=indices,
        src=src,
    )

    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    
    visualize_sample_save_sphere_cerra(
        img=mean_abs_bias,
        out_path=out_path,
        land_mask=land_mask,
        extent=extent,
        index=0,
        title=f"Pixelwise Mean Absolute Bias: {variable}",
        content="bias",   # reuse bias colormap
        abs_bias=True,
    )
    if save_data_path is not None:
        np.savez(
            save_data_path,
            mean_abs_bias=mean_abs_bias,
            lat=lat,
            lon=lon,
            variable=variable,
        )