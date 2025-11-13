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

def visualize_at_index(mm, dm, in_transform, out_transform, variable, src, index=0):
    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    channel = dm.hparams.out_vars.index(variable)
    history = dm.hparams.history
    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"Daily Max Temperature (C)"
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
    if dm.hparams.task == "continuous-forecasting":
        xx = xx[:, :-1]

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
        if src == "era5":
            img = np.flip(img, 0)
        visualize_sample(img, extent, f"Input: {variable_with_units}")
        anim = None
        plt.show()

    # Plot the ground truth
    yy = out_transform(y[adj_index])
    yy = yy[channel].detach().cpu().numpy()
    if src == "era5":
        yy = np.flip(yy, 0)
    visualize_sample(yy, extent, f"Ground truth: {variable_with_units}")
    plt.show()

    # Plot the prediction
    ppred = out_transform(pred[adj_index])
    ppred = ppred[channel].detach().cpu().numpy()
    if src == "era5":
        ppred = np.flip(ppred, 0)
    visualize_sample(ppred, extent, f"Prediction: {variable_with_units}")
    plt.show()

    # Plot the bias
    bias = ppred - yy
    visualize_sample(bias, extent, f"Bias: {variable_with_units}")
    plt.show()

    # None, if no history
    return anim

def visualize_at_index_save(mm, dm, in_transform, out_transform, variable, src, out_path, index=0):
    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    channel = dm.hparams.out_vars.index(variable)
    history = dm.hparams.history
    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"Daily Max Temperature (C)"
    elif src == 'cerra':        
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
    if dm.hparams.task == "continuous-forecasting":
        xx = xx[:, :-1]

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
        visualize_sample_save(img, out_path, extent, index, f"Input: {variable_with_units}",'in')
        anim = None
        plt.show()

    # Plot the ground truth
    yy = out_transform(y[adj_index])
    yy = yy[channel].detach().cpu().numpy()
    if src == "era5" or 'cerra':
        yy = np.flip(yy, 0)
    visualize_sample_save(yy, out_path, extent, index, f"Ground truth: {variable_with_units}", 'gt')
    plt.show()

    # Plot the prediction
    ppred = out_transform(pred[adj_index])
    ppred = ppred[channel].detach().cpu().numpy()
    if src == "era5" or 'cerra':
        ppred = np.flip(ppred, 0)

    ## for saving the prediction
    out_npz = os.path.join(out_path, 'pred.npz')
    np.savez(out_npz, pred=ppred)
    ###

    visualize_sample_save(ppred, out_path, extent, index, f"Prediction: {variable_with_units}", 'pred')
    plt.show()

    # Plot the bias
    bias = ppred - yy
    visualize_sample_save(bias, out_path, extent, index, f"Bias: {variable_with_units}", 'bias')
    plt.show()

    # None, if no history
    return anim

def visualize_sphere_at_index_save(mm, dm, in_transform, out_transform, variable, src, out_path, index=0, is_global=True):
    lat, lon = dm.get_lat_lon()
    print(lat.shape)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    channel = dm.hparams.out_vars.index(variable)
    history = dm.hparams.history
    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"Daily Max Temperature (C)"
    elif src == 'cerra':        
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
    if dm.hparams.task == "continuous-forecasting":
        xx = xx[:, :-1]

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
        visualize_sample_save_sphere(img, out_path, extent, index, f"Input: {variable_with_units}",'in',is_global=is_global)
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
    else:
        visualize_sample_save_sphere(yy, out_path, extent, index, f"Ground truth: {variable_with_units}", 'gt',is_global=is_global)
    plt.show()

    # Plot the prediction
    ppred = out_transform(pred[adj_index])
    ppred = ppred[channel].detach().cpu().numpy()
    if src == "era5" or 'cerra':
        ppred = np.flip(ppred, 0)
    if src == "cerra":
        visualize_sample_save_sphere_cerra(ppred, out_path, land_mask, extent, index, f"Prediction: {variable_with_units}", 'pred')
    else:
        visualize_sample_save_sphere(ppred, out_path, extent, index, f"Prediction: {variable_with_units}", 'pred',is_global=is_global)
    plt.show()

    # Plot the bias
    bias = ppred - yy
    if src == "cerra":
        visualize_sample_save_sphere_cerra(bias, out_path, land_mask, extent, index, f"Bias: {variable_with_units}", 'bias')
    else:
        visualize_sample_save_sphere(bias, out_path, extent, index, f"Bias: {variable_with_units}", 'bias',is_global=is_global)
    plt.show()

    return anim


def visualize_sample(img, extent, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cmap = plt.cm.coolwarm
    cmap.set_bad("black", 1)
    ax.imshow(img, cmap=cmap, extent=extent)
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.02,
            ax.get_position().y0,
            0.02,
            ax.get_position().y1 - ax.get_position().y0,
        ]
    )
    fig.colorbar(ax.get_images()[0], cax=cax)
    return (fig, ax)

def visualize_sample_save(img, out_path, extent, index, title, content):
    filename = f"{str(index)}_{content}.png"

    # ---------- make sure target folder exists ----------
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename
    font_size = 16

    # ---------- create and save plot ----------
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("Longitude", fontsize=font_size)
    ax.set_ylabel("Latitude", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    cmap = plt.cm.coolwarm
    cmap.set_bad("black", 1)
    # ----- unified range for ERA-5 bias -----
    vmin = vmax = None
    if content == "bias":
        norm = Normalize(vmin=-5, vmax=5)
    else:
        norm = None

    im = ax.imshow(img, cmap=cmap, extent=extent, norm=norm)
    # add colour-bar
    # only for cerra! remember to comment for ERA5
    ax.set_aspect((extent[1] - extent[0]) / (extent[3] - extent[2]))
    cax = fig.add_axes([
        ax.get_position().x1 + 0.02,
        ax.get_position().y0,
        0.02,
        ax.get_position().y1 - ax.get_position().y0,
    ])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=font_size)
    # if norm is not None:
    #     cb.set_ticks([-10, -5, 0, 5, 10])
    fig.savefig(save_path, bbox_inches="tight", dpi=300)  # always PNG
    return save_path

def visualize_sample_save_sphere(img, out_path, extent, index, title, content, is_global):
    """
    img:      H x W numpy array (lat-lon grid)
    extent:   [lon_min, lon_max, lat_min, lat_max]  (global or regional)
    out_path: output directory
    """
    font_size=14
    if is_global:
        img, extent = roll_to_greenwich(img, [0, 360, -90, 90])

    filename = f"{str(index)}_{content}.png"
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename

    lon_min, lon_max, lat_min, lat_max = extent
    # Colormap + optional bias normalization
    cmap = plt.cm.coolwarm
    cmap.set_bad("black", 1)
    norm = Normalize(vmin=-5, vmax=5) if content == "bias" else None

    if is_global:
        # ---- GLOBAL VIEW (curved earth) ----
        # proj = ccrs.Robinson()                           # curved, visually nice world map
        proj = ccrs.PlateCarree()
        data_crs = ccrs.PlateCarree()                    # your data are in lon/lat
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=proj)
        ax.set_global()

        # close the seam at 180°E/W to avoid a white gap
        img_cyc, lons_cyc = add_cyclic_point(
            img, np.linspace(-180, 180, img.shape[1], endpoint=False)
        )

        im = ax.imshow(
            img_cyc,
            transform=data_crs,
            extent=(-180, 180, -90, 90),
            origin="upper",        # change to 'lower' if your first row is lat_min
            cmap=cmap, norm=norm, interpolation="nearest"
        )
        # Add map decorations (works for global & regional)
        ax.coastlines(linewidth=0.7)                         # coastlines
    else:
        # ---- REGIONAL VIEW ----
        proj = ccrs.PlateCarree()
        data_crs = ccrs.PlateCarree()
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=proj)
        ax.set_extent(extent, crs=data_crs)

        im = ax.imshow(
            img,
            transform=data_crs,
            extent=extent,
            origin="upper",        # change to 'lower' if your first row is lat_min
            cmap=cmap, norm=norm, interpolation="nearest"
        )

        ax.set_aspect((extent[1] - extent[0]) / (extent[3] - extent[2]))


    ax.set_title(title, pad=6)
    # gridlines (optional)
    # gl = ax.gridlines(draw_labels=False, linewidth=0.3)

    # Colorbar on the side
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=font_size)
    # If you want fixed ticks: cb.set_ticks([-5,-2.5,0,2.5,5])

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return save_path

def _load_mask_any(land_mask, index: int):
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

def visualize_sample_save_sphere_cerra(img, out_path, land_mask, extent, index, title, content):
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
    norm = Normalize(vmin=-5, vmax=5) if content == "bias" else Normalize(vmin=220, vmax=320)

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

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        img,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",              
        cmap="coolwarm",
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
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect((lon_max - lon_min) / (lat_max - lat_min))

    ax.set_title(title, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=font_size)
    fig.text(0.5, 0.01, content, ha="center", va="bottom", fontsize=font_size)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path



def roll_to_greenwich(img, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    W = img.shape[1]
    L0 = (lon_min % 360 + 360) % 360
    L1 = (lon_max % 360 + 360) % 360
    if L1 <= L0: L1 += 360
    left_target = -180.0
    shift_deg = (L0 - left_target) % 360
    shift_cols = int(round(shift_deg / 360 * W)) % W
    img2 = np.roll(img, -shift_cols, axis=1)
    new_extent = [-180, 180, lat_min, lat_max]
    return img2, new_extent


def visualize_mean_bias(dm, mm, out_transform, variable, src):
    lat, lon = dm.get_lat_lon()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    channel = dm.hparams.out_vars.index(variable)
    if src == "era5":
        variable_with_units = f"{variable} ({ERA5_VAR_TO_UNIT[variable]})"
    elif src == "cmip6":
        variable_with_units = f"{variable} ({CMIP6_VAR_TO_UNIT[variable]})"
    elif src == "prism":
        variable_with_units = f"Daily Max Temperature (C)"
    else:
        raise NotImplementedError(f"{src} is not a supported source")

    all_biases = []
    for batch in tqdm(dm.test_dataloader()):
        x, y = batch[:2]
        x = x.to(mm.device)
        y = y.to(mm.device)
        pred = mm.forward(x)
        pred = out_transform(pred)[:, channel].detach().cpu().numpy()
        obs = out_transform(y)[:, channel].detach().cpu().numpy()
        bias = pred - obs
        all_biases.append(bias)

    fig, ax = plt.subplots()
    all_biases = np.concatenate(all_biases)
    mean_bias = np.mean(all_biases, axis=0)
    if src == "era5":
        mean_bias = np.flip(mean_bias, 0)
    ax.imshow(mean_bias, cmap=plt.cm.coolwarm, extent=extent)
    ax.set_title(f"Mean Bias: {variable_with_units}")

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.02,
            ax.get_position().y0,
            0.02,
            ax.get_position().y1 - ax.get_position().y0,
        ]
    )
    fig.colorbar(ax.get_images()[0], cax=cax)
    plt.show()


# based on https://github.com/oliverangelil/rankhistogram/tree/master
def rank_histogram(obs, ensemble, channel):
    obs = obs.numpy()[:, channel]
    ensemble = ensemble.numpy()[:, :, channel]
    combined = np.vstack((obs[np.newaxis], ensemble))
    ranks = np.apply_along_axis(lambda x: rankdata(x, method="min"), 0, combined)
    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)
    for i in range(1, len(tie)):
        idx = ranks[ties == tie[i]]
        ranks[ties == tie[i]] = [
            np.random.randint(idx[j], idx[j] + tie[i] + 1, tie[i])[0]
            for j in range(len(idx))
        ]
    hist = np.histogram(
        ranks, bins=np.linspace(0.5, combined.shape[0] + 0.5, combined.shape[0] + 1)
    )
    plt.bar(range(1, ensemble.shape[0] + 2), hist[0])
    plt.show()
