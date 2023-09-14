import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from matplotlib.patches import Ellipse
from skimage.feature import peak_local_max


class AverageMeter(object):
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x / y if x > 0 else 0 for x, y in zip(self.sum, self.count)
            ]
            self.avg = sum(self.avg_per_class) / len(self.avg_per_class)


class Visualizer:
    def __init__(self, keys, cmap):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 2, figsize=(10, 10))

        self.ax1[1].axis("off")
        self.ax2[0].axis("off")
        self.ax2[1].axis("off")
        self.cmap = cmap

    def display(
        self,
        image,
        key,
        title,
        center_x=None,
        center_y=None,
        samples_x=None,
        samples_y=None,
        sample_spatial_embedding_x=None,
        sample_spatial_embedding_y=None,
        sigma_x=None,
        sigma_y=None,
        color_sample=None,
        color_embedding=None,
    ):
        len(image) if isinstance(image, (list, tuple)) else 1

        self.fig.frameon = False
        if key == "image":
            self.ax1[0].cla()
            self.ax1[0].imshow(self.prepare_img(image), cmap="magma")
            self.ax1[0].axis("off")
        elif key == "groundtruth":
            self.ax1[1].cla()
            self.ax1[1].imshow(
                self.prepare_img(image), cmap=self.cmap, interpolation="None"
            )
            self.ax1[1].axis("off")
        elif key == "prediction":
            self.ax2[1].cla()
            self.ax2[1].imshow(
                self.prepare_img(image), cmap=self.cmap, interpolation="None"
            )
            self.ax2[1].axis("off")
        elif key == "center":
            self.ax2[0].cla()
            self.ax2[0].imshow(self.prepare_img(image), cmap="gray")
            for i in range(len(color_sample.items())):
                self.ax2[0].plot(
                    center_x[i + 1],
                    center_y[i + 1],
                    color=color_embedding[i + 1],
                    marker="x",
                )
                self.ax2[0].scatter(
                    samples_x[i + 1],
                    samples_y[i + 1],
                    color=color_sample[i + 1],
                    marker="+",
                )
                self.ax2[0].scatter(
                    sample_spatial_embedding_x[i + 1],
                    sample_spatial_embedding_y[i + 1],
                    color=color_embedding[i + 1],
                    marker=".",
                )
                ellipse = Ellipse(
                    (center_x[i + 1], center_y[i + 1]),
                    width=sigma_x[i + 1],
                    height=sigma_y[i + 1],
                    angle=0,
                    color=color_embedding[i + 1],
                    alpha=0.5,
                )
                self.ax2[0].add_artist(ellipse)
            self.ax2[0].axis("off")
        plt.tight_layout()
        plt.draw()
        self.mypause(0.0001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams["backend"]
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


class Cluster_3d:
    """
    A class used to cluster pixel embeddings in 3D

    Attributes
    ----------
    xyzm :  float (3, W, D, H)
            pixel coordinates of tile /grid

    one_hot : bool
            Should be set to True, if the GT label masks are present
            in a one-hot encoded fashion
            Not applicable to 3D. This parameter will be deprecated
            in a future release

    grid_x: int
            Length (width) of grid / tile

    grid_y: int
            Height of grid / tile

    grid_z: int
            Depth of grid / tile

    pixel_x: float
            if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing
            along the x direction is pixel_x/(grid_x-1) = 1/999
    pixel_y: float
            if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing
            along the y direction is pixel_y/(grid_y-1) = 1/999
    pixel_z: float
            if grid_z = 1000 and pixel_z = 1, then the pixel spacing
            along the z direction is pixel_z/(grid_z-1) = 1/999


    Methods
    -------
    __init__: Initializes an object of class `Cluster_3d`

    cluster_with_gt: use the predicted spatial embeddings
            from all pixels belonging to the GT label mask
            to identify the predicted cluster
            (used during training and validation)

    cluster: use the  predicted spatial embeddings
             from all pixels in the test image.
             Employs `fg_thresh` and `seed_thresh`
    cluster_local_maxima: use the  predicted spatial embeddings
            from all pixels in the test image.
            Employs only `fg_thresh`
    """

    def __init__(
        self, grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, device, one_hot=False
    ):
        """
        Parameters
        ----------
        xyzm :  float (3, W, D, H)
                 pixel coordinates of tile /grid

         one_hot : bool
                 Should be set to True, if the GT label masks are present
                 in a one-hot encoded fashion
                 Not applicable to 3D. This parameter will be deprecated
                 in a future release

         grid_x: int
                 Length (width) of grid / tile

         grid_y: int
                 Height of grid / tile

         grid_z: int
                 Depth of grid / tile

         pixel_x: float
                 if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing
                 along the x direction is pixel_x/(grid_x-1) = 1/999
         pixel_y: float
                 if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing
                 along the y direction is pixel_y/(grid_y-1) = 1/999
         pixel_z: float
                 if grid_z = 1000 and pixel_z = 1, then the pixel spacing
                 along the z direction is pixel_z/(grid_z-1) = 1/999

        """

        xm = (
            torch.linspace(0, pixel_x, grid_x)
            .view(1, 1, 1, -1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        ym = (
            torch.linspace(0, pixel_y, grid_y)
            .view(1, 1, -1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        zm = (
            torch.linspace(0, pixel_z, grid_z)
            .view(1, -1, 1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        xyzm = torch.cat((xm, ym, zm), 0)

        self.xyzm = xyzm.to(device)
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.pixel_z = pixel_z
        self.device = device

    def cluster_with_gt(
        self,
        prediction,
        gt_instance,
        n_sigma=3,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (7, D, H, W)
        gt_instance : PyTorch Tensor
                Ground Truth Instance Segmentation Label Map

        n_sigma: int, default = 3
                Number of dimensions in Raw Image
        Returns
        ----------
        instance: PyTorch Tensor (D, H, W)
                instance segmentation
        """

        depth, height, width = (
            prediction.size(1),
            prediction.size(2),
            prediction.size(3),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]  # 3 x d x h x w
        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w
        sigma = prediction[3 : 3 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(depth, height, width).short().to(self.device)
        unique_instances = gt_instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = gt_instance.eq(id).view(1, depth, height, width)
            center = (
                spatial_emb[mask.expand_as(spatial_emb)]
                .view(3, -1)
                .mean(1)
                .view(3, 1, 1, 1)
            )  # 3 x 1 x 1 x 1
            s = (
                sigma[mask.expand_as(sigma)]
                .view(n_sigma, -1)
                .mean(1)
                .view(n_sigma, 1, 1, 1)
            )

            s = torch.exp(s * 10)  # n_sigma x 1 x 1
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = dist > 0.5
            instance_map[proposal] = id.item()  # TODO

        return instance_map

    def cluster(
        self,
        prediction,
        n_sigma=3,
        seed_thresh=0.9,
        fg_thresh=0.5,
        min_mask_sum=0,
        min_unclustered_sum=0,
        min_object_size=36,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (7, D, H, W)
        n_sigma: int, default = 3
                Number of dimensions in Raw Image
        seed_thresh : float, default=0.9
                Seediness Threshold defines which pixels are considered to
                identify object centres
        fg_thresh: float, default=0.5
                Foreground Threshold defines which pixels are considered to
                form the "Foreground"
                and hence which  would need to be clustered into unique objects
        min_mask_sum: int
                Only start creating instances, if there are at least
                `min_mask_sum` pixels in foreground!
        min_unclustered_sum: int
                Stop when the number of seed candidates are less than
                `min_unclustered_sum`
        min_object_size: int
            Predicted Objects below this threshold are ignored

        Returns
        ----------
        instance: PyTorch Tensor (D, H, W)
                instance segmentation
        """

        depth, height, width = (
            prediction.size(1),
            prediction.size(2),
            prediction.size(3),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w

        sigma = prediction[3 : 3 + n_sigma]  # n_sigma x d x h x w
        seed_map = torch.sigmoid(
            prediction[3 + n_sigma : 3 + n_sigma + 1]
        )  # 1 x d x h x w
        instance_map = torch.zeros(depth, height, width).short()

        count = 1
        mask = seed_map > fg_thresh
        if (
            mask.sum() > min_mask_sum
        ):  # top level decision: only start creating instances,
            # if there are atleast `min_mask_sum` pixels in foreground!
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(
                n_sigma, -1
            )
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().to(self.device)
            instance_map_masked = torch.zeros(mask.sum()).short().to(self.device)

            while (
                unclustered.sum() > min_unclustered_sum
            ):  # stop when the seed candidates are less than min_unclustered_sum
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed : seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0)
                )

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(depth, height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()
                        count += 1
                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map

    def cluster_local_maxima(
        self,
        prediction,
        n_sigma=3,
        fg_thresh=0.5,
        min_mask_sum=0,
        min_unclustered_sum=0,
        min_object_size=36,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (7, D, H, W)
        n_sigma: int, default = 3
                Number of dimensions in Raw Image
        fg_thresh: float, default=0.5
                Foreground Threshold defines which pixels are considered to
                form the "Foreground"
                and which would need to be clustered into unique objects
        min_mask_sum: int
                Only start creating instances, if there are at least
                `min_mask_sum` pixels in foreground!
        min_unclustered_sum: int
                Stop when the number of seed candidates are less
                than `min_unclustered_sum`
        min_object_size: int
            Predicted Objects below this threshold are ignored

        Returns
        ----------
        instance: PyTorch Tensor (D, H, W)
                instance segmentation
        """

        from scipy.ndimage import gaussian_filter

        depth, height, width = (
            prediction.size(1),
            prediction.size(2),
            prediction.size(3),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]
        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w
        sigma = prediction[3 : 3 + n_sigma]  # n_sigma x d x h x w
        seed_map = torch.sigmoid(prediction[3 + n_sigma : 3 + n_sigma + 1])  # 1 x h x w
        instance_map = torch.zeros(depth, height, width).short()
        # instances = []  # list
        count = 1
        mask_fg = seed_map > fg_thresh

        seed_map_cpu = seed_map.cpu().detach().numpy()
        seed_map_cpu_smooth = gaussian_filter(seed_map_cpu[0], sigma=(1, 2, 2))  # TODO
        coords = peak_local_max(seed_map_cpu_smooth)
        zeros = np.zeros((coords.shape[0], 1), dtype=np.uint8)
        coords = np.hstack((zeros, coords))

        mask_local_max_cpu = np.zeros(seed_map_cpu.shape, dtype=np.bool)
        mask_local_max_cpu[tuple(coords.T)] = True
        mask_local_max = torch.from_numpy(mask_local_max_cpu).bool().to(self.device)

        mask_seed = mask_fg * mask_local_max
        if mask_fg.sum() > min_mask_sum:
            spatial_emb_fg_masked = spatial_emb[mask_fg.expand_as(spatial_emb)].view(
                n_sigma, -1
            )  # fg candidate pixels
            spatial_emb_seed_masked = spatial_emb[
                mask_seed.expand_as(spatial_emb)
            ].view(
                n_sigma, -1
            )  # seed candidate pixels

            sigma_seed_masked = sigma[mask_seed.expand_as(sigma)].view(
                n_sigma, -1
            )  # sigma for seed candidate pixels
            seed_map_seed_masked = seed_map[mask_seed].view(
                1, -1
            )  # seediness for seed candidate pixels

            unprocessed = (
                torch.ones(mask_seed.sum()).short().to(self.device)
            )  # unprocessed seed candidate pixels
            unclustered = (
                torch.ones(mask_fg.sum()).short().to(self.device)
            )  # unclustered fg candidate pixels
            instance_map_masked = torch.zeros(mask_fg.sum()).short().to(self.device)
            while unprocessed.sum() > min_unclustered_sum:
                seed = (seed_map_seed_masked * unprocessed.float()).argmax().item()
                center = spatial_emb_seed_masked[:, seed : seed + 1]
                unprocessed[seed] = 0
                s = torch.exp(sigma_seed_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_fg_masked - center, 2) * s, 0)
                )
                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        count += 1
                        unclustered[proposal] = 0
            instance_map[mask_fg.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map


class Cluster:
    """
    A class used to cluster pixel embeddings in 2D

    Attributes
    ----------
    xym :  float (2, W, D)
            pixel coordinates of tile /grid

    one_hot : bool
            Should be set to True, if the GT label masks are present in a
            one-hot encoded fashion

    grid_x: int
            Length (width) of grid / tile

    grid_y: int
            Height of grid / tile

    pixel_x: float
            if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing
            along the x direction is pixel_x/(grid_x-1) = 1/999
    pixel_y: float
            if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing
            along the y direction is pixel_y/(grid_y-1) = 1/999


    Methods
    -------
    __init__: Initializes an object of class `Cluster_3d`

    cluster_with_gt: use the predicted spatial embeddings
                from all pixels belonging to the GT label mask
                to identify the predicted cluster
                (used during training and validation)

    cluster:    use the  predicted spatial embeddings
                from all pixels in the test image.
                Employs `fg_thresh` and `seed_thresh`
    cluster_local_maxima: use the  predicted spatial embeddings
                from all pixels in the test image.
                Employs only `fg_thresh`
    """

    def __init__(self, grid_y, grid_x, pixel_y, pixel_x, device, one_hot=False):
        """
        Parameters
        ----------
        xym :  float (2, W, D)
                 pixel coordinates of tile /grid

         one_hot : bool
                 Should be set to True, if the GT label masks are present
                 in a one-hot encoded fashion

         grid_x: int
                 Length (width) of grid / tile

         grid_y: int
                 Height of grid / tile

         pixel_x: float
                 if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing
                 along the x direction is pixel_x/(grid_x-1) = 1/999
         pixel_y: float
                 if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing
                 along the y direction is pixel_y/(grid_y-1) = 1/999

        """
        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        xym = torch.cat((xm, ym), 0)

        self.device = device
        self.xym = xym.to(self.device)
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def cluster_with_gt(
        self,
        prediction,
        gt_instance,
        n_sigma=2,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (5, H, W)
        gt_instance : PyTorch Tensor
                Ground Truth Instance Segmentation Label Map

        n_sigma: int, default = 2
                Number of dimensions in Raw Image
        Returns
        ----------
        instance: PyTorch Tensor (H, W)
                instance segmentation
        """

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).short().to(self.device)
        if self.one_hot:
            unique_instances = torch.arange(gt_instance.size(0))
        else:
            unique_instances = gt_instance.unique()
            unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            if self.one_hot:
                mask = gt_instance[id].eq(1).view(1, height, width)
            else:
                mask = gt_instance.eq(id).view(1, height, width)

            center = (
                spatial_emb[mask.expand_as(spatial_emb)]
                .view(2, -1)
                .mean(1)
                .view(2, 1, 1)
            )  # 2 x 1 x 1

            s = (
                sigma[mask.expand_as(sigma)]
                .view(n_sigma, -1)
                .mean(1)
                .view(n_sigma, 1, 1)
            )

            s = torch.exp(s * 10)  # n_sigma x 1 x 1 #
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = dist > 0.5
            if self.one_hot:
                instance_map[proposal] = id.item() + 1
            else:
                instance_map[proposal] = id.item()

        return instance_map

    def cluster_local_maxima(
        self,
        prediction,
        n_sigma=2,
        fg_thresh=0.5,
        min_mask_sum=0,
        min_unclustered_sum=0,
        min_object_size=36,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (5, H, W)
        n_sigma: int, default = 2
                Number of dimensions in Raw Image
        fg_thresh: float, default=0.5
                Foreground Threshold defines which pixels are considered
                to form the Foreground
                and which would need to be clustered into unique objects
        min_mask_sum: int
                Only start creating instances, if there are at least
                `min_mask_sum` pixels in foreground!
        min_unclustered_sum: int
                Stop when the number of seed candidates are less
                than `min_unclustered_sum`
        min_object_size: int
            Predicted Objects below this threshold are ignored

        Returns
        ----------
        instance: PyTorch Tensor (H, W)
                instance segmentation
        """

        from scipy.ndimage import gaussian_filter

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma : 2 + n_sigma + 1])  # 1 x h x w
        instance_map = torch.zeros(height, width).short()
        # instances = []  # list
        count = 1
        mask_fg = seed_map > fg_thresh
        seed_map_cpu = seed_map.cpu().detach().numpy()
        seed_map_cpu_smooth = gaussian_filter(seed_map_cpu[0], sigma=1)
        coords = peak_local_max(seed_map_cpu_smooth)
        zeros = np.zeros((coords.shape[0], 1), dtype=np.uint8)
        coords = np.hstack((zeros, coords))

        mask_local_max_cpu = np.zeros(seed_map_cpu.shape, dtype=np.bool)
        mask_local_max_cpu[tuple(coords.T)] = True
        mask_local_max = torch.from_numpy(mask_local_max_cpu).bool().to(self.device)

        mask_seed = mask_fg * mask_local_max
        if mask_fg.sum() > min_mask_sum:
            spatial_emb_fg_masked = spatial_emb[mask_fg.expand_as(spatial_emb)].view(
                n_sigma, -1
            )  # fg candidate pixels
            spatial_emb_seed_masked = spatial_emb[
                mask_seed.expand_as(spatial_emb)
            ].view(
                n_sigma, -1
            )  # seed candidate pixels

            sigma_seed_masked = sigma[mask_seed.expand_as(sigma)].view(
                n_sigma, -1
            )  # sigma for seed candidate pixels
            seed_map_seed_masked = seed_map[mask_seed].view(
                1, -1
            )  # seediness for seed candidate pixels

            unprocessed = (
                torch.ones(mask_seed.sum()).short().to(self.device)
            )  # unclustered seed candidate pixels
            unclustered = (
                torch.ones(mask_fg.sum()).short().to(self.device)
            )  # unclustered fg candidate pixels
            instance_map_masked = torch.zeros(mask_fg.sum()).short().to(self.device)
            while unprocessed.sum() > min_unclustered_sum:
                seed = (seed_map_seed_masked * unprocessed.float()).argmax().item()
                center = spatial_emb_seed_masked[:, seed : seed + 1]
                unprocessed[seed] = 0
                s = torch.exp(sigma_seed_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_fg_masked - center, 2) * s, 0)
                )
                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        count += 1
                        unclustered[proposal] = 0
            instance_map[mask_fg.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map

    def cluster(
        self,
        prediction,
        n_sigma=2,
        seed_thresh=0.9,
        fg_thresh=0.5,
        min_mask_sum=0,
        min_unclustered_sum=0,
        min_object_size=36,
    ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (5, H, W)
        n_sigma: int, default = 2
                Number of dimensions in Raw Image
        seed_thresh : float, default=0.9
                Seediness Threshold defines which pixels are considered to
                identify object centres
        fg_thresh: float, default=0.5
                Foreground Threshold defines which pixels are considered to
                form the "foreground"
                and hence would need to be clustered into unique objects
        min_mask_sum: int
                Only start creating instances, if there are at least
                `min_mask_sum` pixels in foreground!
        min_unclustered_sum: int
                Stop when the number of seed candidates are less than
                `min_unclustered_sum`
        min_object_size: int
            Predicted Objects below this threshold are ignored

        Returns
        ----------
        instance: PyTorch Tensor (H, W)
                instance segmentation
        """

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w

        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma : 2 + n_sigma + 1])
        # 1 x h x w

        instance_map = torch.zeros(height, width).short()
        # instances = []  # list

        count = 1
        mask = seed_map > fg_thresh

        if mask.sum() > min_mask_sum:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(
                n_sigma, -1
            )
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().to(self.device)
            instance_map_masked = torch.zeros(mask.sum()).short().to(self.device)

            while unclustered.sum() > min_unclustered_sum:
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed : seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0)
                )

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map


class Logger:
    def __init__(self, keys, title=""):
        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print("Created logger with keys:  {}".format(keys))

    def plot(self, save=False, save_dir=""):
        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        count = 0
        for key in self.data:
            if count < 3:
                keys.append(key)
                data = self.data[key]
                ax.plot(range(len(data)), data, marker=".")
                count += 1
        ax.legend(keys, loc="upper right")
        ax.set_title(self.title)

        plt.draw()
        plt.close(fig)
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + ".png"))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + ".csv"))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)


def degrid(meter, grid_size, pixel_size):
    return int(meter * (grid_size - 1) / pixel_size + 1)


def add_samples(samples, ax, n, amax):
    samples_list = []
    for i in range(samples.shape[1]):
        samples_list.append(degrid(samples[ax, i], n, amax))
    return samples_list


def prepare_embedding_for_train_image(
    one_hot,
    grid_x,
    grid_y,
    pixel_x,
    pixel_y,
    predictions,
    instance_ids,
    center_images,
    output,
    instances,
    n_sigma,
):
    xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
    ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
    xym = torch.cat((xm, ym), 0)
    height, width = predictions.size(0), predictions.size(1)
    xym_s = xym[:, 0:height, 0:width].contiguous()
    spatial_emb = torch.tanh(output[0, 0:2]).cpu() + xym_s
    sigma = output[0, 2 : 2 + n_sigma]  # 2/3 Y X
    color_sample = sns.color_palette("dark")
    color_embedding = sns.color_palette("bright")
    color_sample_dic = {}
    color_embedding_dic = {}
    samples_x = {}
    samples_y = {}
    sample_spatial_embedding_x = {}
    sample_spatial_embedding_y = {}
    center_x = {}
    center_y = {}
    sigma_x = {}
    sigma_y = {}
    if one_hot:
        instance_ids += 1  # make the ids go from 1 to ...

    for id in instance_ids:
        if one_hot:
            in_mask = instances[0, id - 1, ...].eq(
                1
            )  # for one_hot, id goes from 1 to ...
        else:
            in_mask = instances[0].eq(id)  # 1 x h x w

        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)  # 2 N
        perm = torch.randperm(xy_in.size(1))
        idx = perm[:5]
        samples = xy_in[:, idx]
        samples_x[id.item()] = add_samples(samples, 0, grid_x - 1, pixel_x)
        samples_y[id.item()] = add_samples(samples, 1, grid_y - 1, pixel_y)

        # embeddings
        spatial_emb_in = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1)
        samples_spatial_embeddings = spatial_emb_in[:, idx]

        sample_spatial_embedding_x[id.item()] = add_samples(
            samples_spatial_embeddings, 0, grid_x - 1, pixel_x
        )
        sample_spatial_embedding_y[id.item()] = add_samples(
            samples_spatial_embeddings, 1, grid_y - 1, pixel_y
        )

        # center_mask = in_mask & center_images[0].byte()
        center_mask = in_mask & center_images[0]
        if center_mask.sum().eq(1):
            center = xym_s[center_mask.expand_as(xym_s)].view(2, 1, 1)
        else:
            xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
            center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

        center_x[id.item()] = degrid(center[0], grid_x - 1, pixel_x)
        center_y[id.item()] = degrid(center[1], grid_y - 1, pixel_y)

        # sigma
        s = sigma[in_mask.expand_as(sigma)].view(n_sigma, -1).mean(1)
        s = torch.exp(s * 10)  # TODO
        sigma_x_tmp = 0.5 / s[0]
        sigma_y_tmp = 0.5 / s[1]
        sigma_x[id.item()] = degrid(torch.sqrt(sigma_x_tmp), grid_x - 1, pixel_x)
        sigma_y[id.item()] = degrid(torch.sqrt(sigma_y_tmp), grid_y - 1, pixel_y)

        # colors
        color_sample_dic[id.item()] = color_sample[int(id % 10)]
        color_embedding_dic[id.item()] = color_embedding[int(id % 10)]
    return (
        center_x,
        center_y,
        samples_x,
        samples_y,
        sample_spatial_embedding_x,
        sample_spatial_embedding_y,
        sigma_x,
        sigma_y,
        color_sample_dic,
        color_embedding_dic,
    )


def prepare_embedding_for_test_image(
    instance_map, output, grid_x, grid_y, pixel_x, pixel_y, predictions, n_sigma
):
    instance_ids = instance_map.unique()
    instance_ids = instance_ids[instance_ids != 0]

    xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
    ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
    xym = torch.cat((xm, ym), 0)
    height, width = instance_map.size(0), instance_map.size(1)
    xym_s = xym[:, 0:height, 0:width].contiguous()
    spatial_emb = torch.tanh(output[0, 0:2]).cpu() + xym_s
    sigma = output[0, 2 : 2 + n_sigma]
    color_sample = sns.color_palette("dark")
    color_embedding = sns.color_palette("bright")
    color_sample_dic = {}
    color_embedding_dic = {}
    samples_x = {}
    samples_y = {}
    sample_spatial_embedding_x = {}
    sample_spatial_embedding_y = {}
    center_x = {}
    center_y = {}
    sigma_x = {}
    sigma_y = {}
    for id in instance_ids:
        in_mask = instance_map.eq(id)
        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)  # 2 N
        perm = torch.randperm(xy_in.size(1))
        idx = perm[:5]
        samples = xy_in[:, idx]
        samples_x[id.item()] = add_samples(samples, 0, grid_x - 1, pixel_x)
        samples_y[id.item()] = add_samples(samples, 1, grid_y - 1, pixel_y)

        # embeddings
        spatial_emb_in = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1)
        samples_spatial_embeddings = spatial_emb_in[:, idx]

        sample_spatial_embedding_x[id.item()] = add_samples(
            samples_spatial_embeddings, 0, grid_x - 1, pixel_x
        )
        sample_spatial_embedding_y[id.item()] = add_samples(
            samples_spatial_embeddings, 1, grid_y - 1, pixel_y
        )
        center_image = predictions[id.item() - 1][
            "center-image"
        ]  # predictions is a list!
        center_mask = in_mask & center_image.byte()

        if center_mask.sum().eq(1):
            center = xym_s[center_mask.expand_as(xym_s)].view(2, 1, 1)
        else:
            xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
            center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

        center_x[id.item()] = degrid(center[0], grid_x - 1, pixel_x)
        center_y[id.item()] = degrid(center[1], grid_y - 1, pixel_y)

        # sigma
        s = sigma[in_mask.expand_as(sigma)].view(n_sigma, -1).mean(1)
        s = torch.exp(s * 10)
        sigma_x_tmp = 0.5 / s[0]
        sigma_y_tmp = 0.5 / s[1]
        sigma_x[id.item()] = degrid(torch.sqrt(sigma_x_tmp), grid_x - 1, pixel_x)
        sigma_y[id.item()] = degrid(torch.sqrt(sigma_y_tmp), grid_y - 1, pixel_y)

        # colors
        color_sample_dic[id.item()] = color_sample[int(id % 10)]
        color_embedding_dic[id.item()] = color_embedding[int(id % 10)]

    return (
        center_x,
        center_y,
        samples_x,
        samples_y,
        sample_spatial_embedding_x,
        sample_spatial_embedding_y,
        sigma_x,
        sigma_y,
        color_sample_dic,
        color_embedding_dic,
    )
