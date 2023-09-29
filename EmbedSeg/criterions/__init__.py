from EmbedSeg.criterions.my_loss import SpatialEmbLoss
from EmbedSeg.criterions.my_loss_3d import SpatialEmbLoss_3d


def get_loss(
    grid_z,
    grid_y,
    grid_x,
    pixel_z,
    pixel_y,
    pixel_x,
    one_hot,
    loss_opts,
):
    if grid_z is None:
        return SpatialEmbLoss(grid_y, grid_x, pixel_y, pixel_x, one_hot, **loss_opts)
    else:
        return SpatialEmbLoss_3d(
            grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, one_hot, **loss_opts
        )
