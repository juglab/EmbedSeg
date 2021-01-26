from EmbedSeg.criterions.my_loss import SpatialEmbLoss


def get_loss(grid_y, grid_x, pixel_y, pixel_x, one_hot, loss_opts):
    return SpatialEmbLoss(grid_y, grid_x, pixel_y, pixel_x, one_hot, **loss_opts)
    