import torch
import numpy as np
# from model.dataset.homography_warper import get_rotation_matrix2d, warp_affine_crop



#
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


pi = torch.tensor(3.14159265358979323846)


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    r"""Function that converts angles from degrees to radians.
    Args:
        tensor (torch.Tensor): Tensor of arbitrary shape.
    Returns:
        torch.Tensor: tensor with same shape as input.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.


def angle_to_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """
    Creates a rotation matrix out of angles in degrees
    Args:
        angle: (torch.Tensor): tensor of angles in degrees, any shape.
    Returns:
        torch.Tensor: tensor of *x2x2 rotation matrices.
    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*, 2, 2)`
    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = kornia.angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = deg2rad(angle)
    cos_a: torch.Tensor = torch.cos(ang_rad)
    sin_a: torch.Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def get_rotation_matrix2d(
        center: torch.Tensor,
        angle: torch.Tensor,
        scale: torch.Tensor) -> torch.Tensor:
    r"""Calculates an affine matrix of 2D rotation.
    The function calculates the following matrix:
    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}
    where
    .. math::
        \alpha = \text{scale} \cdot cos(\text{radian}) \\
        \beta = \text{scale} \cdot sin(\text{radian})
    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.
    Args:
        center (Tensor): center of the rotation in the source image.
        angle (Tensor): rotation radian in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner).
        scale (Tensor): isotropic scale factor.
    Returns:
        Tensor: the affine matrix of 2D rotation.
    Shape:
        - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
        - Output: :math:`(B, 2, 3)`
    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1)
        >>> radian = 45. * torch.ones(1)
        >>> M = kornia.get_rotation_matrix2d(center, radian, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    """
    if not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if not torch.is_tensor(angle):
        raise TypeError("Input radian type is not a torch.Tensor. Got {}"
                        .format(type(angle)))
    if not torch.is_tensor(scale):
        raise TypeError("Input scale type is not a torch.Tensor. Got {}"
                        .format(type(scale)))
    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}"
                         .format(center.shape))
    if not len(angle.shape) == 1:
        raise ValueError("Input radian must be a B tensor. Got {}"
                         .format(angle.shape))
    if not len(scale.shape) == 1:
        raise ValueError("Input scale must be a B tensor. Got {}"
                         .format(scale.shape))
    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got {}"
                         .format(center.shape, angle.shape, scale.shape))
    # convert radian and apply scale
    scaled_rotation: torch.Tensor = angle_to_rotation_matrix(angle) * scale.view(-1, 1, 1)
    alpha: torch.Tensor = scaled_rotation[:, 0, 0]
    beta: torch.Tensor = scaled_rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x: torch.Tensor = center[..., 0]
    y: torch.Tensor = center[..., 1]

    # create output tensor
    batch_size: int = center.shape[0]
    M: torch.Tensor = torch.zeros(
        batch_size, 2, 3, device=center.device, dtype=center.dtype)
    M[..., 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (torch.tensor(1.) - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (torch.tensor(1.) - alpha) * y
    return M

def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.
    Examples::
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = kornia.convert_points_to_homogeneous(input)  # BxNx4
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.
    Examples::
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = kornia.convert_points_from_homogeneous(input)  # BxNx2
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))

    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    # we check for points at infinity
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: torch.Tensor = torch.abs(z_vec) > eps
    scale: torch.Tensor = torch.ones_like(z_vec).masked_scatter_(
        mask, torch.tensor(1.0).to(points.device) / z_vec[mask])

    return scale * points[..., :-1]

def transform_points(trans_01: torch.Tensor,
                     points_1: torch.Tensor) -> torch.Tensor:
    r"""Function that applies transformations to a set of points.
    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.
    Shape:
        - Output: :math:`(B, N, D)`
    Examples:
        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = kornia.transform_points(trans_01, points_1)  # BxNx3
    """
    if not torch.is_tensor(trans_01) or not torch.is_tensor(points_1):
        raise TypeError("Input type is not a torch.Tensor")
    if not trans_01.device == points_1.device:
        raise TypeError("Tensor must be in the same device")
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError("Input batch size must be the same for both tensors or 1")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differe by one unit")
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.matmul(
        trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    return points_0


def multi_linspace(a, b, num, endpoint=True, device='cpu', dtype=torch.float):
    """This function is just like np.linspace, but will create linearly
    spaced vectors from a start to end vector.
    Inputs:
        a - Start vector.
        b - End vector.
        num - Number of samples to generate. Default is 50. Must be above 0.
        endpoint - If True, b is the last sample.
                   Otherwise, it is not included. Default is True.
    """

    return a[..., None] + (b-a)[..., None]/(num-endpoint) * torch.arange(num, device=device, dtype=dtype)


def create_batched_meshgrid(
        x_min: torch.Tensor,
        y_min: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor,
        height: int,
        width: int,
        device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """Generates a coordinate grid for an image.
    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample
    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.
    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs = multi_linspace(x_min, x_max, width, device=device, dtype=torch.float)
    ys = multi_linspace(y_min, y_max, height, device=device, dtype=torch.float)

    # generate grid by stacking coordinates
    bs = x_min.shape[0]
    batched_grid_i_list = list()
    for i in range(bs):
        # batched_grid_i_list.append(torch.stack(torch.meshgrid([xs[i], ys[i]])).transpose(1, 2))  # 2xHxW
        grid_i = torch.stack(torch.meshgrid(xs[i], ys[i], indexing="ij"))  # 加 indexing
        batched_grid_i_list.append(grid_i.transpose(1, 2))
    batched_grid: torch.Tensor = torch.stack(batched_grid_i_list, dim=0)
    return batched_grid.permute(0, 2, 3, 1)  # BxHxWx2


def homography_warp(patch_src: torch.Tensor,
                    centers: torch.Tensor,
                    dst_homo_src: torch.Tensor,
                    dsize: Tuple[int, int],
                    mode: str = 'bilinear',
                    padding_mode: str = 'zeros') -> torch.Tensor:
    r"""Function that warps image patchs or tensors by homographies.
    See :class:`~kornia.geometry.warp.HomographyWarper` for details.
    Args:
        patch_src (torch.Tensor): The image or tensor to warp. Should be from
                                  source of shape :math:`(N, C, H, W)`.
        dst_homo_src (torch.Tensor): The homography or stack of homographies
                                     from source to destination of shape
                                     :math:`(N, 3, 3)`.
        dsize (Tuple[int, int]): The height and width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
    Return:
        torch.Tensor: Patch sampled at locations from source to destination.
    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = kornia.homography_warp(input, homography, (32, 32))
    """

    out_height, out_width = dsize
    image_height, image_width = patch_src.shape[-2:]
    x_min = 2. * (centers[..., 0] - out_width/2) / image_width - 1.
    y_min = 2. * (centers[..., 1] - out_height/2) / image_height - 1.
    x_max = 2. * (centers[..., 0] + out_width/2) / image_width - 1.
    y_max = 2. * (centers[..., 1] + out_height/2) / image_height - 1.
    warper = HomographyWarper(x_min, y_min, x_max, y_max, out_height, out_width, mode, padding_mode)
    return warper(patch_src, dst_homo_src)


def normal_transform_pixel(height, width):

    tr_mat = torch.Tensor([[1.0, 0.0, -1.0],
                           [0.0, 1.0, -1.0],
                           [0.0, 0.0, 1.0]])  # 1x3x3

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)

    tr_mat = tr_mat.unsqueeze(0)

    return tr_mat


def src_norm_to_dst_norm(dst_pix_trans_src_pix: torch.Tensor,
                         dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]) -> torch.Tensor:
    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst
    # the devices and types
    device: torch.device = dst_pix_trans_src_pix.device
    dtype: torch.dtype = dst_pix_trans_src_pix.dtype
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(
        src_h, src_w).to(device, dtype)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(
        dst_h, dst_w).to(device, dtype)
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = (
        dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    )
    return dst_norm_trans_src_norm


def transform_warp_impl(src: torch.Tensor, centers: torch.Tensor, dst_pix_trans_src_pix: torch.Tensor,
                        dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int],
                        grid_mode: str, padding_mode: str) -> torch.Tensor:
    """Compute the transform in normalized cooridnates and perform the warping.
    """
    dst_norm_trans_src_norm: torch.Tensor = src_norm_to_dst_norm(
        dst_pix_trans_src_pix, dsize_src, dsize_src)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    return homography_warp(src, centers, src_norm_trans_dst_norm, dsize_dst, grid_mode, padding_mode)


class HomographyWarper(nn.Module):
    r"""Warps image patches or tensors by homographies.
    .. math::
        X_{dst} = H_{src}^{\{dst\}} * X_{src}
    Args:
        height (int): The height of the image to warp.
        width (int): The width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
    """

    def __init__(
            self,
            x_min: torch.Tensor,
            y_min: torch.Tensor,
            x_max: torch.Tensor,
            y_max: torch.Tensor,
            height: int,
            width: int,
            mode: str = 'bilinear',
            padding_mode: str = 'zeros') -> None:
        super(HomographyWarper, self).__init__()
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode

        # create base grid to compute the flow
        self.grid: torch.Tensor = create_batched_meshgrid(x_min, y_min, x_max, y_max, height, width)

    def warp_grid(self, dst_homo_src: torch.Tensor) -> torch.Tensor:
        r"""Computes the grid to warp the coordinates grid by an homography.
        Args:
            dst_homo_src (torch.Tensor): Homography or homographies (stacked) to
                              transform all points in the grid. Shape of the
                              homography has to be :math:`(N, 3, 3)`.
        Returns:
            torch.Tensor: the transformed grid of shape :math:`(N, H, W, 2)`.
        """
        batch_size: int = dst_homo_src.shape[0]
        device: torch.device = dst_homo_src.device
        dtype: torch.dtype = dst_homo_src.dtype
        # expand grid to match the input batch size
        grid: torch.Tensor = self.grid
        if len(dst_homo_src.shape) == 3:  # local homography case
            dst_homo_src = dst_homo_src.view(batch_size, 1, 3, 3)  # NxHxWx3x3
        # perform the actual grid transformation,
        # the grid is copied to input device and casted to the same type
        flow: torch.Tensor = transform_points(
            dst_homo_src, grid.to(device).to(dtype))  # NxHxWx2
        return flow.view(batch_size, self.height, self.width, 2)  # NxHxWx2

    def forward(  # type: ignore
            self,
            patch_src: torch.Tensor,
            dst_homo_src: torch.Tensor) -> torch.Tensor:
        r"""Warps an image or tensor from source into reference frame.
        Args:
            patch_src (torch.Tensor): The image or tensor to warp.
                                      Should be from source.
            dst_homo_src (torch.Tensor): The homography or stack of homographies
             from source to destination. The homography assumes normalized
             coordinates [-1, 1].
        Return:
            torch.Tensor: Patch sampled at locations from source to destination.
        Shape:
            - Input: :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
            - Output: :math:`(N, C, H, W)`
        Example:
            >>> input = torch.rand(1, 3, 32, 32)
            >>> homography = torch.eye(3).view(1, 3, 3)
            >>> warper = kornia.HomographyWarper(32, 32)
            >>> output = warper(input, homography)  # NxCxHxW
        """
        if not dst_homo_src.device == patch_src.device:
            raise TypeError("Patch and homography must be on the same device. \
                            Got patch.device: {} dst_H_src.device: {}."
                            .format(patch_src.device, dst_homo_src.device))

        return F.grid_sample(patch_src, self.warp_grid(dst_homo_src),  # type: ignore
                             mode=self.mode, padding_mode=self.padding_mode, align_corners=True)


def warp_affine_crop(src: torch.Tensor, centers: torch.Tensor, M: torch.Tensor,
                dsize: Tuple[int, int], flags: str = 'bilinear',
                padding_mode: str = 'zeros') -> torch.Tensor:
    r"""Applies an affine transformation to a tensor.

    The function warp_affine transforms the source tensor using
    the specified matrix:

    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )

    Args:
        src (torch.Tensor): input tensor of shape :math:`(B, C, H, W)`.
        M (torch.Tensor): affine transformation of shape :math:`(B, 2, 3)`.
        dsize (Tuple[int, int]): size of the output image (height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.

    Returns:
        torch.Tensor: the warped tensor.

    Shape:
        - Output: :math:`(B, C, H, W)`

    .. note::
       See a working example `here <https://kornia.readthedocs.io/en/latest/
       tutorials/warp_affine.html>`__.
    """
    if not torch.is_tensor(src):
        raise TypeError("Input src type is not a torch.Tensor. Got {}"
                        .format(type(src)))

    if not torch.is_tensor(M):
        raise TypeError("Input M type is not a torch.Tensor. Got {}"
                        .format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}"
                         .format(src.shape))

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: torch.Tensor = F.pad(M, [0, 0, 0, 1, 0, 0],
                                mode="constant", value=0)
    M_3x3[:, 2, 2] += 1.0

    # launches the warper
    h, w = src.shape[-2:]
    return transform_warp_impl(src, centers, M_3x3, (h, w), dsize, flags, padding_mode)
#


class Map(object):
    def __init__(self, data, homography, description=None):
        self.data = data
        self.homography = homography
        self.description = description

    def as_image(self):
        raise NotImplementedError

    def get_cropped_maps(self, world_pts, patch_size, rotation=None, device='cpu'):
        raise NotImplementedError

    def to_map_points(self, scene_pts):
        raise NotImplementedError


class GeometricMap(Map):
    """
    A Geometric Map is a int tensor of shape [layers, x, y]. The homography must transform a point in scene
    coordinates to the respective point in map coordinates.

    :param data: Numpy array of shape [layers, x, y]
    :param homography: Numpy array of shape [3, 3]
    """
    def __init__(self, data, homography, description=None):
        #assert isinstance(data.dtype, np.floating), "Geometric Maps must be float values."
        super(GeometricMap, self).__init__(data, homography, description=description)

        self._last_padding = None
        self._last_padded_map = None
        self._torch_map = None

    def torch_map(self, device):
        if self._torch_map is not None:
            return self._torch_map
        self._torch_map = torch.tensor(self.data, dtype=torch.uint8, device=device)
        return self._torch_map

    def as_image(self):
        # We have to transpose x and y to rows and columns. Assumes origin is lower left for image
        # Also we move the channels to the last dimension
        return (np.transpose(self.data, (2, 1, 0))).astype(np.uint)

    def get_padded_map(self, padding_x, padding_y, device):
        if self._last_padding == (padding_x, padding_y):
            return self._last_padded_map
        else:
            self._last_padding = (padding_x, padding_y)
            self._last_padded_map = torch.full((self.data.shape[0],
                                                self.data.shape[1] + 2 * padding_x,
                                                self.data.shape[2] + 2 * padding_y),
                                               False, dtype=torch.uint8)
            self._last_padded_map[..., padding_x:-padding_x, padding_y:-padding_y] = self.torch_map(device)
            return self._last_padded_map

    @staticmethod
    def batch_rotate(map_batched, centers, angles, out_height, out_width):
        """
        As the input is a map and the warp_affine works on an image coordinate system we would have to
        flip the y axis updown, negate the angles, and flip it back after transformation.
        This, however, is the same as not flipping at and not negating the radian.

        :param map_batched:
        :param centers:
        :param angles:
        :param out_height:
        :param out_width:
        :return:
        """
        M = get_rotation_matrix2d(centers, angles, torch.ones_like(angles))
        rotated_map_batched = warp_affine_crop(map_batched, centers, M,
                                               dsize=(out_height, out_width), padding_mode='zeros')

        return rotated_map_batched

    @classmethod
    def get_cropped_maps_from_scene_map_batch(cls, maps, scene_pts, patch_size, rotation=None, device='cpu'):
        """
        Returns rotated patches of each map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param maps: List of GeometricMap objects [bs]
        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-x, -y, +x, +y]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        batch_size = scene_pts.shape[0]
        lat_size = 2 * np.max((patch_size[0], patch_size[2]))
        long_size = 2 * np.max((patch_size[1], patch_size[3]))
        assert lat_size % 2 == 0, "Patch width must be divisible by 2"
        assert long_size % 2 == 0, "Patch length must be divisible by 2"
        lat_size_half = lat_size // 2
        long_size_half = long_size // 2

        context_padding_x = int(np.ceil(np.sqrt(2) * lat_size))
        context_padding_y = int(np.ceil(np.sqrt(2) * long_size))

        # centers = torch.tensor([s_map.to_map_points(scene_pts[np.newaxis, i]) for i, s_map in enumerate(maps)],
        #                        dtype=torch.long, device=device).squeeze(dim=1) \
        centers = torch.from_numpy(np.stack([s_map.to_map_points(scene_pts[np.newaxis, i]) for i, s_map in enumerate(maps)],axis=0)).long().to(device).squeeze(1) \
                  + torch.tensor([context_padding_x, context_padding_y], device=device, dtype=torch.long)

        padded_map = [s_map.get_padded_map(context_padding_x, context_padding_y, device=device) for s_map in maps]

        padded_map_batched = torch.stack([padded_map[i][...,
                                          centers[i, 0] - context_padding_x: centers[i, 0] + context_padding_x,
                                          centers[i, 1] - context_padding_y: centers[i, 1] + context_padding_y]
                                          for i in range(centers.shape[0])], dim=0)

        center_patches = torch.tensor([[context_padding_y, context_padding_x]],
                                      dtype=torch.int,
                                      device=device).repeat(batch_size, 1)

        if rotation is not None:
            angles = torch.Tensor(rotation)
        else:
            angles = torch.zeros(batch_size)

        rotated_map_batched = cls.batch_rotate(padded_map_batched/255.,
                                                center_patches.float(),
                                                angles,
                                                long_size,
                                                lat_size)

        del padded_map_batched

        return rotated_map_batched[...,
               long_size_half - patch_size[1]:(long_size_half + patch_size[3]),
               lat_size_half - patch_size[0]:(lat_size_half + patch_size[2])]

    def get_cropped_maps(self, scene_pts, patch_size, rotation=None, device='cpu'):
        """
        Returns rotated patches of the map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-lat, -long, +lat, +long]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        return self.get_cropped_maps_from_scene_map_batch([self]*scene_pts.shape[0], scene_pts,
                                                          patch_size, rotation=rotation, device=device)

    def to_map_points(self, scene_pts):
        org_shape = None
        if len(scene_pts.shape) > 2:
            org_shape = scene_pts.shape
            scene_pts = scene_pts.reshape((-1, 2))
        N, dims = scene_pts.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = scene_pts.T
        map_points = (self.homography @ points_with_one).T[..., :dims]
        if org_shape is not None:
            map_points = map_points.reshape(org_shape)
        return map_points


class ImageMap(Map):  # TODO Implement for image maps -> watch flipped coordinate system
    def __init__(self):
        raise NotImplementedError