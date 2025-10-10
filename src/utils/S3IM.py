import torch
import torchgeometry as tgm


class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 5)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 96)
        patch_width (height): width of virtual patch(default: 96)
    """
    def __init__(self, kernel_size=5, repeat_time=10, patch_height=96, patch_width=96):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = tgm.losses.SSIM(window_size=kernel_size, reduction='none')

    def forward(self, src, tar):
        r"""Calculate S3IM loss.
        Arguments:
            src (Tensor): source image with shape (N, 1, H, W)
            tar (Tensor): target image with shape (N, 1, H, W)
        Returns:
            Tensor: S3IM loss.
        """
        # flatten the image keep the batch dimension
        src_vec = src.reshape(src.size(0), -1)
        tar_vec = tar.reshape(tar.size(0), -1)
        loss = 0.0
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(tar_vec.size(1))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(tar_vec.size(1))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[:, res_index]
        src_all = src_vec[:, res_index]
        tar_patch = tar_all.reshape(tar.size(0), 1, self.patch_height, self.patch_width * self.repeat_time)
        src_patch = src_all.reshape(src.size(0), 1, self.patch_height, self.patch_width * self.repeat_time)
        loss = self.ssim_loss(src_patch, tar_patch)
        return loss


# unit test
if __name__ == '__main__':
    s3im = S3IM()
    src = torch.rand(4, 1, 96, 96)
    tar = torch.rand(4, 1, 96, 96)
    loss = s3im(src, tar)
    print(loss)
    print(loss.size())





