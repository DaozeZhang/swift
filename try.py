import torch
import torch.nn.functional as F


logits = torch.tensor([
    [1, 2, 3],
    [6, 5, 4]
]).float()

res1 = F.gumbel_softmax(logits, tau=1, hard=False)

res2 = F.gumbel_softmax(logits, tau=1, hard=True)

...



#  import torch

# from decord import VideoReader, cpu

# video_io = '/mnt/nas1/.cache/modelscope/media_resources/video_chatgpt/Test_Videos/v_zB8knKX0W8Q.mp4'
# vr = VideoReader(video_io, ctx=cpu(0), num_threads=1)
# max_frame = len(vr) - 1
# fps = float(vr.get_avg_fps())

# video=vr
# ori_fps=fps
# num_segments=8






# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# down_sample = ori_fps  # 下采样到一秒一帧
# # new_fps = ori_fps / down_sample
# if isinstance(video, torch.Tensor):
#     frames = video[::round(down_sample), :, :, :]
# else:  # video is VideoReader obj
#     frames = [torch.tensor(video[i].asnumpy()).float().to(device) for i in range(0, len(video), round(down_sample))]
# frames = torch.stack(frames)

# if frames.shape[1] != 3:  # (N, H, W, C) -> (N, C, H, W)
#     frames = frames.permute(0, 3, 1, 2)

# N, C, H, W = frames.shape
# h_grid_num, w_grid_num = 4, 4
# if H % h_grid_num != 0:
#     trunc_H = H // h_grid_num * h_grid_num
#     frames = frames[:, :, :trunc_H, :]
#     print(f'Truncating H={H} to H={trunc_H} due to {H} % {h_grid_num} != 0')
# if W % w_grid_num != 0:
#     trunc_W = W // w_grid_num * w_grid_num
#     frames = frames[:, :, :, :trunc_W]
#     print(f'Truncating W={W} to W={trunc_W} due to {W} % {w_grid_num} != 0')
# # assert H % h_grid_num == 0 and W % w_grid_num == 0, f'H={H}, h_grid_num={h_grid_num}; W={W}, w_grid_num={w_grid_num}'
# grid_h, grid_w = H // h_grid_num, W // w_grid_num
# n_bin = 32

# unfold = torch.nn.Unfold(kernel_size=(grid_h, grid_w), stride=(grid_h, grid_w))
# patches = unfold(frames)
# patches = patches.reshape(N, C, grid_h, grid_w, h_grid_num * w_grid_num)

# patches = patches.permute(0, 1, 4, 2, 3).reshape(N * C * h_grid_num * w_grid_num, grid_h, grid_w)
# histograms = [torch.histc(patches[i].flatten(), bins=n_bin, min=0, max=255) for i in range(patches.shape[0])]
# histograms = torch.stack(histograms).reshape(N, C, h_grid_num * w_grid_num, -1)  # (N, C, n_patch, n_bin)
# difference = torch.diff(histograms, dim=0)  # (N-1, C, n_patch, n_bin)
# difference = torch.sum(torch.abs(difference), dim=(1, 2, 3)) / (C * H * W * n_bin * 255)


# def find_peaks(tensor, device):
#     padded_tensor = torch.concat([tensor.new_zeros(1).to(device),
#                                   tensor,
#                                   tensor.new_zeros(1).to(device)])
#     is_peak = (padded_tensor[:-2] < padded_tensor[1:-1]) & (padded_tensor[1:-1] > padded_tensor[2:])
#     peak_values = tensor[is_peak]
#     peak_indices = torch.tensor(list(range(len(tensor)))).to(device)[is_peak]
#     return peak_values, peak_indices


# peak_values, peak_indices = find_peaks(difference, device)
# if len(peak_values) < num_segments + 1:
#     ret = torch.linspace(0, len(video) - 1, num_segments).round().int()

# keep_mask = torch.tensor([True] * len(peak_indices)).to(device)
# key_indices = []
# already_is_key_num = 0
# for t in [0, difference.shape[0] - 1]:
#     if t in peak_indices:
#         key_indices += [t]
#         already_is_key_num += 1
#         keep_mask = torch.logical_and(keep_mask, peak_indices != t)

# peak_indices, peak_values = peak_indices[keep_mask], peak_values[keep_mask]

# top_peaks_values, top_peaks_indices = torch.topk(peak_values, k=num_segments + 1 - already_is_key_num, dim=0,
#                                                  largest=True)
# key_indices += peak_indices[top_peaks_indices].tolist()

# sort_key_indices = sorted(key_indices)
# # draw_key_indices(difference, key_indices)

# semantic_indices = [round((sort_key_indices[i] + sort_key_indices[i + 1]) / 2) for i in range(num_segments)]

# seman_indices_in_ori = [round(down_sample) * idx for idx in semantic_indices]
# seman_indices_in_ori = torch.tensor(seman_indices_in_ori)

