import torch
import torch.nn as nn
import torch.nn.functional as F

# 简单的Transformer Encoder层示例
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        attn_output, _ = self.attention(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        ffn_output = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_output))
        return src

# 模型示例
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout=0.1):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_size, num_heads, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)

        self.l1 = nn.Linear(embed_size, embed_size)
        self.l2 = nn.Linear(embed_size, embed_size)
        self.l3 = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        embedding = self.embedding(x).transpose(0, 1)  # Shape: [seq_len, batch_size, embed_size]

        embedding_1 = self.l1(embedding[0:7, :, :])
        embedding_2 = self.l2(embedding[7:10, :, :])
        embedding_3 = self.l3(embedding[10:, :, :])
        embedding = torch.cat([embedding_1, embedding_2, embedding_3], dim=0)

        for layer in self.layers:
            embedding = layer(embedding, src_mask=mask)
        output = self.fc_out(embedding).transpose(0, 1)  # Shape: [batch_size, seq_len, vocab_size]
        return output


        
vocab_size = 50
model = SimpleLLM(vocab_size=vocab_size, embed_size=16, num_heads=2, num_layers=3, )

# 输入序列: batch_size=1, seq_len=15
input_ids = torch.randint(0, vocab_size, (1, 15))

# 设置labels
labels = torch.full((1, 15), -100)
labels[:, 12:] = torch.randint(0, vocab_size, (1, 3))
# labels[:, 10:] = torch.randint(0, vocab_size, (1, 5))

# Attention Mask设计  # mask shape: [seq_len, seq_len]
# 1表示可以关注，0表示不可以
mask = torch.zeros(15, 15)


mask[0:7, :] = 1    # 0-6可以关注0-6和7-9和10-14
mask[7:10, 0:10] = 1 # 7-9可以关注0-6和7-9
mask[10:, 0:7] = 1  # 10-14可以关注0:6和10-14
mask[10:, 10:] = 1
# 7-9与10-14不互相关注  already set to 0 by default

def create_lower_triangle_matrix(n):
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            matrix[i][j] = 1
    return torch.tensor(matrix)

lower_triangle = create_lower_triangle_matrix(15)

mask = torch.logical_and(mask, lower_triangle)

attn_mask = (mask == 0).float() * -1e9  # 将mask转为float，并将不能关注的位置设为 -inf


logits = model(input_ids, mask=attn_mask)

shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
shift_logits = shift_logits.view(-1, vocab_size)
shift_labels = shift_labels.view(-1).to(shift_logits.device)
loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(shift_logits, shift_labels)


def hook_fn(grad):
    print("Gradient flowing through l2:", grad.norm().item())

model.l2.weight.register_hook(hook_fn)
model.l2.bias.register_hook(hook_fn)


loss.backward()

for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name} grad norm: {param.grad.norm().item()}")
    elif param.requires_grad:
        print(f"{name} grad is None")




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

