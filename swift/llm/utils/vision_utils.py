import base64
import math
import os
from io import BytesIO
from typing import Any, Callable, List, TypeVar, Union

import numpy as np
import requests
import torch
from packaging import version

# >>> internvl
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, ((i //
                                                                        (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# <<< internvl


def rescale_image(img: 'PIL.Image.Image', rescale_image: int = -1) -> 'PIL.Image.Image':
    import torchvision.transforms as T
    width = img.width
    height = img.height
    if rescale_image <= 0 or width * height <= rescale_image:
        return img

    ratio = width / height
    height_scaled = math.pow(rescale_image / ratio, 0.5)
    width_scaled = height_scaled * ratio
    return T.Resize((int(width_scaled), int(height_scaled)))(img)


_T = TypeVar('_T')


def load_file(path: Union[str, _T]) -> Union[BytesIO, _T]:
    res = path
    if isinstance(path, str):
        path = path.strip()
        if path.startswith('http'):
            request_kwargs = {}
            timeout = float(os.getenv('TIMEOUT', '60'))
            if timeout > 0:
                request_kwargs['timeout'] = timeout
            content = requests.get(path, **request_kwargs).content
            res = BytesIO(content)
        elif os.path.exists(path):
            with open(path, 'rb') as f:
                res = BytesIO(f.read())
        else:  # base64_str
            import binascii
            try:
                data = base64.b64decode(path)
                res = BytesIO(data)
            except (ValueError, binascii.Error) as error:
                if len(path) < 200:
                    raise ValueError(f'invalid image: "{path}"')
                else:
                    raise ValueError(f'invalid image: {error}')
    return res


def load_file_decorator(func):

    def new_func(path, *args, **kwargs):
        path = load_file(path)
        res = func(path, *args, **kwargs)
        return res

    return new_func


@load_file_decorator
def load_image(image: Union['PIL.Image.Image', BytesIO]) -> 'PIL.Image.Image':
    from PIL import Image
    if isinstance(image, BytesIO):
        image = Image.open(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def load_batch(path_list: List[Union[str, None, Any, BytesIO]],
               load_func: Callable[[Any], _T] = load_image) -> List[_T]:
    res = []
    assert isinstance(path_list, (list, tuple)), f'path_list: {path_list}'
    for path in path_list:
        if path is None:  # ignore None
            continue
        res.append(load_func(path))
    return res


def _get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def transform_image(image, input_size=448, max_num=12):
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def draw_key_indices(difference, key_indices):
    import matplotlib.pyplot as plt
    # 将Tensor转换为NumPy数组
    data_array = difference.cpu().numpy()
    red_points_indices = np.array(key_indices)

    plt.figure(figsize=(10, 5))
    plt.plot(data_array, label='Data', color='blue')

    for index in red_points_indices:
        plt.plot(index, data_array[index], 'ro')  # 'ro' 表示红色圆形标记

    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Diff')

    plt.grid(True)
    plt.show()

def get_semantic_indices(video, ori_fps, num_segments ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    down_sample = ori_fps   # 下采样到一秒一帧
    new_fps = ori_fps / down_sample
    frames = [frame.float().to(device) for frame in video[ ::round(down_sample) ]]
    frames = torch.stack(frames)

    if frames.shape[1] != 3:    # (N, H, W, C) -> (N, C, H, W)
        frames = frames.permute(0, 3, 1, 2)

    N, C, H, W = frames.shape
    h_grid_num, w_grid_num = 4, 4
    assert H % h_grid_num == 0 and W % w_grid_num == 0, f'H={H}, h_grid_num={h_grid_num}; W={W}, w_grid_num={w_grid_num}'
    grid_h, grid_w = H // h_grid_num, W // w_grid_num
    n_bin = 32

    unfold = torch.nn.Unfold(kernel_size=(grid_h, grid_w), stride=(grid_h, grid_w))
    patches = unfold(frames)
    patches = patches.reshape(N, C, grid_h, grid_w, h_grid_num * w_grid_num)

    patches = patches.permute(0, 1, 4, 2, 3).reshape(N * C * h_grid_num * w_grid_num, grid_h, grid_w)
    histograms = [torch.histc(patches[i].flatten(), bins=n_bin, min=0, max=255) for i in range(patches.shape[0])]
    histograms = torch.stack(histograms).reshape(N, C, h_grid_num * w_grid_num, -1)  # (N, C, n_patch, n_bin)
    difference = torch.diff(histograms, dim=0)                                       # (N-1, C, n_patch, n_bin)
    difference = torch.sum(torch.abs(difference), dim=(1, 2, 3)) / (C * H * W * n_bin * 255)

    def find_peaks(tensor, device):
        padded_tensor = torch.concat([tensor.new_zeros(1).to(device),
                                      tensor,
                                      tensor.new_zeros(1).to(device)] )
        is_peak = (padded_tensor[:-2] < padded_tensor[1:-1]) & (padded_tensor[1:-1] > padded_tensor[2:])
        peak_values = tensor[is_peak]
        peak_indices = tensor.nonzero(as_tuple=False)[is_peak]
        return peak_values, peak_indices.squeeze()

    peak_values, peak_indices = find_peaks(difference, device)

    keep_mask = torch.tensor([True] * len(peak_indices)).to(device)
    key_indices = []
    already_is_key_num = 0
    for t in [0, difference.shape[0]-1]:
        if t in peak_indices:
            key_indices += [t]
            already_is_key_num += 1
            keep_mask = torch.logical_and(keep_mask, peak_indices != t)

    peak_indices, peak_values = peak_indices[keep_mask], peak_values[keep_mask]

    top_peaks_values, top_peaks_indices = torch.topk(peak_values, k=num_segments + 1 - already_is_key_num, dim=0, largest=True)
    key_indices += peak_indices[top_peaks_indices].tolist()

    sort_key_indices = sorted(key_indices)
    draw_key_indices(difference, key_indices)

    semantic_indices = [round((sort_key_indices[i] + sort_key_indices[i + 1]) / 2) for i in range(num_segments)]
    semantic_indices = torch.tensor(semantic_indices).to(device)
    return semantic_indices, frames, new_fps


@load_file_decorator
def load_video_internvl(video_io: BytesIO, bound=None, num_segments=32):
    from decord import VideoReader, cpu
    from PIL import Image
    vr = VideoReader(video_io, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    use_key_frames = False
    if use_key_frames:
        frame_indices = _get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    else:
        frame_indices, vr, new_fps = get_semantic_indices(vr, fps, num_segments)

    images = []
    for frame_index in frame_indices:
        images.append(Image.fromarray(vr[frame_index].asnumpy()).convert('RGB'))

    # save_type_str = '_uniform' if not use_key_frames else '_semantic'
    # for i in range(num_segments):
    #     images[i].save(f'/mnt/nas1/daoze/code/swift/_internvl_{save_type_str}_{int(frame_indices[i]/fps)}sec.jpg')
    return images


def draw_plot(img_dir: str, bbox: List[int], bbox_type: str, output_file: str):
    from PIL import Image, ImageDraw
    from .template import Template
    image = Image.open(img_dir)

    objects = [{'bbox': bbox, 'bbox_type': bbox_type, 'image': 0}]
    Template.normalize_bbox(objects, [image], 'real')
    bbox = objects[0]['bbox']
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline='red', width=2)
    image.save(output_file)


@load_file_decorator
def load_video_cogvlm2(video_io: BytesIO) -> np.ndarray:
    from decord import cpu, VideoReader, bridge
    bridge.set_bridge('torch')
    clip_end_sec = 60
    clip_start_sec = 0
    num_frames = 24
    decord_vr = VideoReader(video_io, ctx=cpu(0))
    duration = len(decord_vr)  # duration in terms of frames
    start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
    end_frame = min(duration, int(clip_end_sec * decord_vr.get_avg_fps())) if \
        clip_end_sec is not None else duration
    frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


@load_file_decorator
def load_video_llava(video_io: BytesIO) -> np.ndarray:
    import av
    container = av.open(video_io)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format='rgb24') for x in frames])


@load_file_decorator
def load_video_minicpmv_mplug_owl3(video_io: BytesIO, max_num_frames):
    from PIL import Image
    from decord import VideoReader, cpu  # pip install decord

    def uniform_sample(_l, _n):
        gap = len(_l) / _n
        idxs = [int(i * gap + gap / 2) for i in range(_n)]
        return [_l[i] for i in idxs]

    vr = VideoReader(video_io, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames


@load_file_decorator
def load_audio_qwen(audio_io: BytesIO, sampling_rate: int):
    import librosa
    return librosa.load(audio_io, sr=sampling_rate)[0]


def load_video_qwen2(video_path: str):
    from .template import get_env_args
    import torchvision
    from torchvision import io, transforms
    from qwen_vl_utils.vision_process import (round_by_factor, FPS, FRAME_FACTOR, FPS_MIN_FRAMES, FPS_MAX_FRAMES,
                                              VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS, smart_resize,
                                              ceil_by_factor, floor_by_factor)
    from torchvision.transforms import InterpolationMode

    if version.parse(torchvision.__version__) >= version.parse('0.19'):
        video_path = load_file(video_path)
    video, _, info = io.read_video(
        video_path,
        pts_unit='sec',
        output_format='TCHW',
    )
    nframes = get_env_args('nframes', int, None)
    fps = get_env_args('fps', int, None)
    size_factor = get_env_args('size_factor', int, FRAME_FACTOR)
    assert not (fps and nframes), 'Only accept either `fps` or `nframes`'
    if nframes is not None:
        nframes = round_by_factor(nframes, size_factor)
    else:
        fps = FPS
        nframes = video.size(0) / info['video_fps'] * fps
        nframes = round_by_factor(nframes, size_factor)
        min_frames = get_env_args('min_frames', int, FPS_MIN_FRAMES)
        max_frames = get_env_args('max_frames', int, FPS_MAX_FRAMES)
        if nframes < min_frames:
            nframes = ceil_by_factor(min_frames, size_factor)
        if nframes > max_frames:
            nframes = floor_by_factor(max_frames, size_factor)

    if not (size_factor <= nframes and nframes <= video.size(0)):
        raise ValueError(f'nframes should in interval [{size_factor}, {video.size(0)}], but got {nframes}.')

    use_key_frames = False
    ori_fps = info['video_fps']
    if not use_key_frames:
        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        new_fps = ori_fps
    else:
        idx, video, new_fps = get_semantic_indices(video=video, ori_fps=ori_fps, num_segments=nframes)
        idx = idx.cpu()

    height, width = video.shape[2:]
    video = video[idx]

    min_pixels = get_env_args('min_pixels', int, VIDEO_MIN_PIXELS)
    total_pixels = get_env_args('total_pixels', int, VIDEO_TOTAL_PIXELS)
    max_pixels = get_env_args('max_pixels', int, None)
    if max_pixels is None:
        max_pixels = VIDEO_MAX_PIXELS
        max_pixels = max(min(max_pixels, total_pixels / nframes * size_factor), min_pixels * 1.05)
    # resize
    resized_height = get_env_args('resized_height', int, None)
    resized_width = get_env_args('resized_width', int, None)
    if resized_height and resized_width:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=size_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    if use_key_frames:
        video = torch.clamp(video, 0, 255).type(torch.uint8)  # 使用关键帧后 上面一步resize会导致图像超出范围 并且要转成uint8

    import torchvision.transforms as transforms
    save_type_str = '_uniform' if not use_key_frames else '_semantic'
    for i in range(nframes):
        img = transforms.ToPILImage()(video[i].type(torch.uint8))
        img.save(f'/mnt/nas1/daoze/code/swift/_qwen2vl_{save_type_str}_{int(idx[i]/new_fps)}sec.jpg')

    return video


if __name__ == '__main__':
    # A test main to draw bbox
    draw_plot('man.jpg', [354, 462, 580, 738], 'norm_1000', 'man_bbox.jpg')
