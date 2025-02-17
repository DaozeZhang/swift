# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Dict, List, Literal

import torch
from torch import nn

from swift.utils import get_env_args, is_deepspeed_enabled, get_logger
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs, SophiaTemplateInputs
from ..utils import Context, findall
from ..vision_utils import load_video_internvl, load_video_sophia, get_hierar_mask, transform_image
from .microsoft import Phi3TemplateMeta
from .utils import ChatmlTemplateMeta

logger = get_logger()

class InternvlTemplate(Template):
    skip_prompt = False
    num_image_token = 256

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            image_context = ['<img><image></img>\n']
        else:
            image_context = ['<img>', [-100], '</img>\n']
        return image_context

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        pixel_values = None
        images = inputs.images
        if images:
            labels = encoded.get('labels')
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 12)
            pixel_values_images = [transform_image(image, input_size, max_num) for image in images]
            pixel_values = torch.cat(pixel_values_images, dim=0).to(self.model_info.torch_dtype)
            image_bs = pixel_values.shape[0]

            idx, idx2 = idx_list[0], idx_list[-1]  # remove [-100, -100]
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * image_bs
            input_ids = input_ids[:idx] + img_tokens + input_ids[idx2 + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(img_tokens) + labels[idx2 + 1:]
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
        encoded['pixel_values'] = pixel_values
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs['input_ids']
        inputs_embeds = embedding(input_ids).to(device=device)
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            vit_embeds = model.extract_feature(pixel_values).to(device=device)
            selected = (input_ids == self.processor.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.
        return {'inputs_embeds': inputs_embeds}


register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl,
        default_system='You are an AI assistant whose name is InternLM (书生·浦语).',
        template_cls=InternvlTemplate,
        placeholder_tokens=['<IMG_CONTEXT>'],
        auto_add_bos=True))
register_template(
    Phi3TemplateMeta(
        MLLMTemplateType.internvl_phi3,
        default_system='You are an AI assistant whose name is Phi-3.',
        template_cls=InternvlTemplate,
        placeholder_tokens=['<IMG_CONTEXT>'],
        auto_add_bos=True))


class Internvl2Template(InternvlTemplate):
    video_segments = 8

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        image_context = super().replace_tag('image', index, inputs)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            video_segments = get_env_args('video_segments', int, self.video_segments)
            load_video = partial(load_video_internvl, num_segments=video_segments)
            return self.replace_video2image(load_video, inputs, lambda i: [f'Frame{i + 1}: '] + image_context)

    def replace_object(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        objects = inputs.objects
        if objects:
            object_ = objects[index]
            return [f'<ref>{object_["caption"]}</ref>']
        else:
            return ['<ref-object>']

    def replace_box(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        objects = inputs.objects
        if objects:
            object_ = objects[index]
            if isinstance(object_['bbox'][0], list):
                all_objects = '<box> ['
                for sub_object in object_['bbox']:
                    all_objects += (f'[{sub_object[0]}, {sub_object[1]}, ' f'{sub_object[2]}, {sub_object[3]}],')
                all_objects = all_objects[:-1]
                all_objects += '] </box>'
                return [all_objects]
            else:
                return [
                    f'<box> [[{object_["bbox"][0]}, {object_["bbox"][1]}, '
                    f'{object_["bbox"][2]}, {object_["bbox"][3]}]] </box>'
                ]
        else:
            return ['<bbox>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super(InternvlTemplate, self)._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        labels = encoded['labels']
        images = inputs.images
        if images:
            has_video = bool(inputs.videos)
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 12)
            video_max_num = get_env_args('video_max_num', int, 1)
            if has_video:
                max_num = video_max_num
            pixel_values = [transform_image(image, input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(self.config.torch_dtype)
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'
        added_tokens_len = 0
        for idx, num_patch in zip(idx_list, num_patches):
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patch
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                             + 1:]
            added_tokens_len += len(img_tokens) - 1
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['pixel_values'] = pixel_values
        return encoded


_internvl2_system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl2,
        default_system=_internvl2_system,
        template_cls=Internvl2Template,
        placeholder_tokens=['<IMG_CONTEXT>'],
    ))

register_template(
    Phi3TemplateMeta(
        MLLMTemplateType.internvl2_phi3,
        default_system=_internvl2_system,
        template_cls=Internvl2Template,
        placeholder_tokens=['<IMG_CONTEXT>'],
    ))

register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl2_5,
        template_cls=Internvl2Template,
        placeholder_tokens=['<IMG_CONTEXT>'],
        default_system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'))


from dataclasses import asdict
from swift.utils import get_dist_setting
import re
class SophiaTemplate(InternvlTemplate):
    # video_segments = 16
    # system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int, inputs: StdTemplateInputs) -> List[Context]:
        image_context = super().replace_tag('image', index, inputs)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            video_index = inputs.video_idx
            if len(inputs.videos) == 1:
                video = inputs.videos[0]
            else:
                raise NotImplementedError('Not support multiple video inputs.')

            frm_num = get_env_args('nframes', int, self.model.config.frm_num)

            images, len_type = load_video_sophia(
                video, 
                frm_num=frm_num,
                video_name=video,
                use_diff_ways=self.model.config.use_diff_ways,
            )  # load出所有帧 以备_post_encode()中筛选
            inputs.images = images
            inputs.len_type = len_type

            context_list = ['<IMG_CONTEXT>']     # 原本应该形如 ['Frame1: ', '<img>', [-100], '</img>\n', 'Frame2: ', ...] 但这里不构造 用一个<IMG_CONTEXT>代替
            return context_list

    def replace_object(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        objects = inputs.objects
        if objects:
            object_ = objects[index]
            return [f'<ref>{object_["caption"]}</ref>']
        else:
            return ['<ref-object>']

    def replace_box(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        objects = inputs.objects
        if objects:
            object_ = objects[index]
            if isinstance(object_['bbox'][0], list):
                all_objects = '<box> ['
                for sub_object in object_['bbox']:
                    all_objects += (f'[{sub_object[0]}, {sub_object[1]}, ' f'{sub_object[2]}, {sub_object[3]}],')
                all_objects = all_objects[:-1]
                all_objects += '] </box>'
                return [all_objects]
            else:
                return [
                    f'<box> [[{object_["bbox"][0]}, {object_["bbox"][1]}, '
                    f'{object_["bbox"][2]}, {object_["bbox"][3]}]] </box>'
                ]
        else:
            return ['<bbox>']

    def _encode(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = SophiaTemplateInputs(**asdict(inputs), len_type=None)

        encoded = super(InternvlTemplate, self)._encode(inputs) # in this, will enter the replace_tag() to load the video, and set the inputs.len_type
        try:
            input_ids = encoded['input_ids']
        except Exception as e:
            logger.error('KeyError: `input_ids` not found.')
        
        if encoded['labels'] is None: 
            logger.warn(f"The value of `labels` is None (ignore this info if you are inferring rather than training)."
                        f" This data info: Query: {inputs.messages[0].get('content')} " )   # Resp: {inputs.messages[0].get('content')} " )
        
        encoded['_data'] = {'input_ids': torch.tensor(encoded['input_ids']),
                           'labels': encoded['labels'],
                           'images': inputs.images, 
                           'len_type': inputs.len_type, 
                           'videos': inputs.videos}

        _, local_rank, _, local_world_size = get_dist_setting()
        if self.model.config.use_diff_ways:
            if self.is_training and inputs.len_type != 'long': # 这两行仅在三岔路中的模块(mlp_for_shot_select)被训练时使用 用于保证每次计算图一致
                print(f'[rank:{local_rank}] Skip this data sample when training, to ensure the match between computation graphs. Video:' + str(inputs.videos))
                return {}

            pattern = r"<\|im_start\|>\s*user\s*\n\s*(.*?)\s*<\|im_end\|>"
            regex = re.compile(pattern, re.DOTALL)                        # 预编译正则表达式，提高匹配效率
            queries = regex.findall( self.tokenizer.decode(input_ids) )   # 提取所有匹配的内容
            text_query = ' '.join([re.sub(r"<IMG_CONTEXT>", "", t).strip() for t in queries])    # 去除前后空白符 并过滤掉<IMG_CONTEXT>

            encoded['_data']['text_query'] = text_query
        
        ids_len_wo_v_tokens = len(input_ids)
        if not self.model.config.use_diff_ways:     # 本段代码仅在Stage1-3使用
            max_seq_len_w_vis = 9700                # 使用deepspeed时限制最终长度不超过9700 # 自己的八卡只能改为8800
            if self.is_training and ids_len_wo_v_tokens > max_seq_len_w_vis - 60 * (128+3):
                print(f'\n[rank:{local_rank}] Skip this data sample when training, it contains {ids_len_wo_v_tokens} tokens and will exceed {max_seq_len_w_vis} after inserting the visual tokens.')
                print(f'This data: {self.tokenizer.decode(input_ids)}\n')
                return {}   # 至于Stage4变长度训练时该怎么限制 就到时候再写吧

        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to = None) -> Dict[str, Any]:
        assert len(batch) == 1, 'Not support batch size > 1.'

        super_res_batch = Template._data_collator(self, batch, padding_to=padding_to)

        # 父类的_data_collator会把我们在_encode里构造的字段去掉 给它补回来 在_post_encode里要用
        super_res_batch['images'] = batch[0]['_data']['images']
        super_res_batch['len_type'] = batch[0]['_data']['len_type']
        super_res_batch['videos'] = batch[0]['_data']['videos']
        super_res_batch['text_query'] = batch[0]['_data'].get('text_query', None)

        # 父类的_data_collator构造的attention_mask依据的长度有问题 是未加所有图像的长度 在_post_encode里确定序列长度后记得再次构造
        super_res_batch.pop('attention_mask')
        return super_res_batch

    def insert_all_img(self, inputs, nframes):    # 将唯一的一个<IMG_CONTEXT>替换成ori+coaser的视频token
        input_ids = inputs['input_ids']     # (bsz=1, input_token_num)
        labels = inputs.get('labels')
        assert input_ids.shape[0] == 1, 'Not support batch size > 1.'
        input_ids = input_ids[0]
        if labels is not None: labels = labels[0]

        device = input_ids.device
        if isinstance(input_ids, torch.Tensor): input_ids = input_ids.tolist()
        if isinstance(labels, torch.Tensor): labels = labels.tolist()

        vid_token_id = self.tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False) 
        vid_pos = findall(input_ids, vid_token_id) # find that <IMG_CONTEXT>
        assert len(vid_pos) == 1
        vid_pos = vid_pos[0]
        array_num = self.model.config.array_num

        img_sta_token_id = self.tokenizer.encode('<img>', add_special_tokens=False) 
        img_end_token_id = self.tokenizer.encode('</img>', add_special_tokens=False)
        newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False) 

        res = []
        one_coaser_img = img_sta_token_id + \
                            self.tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False) * self.model.config.frm_token_num * 1 + \
                            img_end_token_id + newline_token_id
        res += one_coaser_img * nframes         # 由于改为了让该函数直接插入ori和coaser的视频token 所以先插入ori的
        
        # 下面操作和原来一样
        i, cur_level_num = 0, nframes // array_num
        level_sizes = [nframes]
        for i in range(3):
            res += one_coaser_img * cur_level_num

            level_sizes.append(cur_level_num)
            cur_level_num = cur_level_num // array_num

        input_ids = input_ids[:vid_pos] + res + input_ids[vid_pos + 1:]
        if labels is not None:
            labels = labels[:vid_pos] + [-100] * len(res) + labels[vid_pos + 1:]

        inputs['input_ids'] = torch.tensor([input_ids], device=device)  # (bsz=1, input_token_num)
        if labels is not None:
            inputs['labels'] = torch.tensor([labels], device=device)        # (bsz=1, input_token_num)
        inputs['hierar_mode'] = 'llm_not_use_coasers'
        inputs['level_sizes'] = level_sizes                             # list
        return inputs, len(res)

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_ids, labels, images, len_type = inputs['input_ids'], inputs.get('labels'), inputs['images'], inputs['len_type']
        ori_img_num = len(images)

        embedding = model.get_input_embeddings()
        llm_device = embedding.weight.device
        vit_device = model.vision_model.embeddings.patch_embedding.weight.device

        has_img_vid_token = ori_img_num != 0
        if not has_img_vid_token:
            inputs['hierar_mode'], inputs['level_sizes'] = 'pure_text_input', None

            inputs_embeds = embedding(input_ids[None])[0].to(device=llm_device)
            inputs['inputs_embeds'] = inputs_embeds

            del inputs['images'], inputs['len_type'], inputs['videos']; inputs.pop('text_query', None)
        else:
            has_video = True
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 1 if has_video else 12)

            try:
                pixel_values = [transform_image(image, input_size, max_num) for image in images]
                num_patches = [pv.shape[0] for pv in pixel_values]
                pixel_values = torch.cat(pixel_values, dim=0)              # [ori_img_num, 3, input_size=448, input_size]
                pixel_values = pixel_values.to(self.model.dtype).to(vit_device)
            except Exception as e:
                print(f"Error occurred in _post_encode(): {e}")

            vit_embeds = model.extract_feature(pixel_values, include_tree_conv=False).to(device=vit_device) # [ori_img_num, frm_token_num, vit_dim]

            if not self.model.config.use_diff_ways:
                # Stage1-3 均匀采16帧 压缩到128tokens 不涉及筛帧和帧数变化
                logger.info(f'[Stage1-3] {ori_img_num} frames * {vit_embeds.shape[1]} tokens.')

            else:
                # Stage4 带筛选的训练和推理
                if self.is_training:
                    assert len_type == 'long', f'Are you training mlp_for_shot_select? This will lead to the mismatch between computation graphs.'

                if len_type == 'long':  # 如果是长视频 要执行镜头筛选与镜内筛选
                    text_query = inputs['text_query']
                    text_query_ids = torch.tensor(self.tokenizer.encode(text_query))

                    shot_list, semantic_indices = model.get_shot_list(images)
                    # plot_shot_split(images, shot_list, thumbnail_size=(32, 32), save_path='shot_split.jpg')
                    keep_img_mask = model.filter_shots(vit_embeds, shot_list, semantic_indices, text_query_ids, text_query,
                                                        input_size, max_num, vit_device, model.dtype, self._is_training)

                    assert sum(keep_img_mask) >= 16, f'sum(keep_img_mask) = {sum(keep_img_mask)} < 16.'

                    # 为了挑选case 查看哪些帧被保留
                    # plot_shot_split_keep(images, shot_list, binary_list=keep_img_mask, thumbnail_size=(32, 32), save_path='shot_split_keep.jpg')

                    # 使用 乘以mask 的方法 完成筛选
                    vit_embeds = vit_embeds * keep_img_mask.to(vit_embeds.dtype).unsqueeze(-1).unsqueeze(-1)
                    vit_embeds = vit_embeds[ torch.where(keep_img_mask==1.0)[0] ]   # 这一步索引是没法获得梯度的 但我们本来就不需要这个索引获得梯度
                    
                    logger.info(f'[Filtering] Reducing {ori_img_num} frames to {vit_embeds.shape[0]} frames.')

                    # 至此 我们知道了最后有多少帧要插入input_ids 并且已获得它们的vit_embeds (但还没获得他们coaser的vit_embeds)
                    ori_img_num = vit_embeds.shape[0]
                            
                elif len_type == 'ex_long':    # 如果是超长视频 默认均匀抽64个帧 此时_post_encode传入的images应该是64个
                    assert vit_embeds.shape[0] == 64, f'{vit_embeds.shape[0]} != 64'
                    logger.info(f'[Too Long] Uniform sample {vit_embeds.shape[0]} frames from original {ori_img_num} frames (longer than 640 sec).')

                elif len_type == 'short':      # 如果不是长视频 默认均匀抽self.model.config.frm_num个帧
                    assert vit_embeds.shape[0] == 16, f'{vit_embeds.shape[0]} != 16'
                    logger.info(f'[Short] Uniform sample {vit_embeds.shape[0]} frames from original {ori_img_num} frames (shorter than 320 sec).')

            # 得到coaser的embeds
            all_vit_embeds = model.tree_conv_from_vit_embeds(vit_embeds)
            del vit_embeds

            # 进行input_ids的组装(包括labels)
            input_img_num = all_vit_embeds.shape[0]
            
            # inputs = self.insert_coaser_img(inputs=inputs, nframes=input_img_num)  # 把inputs里 input_ids中视频位置替换为很多个<IMG_CONTEXT> labels也相应替换
            inputs, v_token_len = self.insert_all_img(inputs=inputs, nframes=ori_img_num)  # 把inputs里 input_ids中视频位置替换为很多个<IMG_CONTEXT> labels也相应替换


            input_ids = inputs['input_ids']     # (bsz=1, input_token_num_with_all_imgs)
            labels = inputs.get('labels')

            # 把all_vit_embeds放入input_embeds中
            inputs_embeds = embedding(input_ids[None])[0].to(device=llm_device)

            img_context_id = self.tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False)[0]
            selected = (input_ids == img_context_id)
            inputs_embeds[selected] = all_vit_embeds.reshape(-1, all_vit_embeds.shape[-1]).to(inputs_embeds.device)
            inputs['inputs_embeds'] = inputs_embeds

            del inputs['images'], inputs['len_type'], inputs['videos']; inputs.pop('text_query', None)

            _, local_rank, _, local_world_size = get_dist_setting()
            print(f'[rank:{local_rank}] len(input_ids) = {input_ids.shape[1]}, in which {v_token_len} are visual tokens.')
            # gc.collect()
            # torch.cuda.empty_cache()

        # 至此终于确定了序列长度 并构造好了inputs['input_ids'], inputs['labels']
        # 在_data_collator中我们没有构造inputs['attention_mask'] (注意: bsz>1时多个样本为了对齐要填充padding 它其实只是标记填充的mask 而不是causal或trsfm_att_mask)
        _inputs = [{
            'input_ids': input_ids[0], 'labels': labels[0] if labels is not None else None,
        }]
        _res = Template._data_collator(self, _inputs, padding_to=None)
        inputs['attention_mask'] = _res['attention_mask']



        ### 构造mask_info

        frm_token_num = self.model.config.frm_token_num

        # 构造vis_sta, vis_end, coaser_sta, coaser_end 
        if inputs['hierar_mode'] == 'pure_text_input':
            inputs['vis_sta'], inputs['vis_end'], inputs['coaser_sta'], inputs['coaser_end'] = -1, -1, -1, -1
            inputs['trsfm_att_mask'] = None
            
        elif inputs['hierar_mode'] == 'llm_not_use_coasers':
            img_sta_token_id = self.tokenizer.encode('<img>', add_special_tokens=False)
            img_end_token_id = self.tokenizer.encode('</img>', add_special_tokens=False)

            inputs['vis_sta'], inputs['vis_end'], inputs['coaser_sta'], inputs['coaser_end'] = [], [], [], []
            att_mask = [
                torch.ones((len(input_ids[i]), len(input_ids[i])), dtype=torch.int64)
                for i in range( len(input_ids) )
            ]
            for i in range(input_ids.shape[0]):
                v_sta = torch.tensor(findall(input_ids[i].tolist(), img_sta_token_id))
                v_end = torch.tensor(findall(input_ids[i].tolist(), img_end_token_id)) + 1   # 闭区间
                assert all(v_end - v_sta + 1 == frm_token_num + 3)

                coaser_img_num = sum(inputs['level_sizes'][1:])
                vis_sta,    vis_end    = v_sta[ : ori_img_num], v_end[ : ori_img_num]
                coaser_sta, coaser_end = v_sta[-coaser_img_num : ], v_end[-coaser_img_num : ]  # 闭区间

                first_coaser_sta, last_coaser_end = coaser_sta[0], coaser_end[-1]

                att_mask[i][first_coaser_sta : last_coaser_end + 1, :] = 0
                att_mask[i][:, first_coaser_sta : last_coaser_end + 1] = 0
                ### set the blocks about videos
                # first make all the video-video blocks as 0
                v_sta = torch.cat([ coaser_sta, vis_sta ])
                v_end = torch.cat([ coaser_end, vis_end ])
                for x in range(len(v_sta)):
                    for y in range(len(v_sta)):
                        att_mask[i][v_sta[x]: v_end[x] + 1, v_sta[y]: v_end[y] + 1] = 0
                # then set the valid video-video blocks as 1 according to the ref
                array_num = self.model.config.array_num
                ref_mask, level_sizes = get_hierar_mask(bottom_size=vis_sta.shape[0], 
                                                        array_sizes=[array_num, array_num, array_num],
                                                        neibor_size=model.config.neibor_size, 
                                                        device='cpu', put_coaser_ahead=True)
                all_x, all_y = torch.where(ref_mask==1)
                for x, y in zip(all_x, all_y):
                    att_mask[i][ v_sta[x] : v_end[x] + 1, v_sta[y] : v_end[y] + 1 ] = 1
                # 至此 att_mask即为batch内第b个样本的trsfm_att_mask
            
            inputs['trsfm_att_mask'] = torch.stack(att_mask, dim=0)

            inputs['vis_sta'] = vis_sta.unsqueeze(0)
            inputs['vis_end'] = vis_end.unsqueeze(0)
            inputs['coaser_sta'] = coaser_sta.unsqueeze(0)
            inputs['coaser_end'] = coaser_end.unsqueeze(0)

        inputs['use_triton'] = self.model.config.use_triton
        if inputs['use_triton']:
            assert not self.is_training, 'The triton kernel is only available in inferring.'
            
            if inputs['hierar_mode'] == 'pure_text_input':
                logger.warn(f'For pure text input, the triton kernel is useless, setting use_triton to False this time...')
                inputs['use_triton'] = False
        
        return inputs 


sophia_system = '你是一个有用无害的人工智能助手，能对视频输入进行理解。'
register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.sophia,
        default_system=sophia_system,
        template_cls=SophiaTemplate,
        placeholder_tokens=['<IMG_CONTEXT>'],
    ))