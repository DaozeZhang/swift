# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from datasets import Dataset as HfDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.versions import require_version

from swift.llm import (Template, TemplateType, dataset_map, get_dataset, get_dataset_from_repo,
                       get_model_tokenizer, get_template, print_example, register_dataset, register_model,
                       register_template)
from swift.llm.utils.model import _use_submodel_func, _clone_hook
from swift.llm.utils.template import Internvl2Template
from swift.utils import get_logger

logger = get_logger()




class CustomTemplateType:
    hierar_internvl2 = 'hierar_internvl2'

register_template(CustomTemplateType.hierar_internvl2, Internvl2Template(), use_model=True, lazy_tokenize=True)



class CustomModelType:
    hierar_internvl2 = 'hierar_internvl2'

@register_model(
    CustomModelType.hierar_internvl2,
    '/mnt/nas1/daoze/code/hierar_internvl2/InternVL2-8B',   # load 8B ckpt as init
    # LoRATM.hierar_internvl2,
    CustomTemplateType.hierar_internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-8B')
def get_model_tokenizer_hierar_internvl2(model_dir: str,
                                         torch_dtype: torch.dtype,
                                         model_kwargs: Dict[str, Any],
                                         load_model: bool = True,
                                         **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    if kwargs.get('eos_token') is None and tokenizer.eos_token != '<|im_end|>':
        try:
            del tokenizer.__class__.eos_token_id
        except AttributeError:
            pass
        tokenizer.eos_token = '<|im_end|>'

    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if hasattr(model_config.llm_config, 'attn_implementation'):
        attr = 'attn_implementation'
    else:
        attr = '_attn_implementation'
    if use_flash_attn:
        setattr(model_config.llm_config, attr, 'flash_attention_2')
    else:
        setattr(model_config.llm_config, attr, 'eager')
        setattr(model_config.llm_config, f'{attr}_internal', None)

    # model_quant_config = getattr(model_config, 'quantization_config', None)
    #
    # use_bnb = False
    # if model_quant_config is not None:
    #     use_bnb = model_quant_config.get('quant_method', None) == 'bitsandbytes'
    # quantization_config = model_kwargs.get('quantization_config', None)
    # if isinstance(quantization_config, BitsAndBytesConfig):
    #     use_bnb = True

    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, model_config=model_config, **kwargs)

    # if use_bnb and kwargs.get('is_training'):
    #     # patch: bnb backward shape mismatch bug
    #     if model is not None and model.language_model is not None:
    #         model.language_model.output.state.force_no_igemmlt = True

    if model is not None:
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        _use_submodel_func(model, 'language_model', func_list)
        embedding = model.language_model.get_input_embeddings()
        embedding.register_forward_hook(_clone_hook)

    return model, tokenizer



# def _preprocess_stsb(dataset: HfDataset) -> HfDataset:
#     prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
# Sentence 1: {text1}
# Sentence 2: {text2}
# Similarity score: """
#     query = []
#     response = []
#     for d in dataset:
#         query.append(prompt.format(text1=d['text1'], text2=d['text2']))
#         response.append(f"{d['label']:.1f}")
#     return HfDataset.from_dict({'query': query, 'response': response})

# class CustomDatasetName:
#     stsb_en = 'stsb-en'

# register_dataset(CustomDatasetName.stsb_en, 'swift/stsb', None, _preprocess_stsb, get_dataset_from_repo)


if __name__ == '__main__':
    # The Shell script can view `examples/pytorch/llm/scripts/custom`.
    # test dataset
    train_dataset, val_dataset = get_dataset(['video-chatgpt'], check_dataset_strategy='warning')
    print(f'train_dataset: {train_dataset}')
    print(f'val_dataset: {val_dataset}')
    # test model base
    model, tokenizer = get_model_tokenizer(CustomModelType.hierar_internvl2, use_flash_attn=False)
    # # test model chat
    # model, tokenizer = get_model_tokenizer(CustomModelType.tigerbot_13b_chat, use_flash_attn=False)
    # test template
    template = get_template(CustomTemplateType.hierar_internvl2, tokenizer)
    train_dataset = dataset_map(train_dataset, template.encode)
    print_example(train_dataset[0], tokenizer)