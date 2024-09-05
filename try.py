from evalscope.third_party.longbench_write import run_task

task_cfg = dict(stage=['infer', 'eval_l'],
                model_id_or_path='/mnt/nas1/daoze/code/swift/output/qwen2-7b/v16-20240823-233737/checkpoint-1500',
                output_dir='./outputs',
                infer_generation_kwargs={
                    'max_new_tokens': 32768,
                    'temperature': 0.5
                },
                proc_num=8)

run_task(task_cfg=task_cfg)
