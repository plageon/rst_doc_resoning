import math
import os
from typing import Optional, List, Dict

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import Trainer
import numpy as np
from graph_reasoning import Config
from main import logger


class ReasoningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()
        for step, inputs in enumerate(eval_dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False,
                                                        ignore_keys=["loss"])
        _preds = p.predictions
        print(f'pred: {_preds}')
        _preds[_preds != _preds] = 0
        dataset_dir = 'ReclorDataset'

        preds = [x.tolist().index(max(x.tolist())) for x in p.predictions]
        torch.cuda.empty_cache()
        if Config.eval_test:
            dev_len = 500 if dataset_dir == 'ReclorDataset' else (len(preds) // 2)
            dev_acc = accuracy_score(p.label_ids[:dev_len], preds[:dev_len])

            result = {"acc_dev": dev_acc, }

            if not dataset_dir == 'ReclorDataset':
                result["acc_test"] = accuracy_score(p.label_ids[dev_len:], preds[dev_len:])
                _p = [[math.e ** xx for xx in x] for x in _preds[:len(preds) // 2]]
                dev_loss = sum([-math.log(x[p.label_ids[index]] / sum(x)) for index, x in enumerate(_p)]) / len(_p)
                result['dev_loss_dev'] = dev_loss
                _p = [[math.e ** xx for xx in x] for x in _preds[len(preds) // 2:]]
                dev_loss = sum([-math.log(x[p.label_ids[index]] / sum(x)) for index, x in enumerate(_p)]) / len(_p)
                result['dev_loss_test'] = dev_loss
            else:
                _p = [[math.e ** xx for xx in x] for x in _preds[:500]]
                dev_loss = sum([-math.log(x[p.label_ids[index]] / sum(x)) for index, x in enumerate(_p)]) / len(_p)
                result['dev_loss2'] = dev_loss
                assert sum(p.label_ids[dev_len:]) == 0, f'{p.label_ids[dev_len:]}'

            if not os.path.exists(os.path.join('results', training_args.run_name)):
                logger.info(f'mkdir {os.path.join("results", training_args.run_name)}')
                os.mkdir(os.path.join('results', training_args.run_name))

            count = 0
            save_path = f'test_{dev_acc}_{count}.npy'
            while os.path.exists(os.path.join('results', training_args.run_name, save_path)):
                count += 1
                save_path = f'test_{dev_acc}_{count}.npy'
            np.save(os.path.join('results', training_args.run_name, save_path), preds[dev_len:])

            return result
        else:
            return {"acc": accuracy_score(p.label_ids, preds)}