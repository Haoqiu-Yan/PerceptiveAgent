"""
task for audio to captioning
---
Author: haoqiu yan
"""


import json
import os

import torch.distributed as dist
import numpy as np

from bubogpt.common.dist_utils import main_process
from bubogpt.common.registry import registry
from bubogpt.tasks.base_task import BaseTask

from bubogpt.common.logger import MetricLogger
from bubogpt.datasets.data_utils import prepare_sample
from bubogpt.common.dist_utils import is_dist_avail_and_initialized


@registry.register_task("audio_to_captioning")
class AudioCaptionTask(BaseTask):
    def __init__(self, cfg, report_metric=True):
        super().__init__()

        self.run_cfg = cfg

        self.evaluate = self.run_cfg.evaluate
        self.max_len = self.run_cfg.max_len
        self.min_len = self.run_cfg.min_len

        self.num_beams = self.run_cfg.get("num_beam", 5)
        self.no_repeat_ngram_size = self.run_cfg.get("no_repeat_ngram_size", 2)
        self.top_k = self.run_cfg.get("top_k", 10)
        self.top_p = self.run_cfg.get("top_p", 0.92)

        if self.run_cfg.get("use_topk", False):
            self.use_topk = True
            self.use_topk_nucleus = False
            self.use_beam = False

            
        elif self.run_cfg.get("use_topk_nucleus", False):
            self.use_topk_nucleus = True  
            self.use_topk = False
            self.use_beam = False         

        elif self.run_cfg.get("use_beam", True):
            self.use_beam = True
            self.use_topk_nucleus = False  
            self.use_topk = False
           

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            cfg=run_cfg,
            report_metric=report_metric,
        )


    def validation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Validation: model loss"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
        # for samples in metric_logger.log_every(range(50), print_freq, header):

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            val_losses = self.valid_step(model=model, samples=samples)
            results.extend(val_losses)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def valid_step(self, model, samples):
        loss = model(samples)["loss"].item()

        return [loss]

    def after_validation(self, val_result):
        avg_loss = np.mean(val_result)
        metrics = {"agg_metrics": avg_loss}
        
        return metrics

    def testation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "test: predicted bleu"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
        # for samples in metric_logger.log_every(range(50), print_freq, header):
            # samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            test_output = self.test_step(model=model, samples=samples)
            results.extend(test_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results
    
    def test_step(self, model, samples):
        results = []

        captions = model.generate(
            samples,
            max_length=self.max_len,
            min_length=self.min_len,
            use_beam = self.use_beam,
            num_beams=self.num_beams,
            no_repeat_ngram_size=self.no_repeat_ngram_size, 
            use_topk=self.use_topk,
            top_k=self.top_k,
            use_topk_nucleus=self.use_topk_nucleus,
            top_p=self.top_p,
        )

        # tensor of audio
        audio_ids = samples["id"]
        real_captions = samples["text_input"]
        for caption, audio_id, real_caption in zip(captions, audio_ids, real_captions):
            results.append({"audio path": audio_id, "pred caption": caption, "real caption":real_caption})

        return results

    def after_testation(self, val_result, split_name, epoch, **kwargs):
        """save predicted captions, and compute BLEU
        ---
        BLEU: sacrebleu
        report_metric is set in config.yaml
        save_result: `{"audio path": , "pred caption": , "real caption": }`
        save_result_sclite: `SENTENCE (audio path)`, line by line
                            for sclite format
        """
        # eval_result_file = self.save_result(
        #     result=val_result,
        #     result_dir=registry.get_path("result_dir"),
        #     filename="{}_epoch{}".format(split_name, epoch),
        #     remove_duplicate="audio path",
        # )        
        # assert os.environ["RANK"]==0, "fun save_result_sclite() does not support \
        #       multiple core saving, unlike fun save_result()"
        eval_result_file = self.save_result_sclite(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="audio path",
        )

        if self.report_metric:
            # metrics = self._report_metrics(
            #     eval_result_file=eval_result_file, split_name=split_name
            # )
            metrics = self._report_metrics_bleu(result=val_result)
        else:
            metrics = {"score": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res


    @staticmethod
    def save_result_sclite(result, result_dir, filename, remove_duplicate=""):
        ids = [sample["audio path"] for sample in result]
        preds = [sample["pred caption"] for sample in result]
        gts = [sample["real caption"] for sample in result]
        
        ids_file = os.path.join(result_dir, f"ids_{filename}.txt")
        preds_file = os.path.join(result_dir, f"hypo_{filename}.txt")
        gts_file = os.path.join(result_dir, f"gt_{filename}.txt")

        with open(ids_file, "w") as idf:
            for id, sample in enumerate(ids):
                to_write = "{} (None-{})\n".format(sample, id)
                idf.write(to_write)
        with open(preds_file, "w") as pf:
            for id, sample in enumerate(preds):
                to_write = "{} (None-{})\n".format(sample, id)
                pf.write(to_write)
        with open(gts_file, "w") as gf:
            for id, sample in enumerate(gts):
                to_write = "{} (None-{})\n".format(sample, id)
                gf.write(to_write)
        
        print("result file [sclite] saved to {} && {} && {}".format(ids_file, preds_file, gts_file))
        
        return preds_file

    @main_process
    def _report_metrics_bleu(self, result):
        """
        sacrebleu from huggingface.datasets
        ---
        result: [{"audio path": , "pred caption": , "real caption": },...]
        score: {'score': bleu, 'counts':  , 'totals':  , 'precisions':  , 'bp': 1.0, 'sys_len': 10, 'ref_len': 7}
        """

        import evaluate

        preds = [sample["pred caption"] for sample in result]
        gts = [[sample["real caption"]] for sample in result]

        metric = evaluate.load('sacrebleu')
        score = metric.compute(predictions=preds, references=gts)

        return score


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval