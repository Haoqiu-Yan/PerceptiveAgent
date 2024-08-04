"""
record valid loss, on the basis of RunnerBase.
---

"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from bubogpt.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from bubogpt.common.registry import registry
from bubogpt.common.utils import is_url
from bubogpt.datasets.data_utils import concat_datasets, reorg_datasets_by_split, WrappedChainDataset
from bubogpt.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


from bubogpt.runners import RunnerBase


@registry.register_runner("runner_val")
class RunnerVal(RunnerBase):
    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)
        self.save_ckpt_interval = self.config.run_cfg.get("save_ckpt_interval", 1)

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, WrappedChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        def regroup_by_data_type(dataset):
            if not isinstance(dataset, (tuple, list)):
                return [dataset]

            dtypes = set([d.data_type for d in dataset])
            type2data = {}
            for dtype in dtypes:
                type2data[dtype] = [d for d in dataset if d.data_type == dtype]

            return list(type2data.values()),  dtypes

        def get_data_type_ratio(datasets):
            ratios = []
            for type_dataests in datasets:
                type_ratio = None
                for dataset in type_dataests:
                    if hasattr(dataset, 'dtype_ratio') and dataset.dtype_ratio is not None:
                        type_ratio = dataset.dtype_ratio
                ratios.append(type_ratio)

            if any([x is None for x in ratios]):
                ratios = []
            else:
                return ratios

            # Use sample ratio as the data_type ratio
            for type_datasets in datasets:
                ratios.append(sum([d.sample_ratio for d in type_datasets]))
            return ratios

        loaders = []
        for mix_dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            mix_dataset, dtypes = regroup_by_data_type(mix_dataset)
            mix_loader = []
            for dataset in mix_dataset:
                if isinstance(dataset, list) or isinstance(dataset, tuple):
                    # for train: load dataset as a infinite iterable dataloader
                    # for val or test: load dataset as finite, to travel all samples
                    if is_train:
                        dataset_ratios = None
                        if hasattr(dataset[0], 'sample_ratio'):
                            dataset_ratios = [d.sample_ratio for d in dataset]
                        loader = MultiIterLoader(
                            loaders=[
                                _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                                for i, d in enumerate(dataset)
                            ],
                            ratios=dataset_ratios,
                        )
                    else:
                        # dataset is a list with one element
                        val_dataset = dataset[0]
                        val_collate_fn = collate_fn[0]
                        loader = _create_loader(val_dataset, num_workers, bsz, is_train, val_collate_fn)
                else:
                    loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)
                mix_loader.append(loader)
            print(f"There are {len(mix_dataset)} of data types, They are:", dtypes)
            if len(mix_loader) == 1:
                loaders.append(mix_loader[0])
            else:
                loader_ratios = get_data_type_ratio(mix_dataset)
                print("Data type ratios are: ", loader_ratios)
                merged_loader = MultiIterLoader(loaders=mix_loader, ratios=loader_ratios)
                loaders.append(merged_loader)

        return loaders

    def train(self):
        start_time = time.time()
        best_agg_metric = 10000
        best_epoch = -1

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    # val_log = self.eval_epoch(
                    #     split_name=split_name, cur_epoch=cur_epoch
                    # )
                    val_log = self.val_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            # if agg_metrics > best_agg_metric and split_name == "val":
                            # loss 越小越好
                            if agg_metrics < best_agg_metric and split_name == "val":
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics

                                self._save_checkpoint(cur_epoch, is_best=True)

                            val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)
                            logging.info("Validation loss: {}".format(agg_metrics))

            # save ckpt per interval
            if not self.evaluate_only and cur_epoch % self.save_ckpt_interval == 0:
                self._save_checkpoint(cur_epoch, is_best=False)
            # else:
            #     # if no validation split is provided, we just save the checkpoint at the end of each epoch.
            #     if not self.evaluate_only:
            #         self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

            if self.config.run_cfg.distributed:
                dist.barrier()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    
    @torch.no_grad()
    def val_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        given a batch of samples, return loss that model generates.
        """
        
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        # function with pass
        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.validation(model, data_loader)

        if results is not None:
            return self.task.after_validation(
                val_result=results
            )
        
    def evaluate(self, cur_epoch="best", skip_reload=False):
        logging.info("Start evaluating")
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.test_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )
                self.log_stats(test_logs[split_name], split_name)
                logging.info("BLEU on {}: {}".format(split_name, test_logs[split_name]["score"]))

            return test_logs
        
    @torch.no_grad()
    def test_epoch(self, split_name, cur_epoch, skip_reload=False):
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)
    
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        # function with pass
        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.testation(model, data_loader)

        if results is not None:
            return self.task.after_testation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )
        
    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """In parent class, eval_epoch() is used for both val and test, 
            but in this class, it is splitted as val_epoch() and test_epoch()."""
        
        raise NotImplementedError
