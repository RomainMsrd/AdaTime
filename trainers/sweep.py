import sys

sys.path.append('../')
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

from configs.sweep_params import sweep_alg_hparams,sweep_train_hparams2
from utils import fix_randomness, starting_logs, DictAsObject, create_logger, starting_task
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter

from trainers.abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # sweep parameters
        self.num_sweeps = args.num_sweeps
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method + '_' + self.backbone,
            'parameters': {**sweep_alg_hparams[self.da_method], **sweep_train_hparams2}
        }
        self.sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)
        """self.sweep_id = sweep_id
        if self.sweep_id is None:
            self.sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)"""
        wandb.agent(self.sweep_id, self.train, count=sweep_runs_count)


    def train(self):
        run = wandb.init(config=self.hparams)
        self.hparams = dict(wandb.config)
        print(self.hparams)
        # create tables for results and risks
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        #table_results = wandb.Table(columns=columns, allow_mixed_types=True)
        risks_columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]
        #table_risks = wandb.Table(columns=columns, allow_mixed_types=True)

        # table with metrics
        table_results = pd.DataFrame(columns=results_columns)
        # table with risks
        table_risks = pd.DataFrame(columns=risks_columns)

        #self.logger = create_logger(self.exp_log_dir)

        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # set random seed and create logger
                fix_randomness(run_id)
                self.logger, self.scenario_log_dir = starting_logs( self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id)
                #self.scenario_log_dir = starting_task(self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id, self.logger)

                # average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # load data and train model
                self.load_data(src_id, trg_id)

                # initiate the domain adaptation algorithm
                self.initialize_algorithm()

                # Train the domain adaptation algorithm
                self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)

                # calculate metrics and risks
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)
                #table_results.add_data(scenario, run_id, *metrics)
                #table_risks.add_data(scenario, run_id, *risks)

        # calculate overall metrics and risks
        table_results = self.average_run_rusults(table_results)
        table_risks = self.average_run_rusults(table_risks)
        # Save tables to file if needed
        self.save_sweep_tables_to_file(table_results, self.sweep_id, run.name, 'results')
        self.save_sweep_tables_to_file(table_risks, self.sweep_id, run.name, 'risks')

        #print(table_results)
        table_results = wandb.Table(dataframe=table_results)
        #print(table_risks)
        table_risks = wandb.Table(dataframe=table_risks)

        total_results, summary_metrics = self.calculate_avg_std_wandb_table(table_results)
        total_risks, summary_risks = self.calculate_avg_std_wandb_table(table_risks)

        # log results to WandB
        self.wandb_logging(total_results, total_risks, summary_metrics, summary_risks)

        '''for artifact in run.logged_artifacts():
            artifact.delete()'''
        # finish the run
        run.finish()

