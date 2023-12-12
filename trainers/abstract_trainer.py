import copy
import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, few_shot_data_generator, get_label_encoder
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs, DictAsObject,AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.experiment_description = args.dataset
        self.run_description = f"{args.da_method}_{args.exp_name}"

        print(args)
        # paths
        self.home_path = os.getcwd() #os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.uniDA = args.uniDA
        print("Universal : ", self.uniDA)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        # metrics
        self.num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.BinACC = Accuracy(task="binary")
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)

        # metrics

    def sweep(self):
        # sweep configurations
        pass

    def initialize_algorithm(self):
        # get algorithm class
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        return last_model, best_model

    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # Training the model
        self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)
        return self.last_model, self.best_model

    def evaluate(self, test_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                if self.uniDA:
                    m = torch.isin(labels, self.trg_private_class.view((-1)).long().to(self.device), invert=True)
                    loss = F.cross_entropy(predictions[m], labels[m])
                else:
                    loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.detach().cpu().item())
                predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def get_trg_private(self, src_loader, trg_loader):
        trg_y = copy.deepcopy(trg_loader.dataset.y_data)
        src_y = src_loader.dataset.y_data
        pri_c = torch.Tensor(np.setdiff1d(trg_y, src_y))
        return pri_c

    def H_score(self, trg_pred, trg_y):
        class_c = np.where(trg_y != -1)
        class_p = np.where(trg_y == -1)

        label_c, pred_c = trg_y[class_c], trg_pred[class_c]
        label_p, pred_p = trg_y[class_p], trg_pred[class_p]
        acc_c = self.ACC(pred_c.argmax(dim=1), label_c)

        pred_p = self.algorithm.decision_function(pred_p)
        acc_p = self.ACC(pred_p, label_p)
        print("Trg Private Acc : ", acc_p.item())
        if acc_c == 0 or acc_p == 0:
            H = torch.Tensor([0])
        else:
            H = 2 * acc_c * acc_p / (acc_p + acc_c)
        return H

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def init_metrics(self):
        self.num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.BinACC = Accuracy(task="binary")
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)

    def load_data(self, src_id, trg_id):
        encoder = get_label_encoder(self.data_path, src_id, self.dataset_configs, self.hparams, "train") if self.uniDA else None
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, encoder, "train", self.dataset_configs.src_balanced)
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, encoder,"test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, encoder, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, encoder, "test")

        self.few_shot_dl_5 = few_shot_data_generator(self.trg_test_dl, self.dataset_configs, encoder,
                                                     5)  # set 5 to other value if you want other k-shot FST
        self.init_metrics()
        if self.uniDA:
            self.trg_private_class = self.get_trg_private(self.src_train_dl, self.trg_train_dl)
            self.init_metrics()

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        risks = src_risk, fst_risk, trg_risk
        metrics = acc, f1, auroc
        if self.uniDA:
            mask = np.isin(self.full_preds.cpu(), self.trg_private_class)
            self.full_labels[mask] = -1
            H_score = self.H_score(self.full_preds.cpu(), self.full_labels.cpu())
            metrics = acc, f1, auroc, H_score
        # f1_sk learn
        # f1 = f1_score(self.full_preds.argmax(dim=1).cpu().numpy(), self.full_labels.cpu().numpy(), average='macro')
        return risks, metrics

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))

    def save_sweep_tables_to_file(self, table_results, sweep_id, run_name, name):
        # save to file if needed
        path = os.path.join(self.exp_log_dir, sweep_id, run_name)
        if not os.path.exists(path):
            os.makedirs(path)
        table_results.to_csv(os.path.join(path, f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def average_run_rusults(self, df):
        cols = df.columns[2:]
        df_mean = df.groupby("scenario")[cols].mean().astype(float)
        df_std = df.groupby("scenario")[cols].std()
        df_std = df_std.rename(columns={f: f + "_std" for f in df_std.columns})
        df = pd.concat([df_mean, df_std], axis=1, join="inner").reset_index()
        #print(df.dtypes)
        #df = df.columns[2:].astype(float)
        return df

    def calculate_avg_std_wandb_table(self, results):

        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[1:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[1:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[1:]}

        '''results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)'''

        results.add_data('mean', *avg_metrics)
        results.add_data('std', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):

        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}


        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks 
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']), allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self):
        self.evaluate(self.trg_test_dl)

        if self.uniDA:
            print("in")
            mask = np.isin(self.full_labels.cpu(), self.trg_private_class, invert=True)
            m = torch.isin(self.full_labels, self.trg_private_class.view((-1)).long().to(self.device),
                           invert=True).cpu()

            # accuracy
            acc = self.ACC(self.full_preds[mask].argmax(dim=1).cpu(), self.full_labels[mask].cpu()).item()
            # f1
            f1 = self.F1(self.full_preds[mask].argmax(dim=1).cpu(), self.full_labels[mask].cpu()).item()
            # auroc
            auroc = self.AUROC(self.full_preds[mask].cpu(), self.full_labels[mask].cpu()).item()

            self.full_labels[~mask] = -1
            H_score = self.H_score(self.full_preds.cpu(), self.full_labels.cpu()).item()
            print("H_score : ", H_score)
            return acc, f1, auroc, H_score

        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        return acc, f1, auroc


    def calculate_risks(self):
         # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, fst_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table

    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.map(format_func)

        return table