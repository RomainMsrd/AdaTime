import copy

import ot
import torch
import torch.nn as nn
import numpy as np
import itertools

from sklearn.cluster import SpectralClustering

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, CNN_ATTN
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss, NTXentLoss, SupConLoss, \
    SliceWassersteinDiscrepancy
from utils import EMA
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F
from scipy.spatial.distance import cdist

from scipy.spatial import distance



def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.is_uniDA = False


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def correct_predictions(self, preds):
        return preds

    def decision_function(self, preds):
        return preds
    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError


class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:

            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class TARGET_ONLY(Algorithm):
    """
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        for trg_x, trg_y in trg_loader:

            trg_x, trg_y = trg_x.to(self.device), trg_y.to(self.device)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            trg_cls_loss = self.cross_entropy(trg_pred, trg_y)

            loss = trg_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Trg_cls_loss': trg_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # correlation alignment loss
        self.coral = CORAL()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # add if statement

        if len(src_loader) > len(trg_loader):
            joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        else:
            joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'coral_loss': coral_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)
            mmd_loss = self.mmd(src_feat, trg_feat)
            cond_ent_loss = self.cond_ent(trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["mmd_wt"] * mmd_loss + \
                self.hparams["cond_ent_wt"] * cond_ent_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                    'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        #self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment Losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            # source features and predictions
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)
            pred_concat = torch.cat((src_pred, trg_pred), dim=0)

            # Domain classification loss
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            # loss of domain discriminator according to fake labels

            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
        #self.lr_scheduler.step()

class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Aligment losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)
        self.ema = EMA(0.998)
        self.ema.register(self.network)

        # Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)

            # Domain classification loss
            disc_prediction = self.domain_classifier(feat_concat.detach())
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            disc_prediction = self.domain_classifier(feat_concat)

            # loss of domain discriminator according to fake labels
            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # Virual advariarial training loss
            loss_src_vat = self.vat_loss(src_x, src_pred)
            loss_trg_vat = self.vat_loss(trg_x, trg_pred)
            total_vat = loss_src_vat + loss_trg_vat
            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

            # update exponential moving average
            self.ema(self.network)

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Alignment losses
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)        # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate lmmd loss
            domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # aligment losses
        self.coral = CORAL()
        self.HoMM_loss = HoMM_loss()

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate lmmd loss
            domain_loss = self.HoMM_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate mmd loss
            domain_loss = self.mmd_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Domain classifier
        self.domain_classifier = Discriminator(configs)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class AdvSKM(Algorithm):
    """
    AdvSKM: https://www.ijcai.org/proceedings/2021/0378.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)
        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
            target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
            mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss.requires_grad = True

            # update discriminator
            self.optimizer_disc.zero_grad()
            mmd_loss.backward()
            self.optimizer_disc.step()

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # domain loss.
            source_embedding_disc = self.AdvSKM_embedder(src_feat)
            target_embedding_disc = self.AdvSKM_embedder(trg_feat)

            mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss_adv.requires_grad = True

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            # update optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': mmd_loss_adv.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                    avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class SASA(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # feature_length for classifier
        configs.features_len = 1
        self.classifier = classifier(configs)
        # feature length for feature extractor
        configs.features_len = 1
        self.feature_extractor = CNN_ATTN(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # Extract features
            src_feature = self.feature_extractor(src_x)
            tgt_feature = self.feature_extractor(trg_x)

            # source classification loss
            y_pred = self.classifier(src_feature)
            src_cls_loss = self.cross_entropy(y_pred, src_y)

            # MMD loss
            domain_loss_intra = self.mmd_loss(src_struct=src_feature,
                                            tgt_struct=tgt_feature, weight=self.hparams['domain_loss_wt'])

            # total loss
            total_loss = self.hparams['src_cls_loss_wt'] * src_cls_loss + domain_loss_intra

            # remove old gradients
            self.optimizer.zero_grad()
            # calculate gradients
            total_loss.backward()
            # update the weights
            self.optimizer.step()

            losses =  {'Total_loss': total_loss.item(), 'MMD_loss': domain_loss_intra.item(),
                    'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = torch.mean(src_struct - tgt_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value


class CoTMix(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

         # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)
        self.entropy_loss = ConditionalEntropyLoss()
        self.sup_contrastive_loss = SupConLoss(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # ====== Temporal Mixup =====================
            src_dominant, trg_dominant = self.temporal_mixup(src_x, trg_x)

            # ====== Source =====================
            self.optimizer.zero_grad()

            # Src original features
            src_orig_feat = self.feature_extractor(src_x)
            src_orig_logits = self.classifier(src_orig_feat)

            # Target original features
            trg_orig_feat = self.feature_extractor(trg_x)
            trg_orig_logits = self.classifier(trg_orig_feat)

            # -----------  The two main losses
            # Cross-Entropy loss
            src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
            loss = src_cls_loss * round(self.hparams["src_cls_weight"], 2)

            # Target Entropy loss
            trg_entropy_loss = self.entropy_loss(trg_orig_logits)
            loss += trg_entropy_loss * round(self.hparams["trg_entropy_weight"], 2)

            # -----------  Auxiliary losses
            # Extract source-dominant mixup features.
            src_dominant_feat = self.feature_extractor(src_dominant)
            src_dominant_logits = self.classifier(src_dominant_feat)

            # supervised contrastive loss on source domain side
            src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
            src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
            loss += src_supcon_loss * round(self.hparams["src_supCon_weight"], 2)

            # Extract target-dominant mixup features.
            trg_dominant_feat = self.feature_extractor(trg_dominant)
            trg_dominant_logits = self.classifier(trg_dominant_feat)

            # Unsupervised contrastive loss on target domain side
            trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
            loss += trg_con_loss * round(self.hparams["trg_cont_weight"], 2)

            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(),
                    'src_cls_loss': src_cls_loss.item(),
                    'trg_entropy_loss': trg_entropy_loss.item(),
                    'src_supcon_loss': src_supcon_loss.item(),
                    'trg_con_loss': trg_con_loss.item()
                    }
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

    def temporal_mixup(self,src_x, trg_x):

        mix_ratio = round(self.hparams["mix_ratio"], 2)
        temporal_shift = self.hparams["temporal_shift"]
        h = temporal_shift // 2  # half

        src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

        trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)

        return src_dominant, trg_dominant



# Untied Approaches: (MCD)
class MCD(Algorithm):
    """
    Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
    MCD: https://arxiv.org/pdf/1712.02560.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.classifier2 = classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)


        # optimizer and scheduler
        self.optimizer_fe = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
                # optimizer and scheduler
        self.optimizer_c1 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
                # optimizer and scheduler
        self.optimizer_c2 = torch.optim.Adam(
            self.classifier2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        #self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        #self.lr_scheduler_c1 = StepLR(self.optimizer_c1, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        #self.lr_scheduler_c2 = StepLR(self.optimizer_c2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        #self.mmd_loss = MMD_loss() #Not useful ?!

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop 
            self.pretrain_epoch(src_loader, avg_meter)

            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)

            loss = src_cls_loss1 + src_cls_loss2 #TODO: Not sure about that !

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            loss.backward()

            self.optimizer_c1.step()
            self.optimizer_c2.step()
            self.optimizer_fe.step()


            losses = {'Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features


            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            # source losses
            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)
            loss_s = src_cls_loss1 + src_cls_loss2


            # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = False
            # update C1 and C2 to maximize their difference on target sample
            trg_feat = self.feature_extractor(trg_x)
            trg_pred1 = self.classifier(trg_feat.detach())
            trg_pred2 = self.classifier2(trg_feat.detach())


            loss_dis = self.discrepancy(trg_pred1, trg_pred2)

            loss = loss_s - loss_dis

            loss.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            # Freeze the classifiers
            for k, v in self.classifier.named_parameters():
                v.requires_grad = False
            for k, v in self.classifier2.named_parameters():
                v.requires_grad = False
                        # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = True
            # update feature extractor to minimize the discrepaqncy on target samples
            trg_feat = self.feature_extractor(trg_x)
            trg_pred1 = self.classifier(trg_feat)
            trg_pred2 = self.classifier2(trg_feat)


            loss_dis_t = self.discrepancy(trg_pred1, trg_pred2)
            domain_loss = self.hparams["domain_loss_wt"] * loss_dis_t

            domain_loss.backward()
            self.optimizer_fe.step()

            self.optimizer_fe.zero_grad()
            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()


            losses =  {'Total_loss': loss.item(), 'discrepancy_loss': domain_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        #self.lr_scheduler_fe.step()
        #self.lr_scheduler_c1.step()
        #self.lr_scheduler_c2.step()

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


class SWD(MCD):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(backbone, configs, hparams, device)
        self.N = self.hparams['N']
        self.swd_loss= SliceWassersteinDiscrepancy(self.device, self.N)
    def discrepancy(self, out1, out2):
        return self.swd_loss(out1, out2)

class DeepJDOT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # initialize the gamma (coupling in OT) with zeros
        self.gamma = torch.zeros(hparams["batch_size"],hparams["batch_size"]) #.dnn.K.zeros(shape=(self.batch_size, self.batch_size))
        # hparams
        self.hparams = hparams
        self.nb_classes = configs.num_classes
        # device
        self.device = device
        self.gamma.to(self.device)

        #OT method
        self.ot_method = "emd"
        self.jdot_alpha = hparams["jdot_alpha"]

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.softmax = torch.nn.Softmax()

    def classifier_cat_loss(self, src_y, trg_pred):
        '''
        classifier loss based on categorical cross entropy in the target domain
        1:batch_size - is source samples
        batch_size:end - is target samples
        self.gamma - is the optimal transport plan
        '''
        # source cross entropy loss
        label_loss = -1*torch.matmul(src_y.float(), self.softmax(trg_pred).T)
        return torch.sum(self.gamma * label_loss)

    # L2 distance
    def L2_dist(self, x, y):
        '''
        compute the squared L2 distance between two matrics
        '''
        distx = torch.reshape(torch.sum(torch.square(x), 1), (-1, 1))
        disty = torch.reshape(torch.sum(torch.square(y), 1), (1, -1))
        dist = distx + disty
        dist -= 2.0 * torch.matmul(x, torch.transpose(y, 0, 1))
        return dist

    # feature allignment loss
    def align_loss2(self, src_feat, trg_feat):
        #gdist = self.L2_dist(src_feat, trg_feat)
        gdist = (src_feat-trg_feat).square().sum()
        return torch.sum(self.gamma * gdist)

    def align_loss(self, src_feat, trg_feat):
        gdist = self.L2_dist(src_feat, trg_feat)
        return torch.sum(self.gamma * (gdist))

    def to_categorical(self, y):
        """ 1-hot encodes a tensor """
        return torch.eye(self.nb_classes, dtype=torch.int8)[y]

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs+1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]') #TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            #self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model
    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""
            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features

            #Freeze Neural Network
            for k,v in self.feature_extractor.named_parameters():
                v.requires_grad = False
            for k,v in self.classifier.named_parameters():
                v.requires_grad = False

            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            #extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            src_y = torch.eye(self.nb_classes, dtype=torch.int8).to(self.device)[src_y]

            C0 = torch.cdist(src_feat, trg_feat, p=2.0)**2
            C1 = torch.cdist(src_y.double(), self.softmax(trg_pred).double(), p=2.0)**2  # COMMENT : I put log_softmax
            C = self.hparams["jdot_alpha"] * C0 + self.hparams["jdot_lambda"] * C1
            self.gamma = ot.emd(torch.Tensor(ot.unif(src_x.shape[0])).to(self.device), torch.Tensor(ot.unif(trg_x.shape[0])).to(self.device), C)

            # UnFreeze Neural Network
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = True
            for k, v in self.classifier.named_parameters():
                v.requires_grad = True

            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            #Compute Losses
            feat_align_loss = self.hparams["jdot_alpha"] * self.align_loss(src_feat, trg_feat)
            src_cls_loss = self.hparams["src_cls_loss_wt"] * self.cross_entropy(src_pred, src_y.double())
            label_align_loss = self.hparams["jdot_lambda"] * self.classifier_cat_loss(src_y, trg_pred)
            total_loss = src_cls_loss + feat_align_loss + label_align_loss

            self.optimizer.zero_grad()

            total_loss.backward()

            self.optimizer.step()

            losses = {'Total_loss': total_loss.item(), 'label_disc_loss': label_align_loss.item(), 'feat_disc_loss': feat_align_loss.item(),
                      'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        # self.lr_scheduler_fe.step()
        # self.lr_scheduler_c1.step()
        # self.lr_scheduler_c2.step()

class PPOT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.configs = configs
        self.device = device

        self.src_prototype = None
        self.num_classes = self.configs.num_classes
        self.alpha = 0
        self.class_weight = 0
        self.beta = 0

        self.softmax = torch.nn.Softmax()
        self.is_uniDA = True


    def get_features(self, dataloader):
        feature_set = []
        label_set = []
        self.feature_extractor.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                feature_set.append(feature)
                label_set.append(label)
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
        return feature_set, label_set

    def get_prototypes(self, dataloader) -> torch.Tensor:
        feature_set, label_set = self.get_features(dataloader)
        class_set = [i for i in range(self.num_classes)]
        source_prototype = torch.zeros((len(class_set), feature_set[0].shape[0]))
        for i in class_set:
            source_prototype[i] = feature_set[label_set == i].sum(0) / feature_set[label_set == i].size(0)
        return source_prototype.to(self.device)

    def update_alpha(self, trg_loader) -> np.ndarray:
        num_conf, num_sample = 0, 0
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (trg_x, _) in enumerate(trg_loader):
                trg_x = trg_x.to(self.device)
                output = self.classifier(self.feature_extractor(trg_x))
                output = F.softmax(output, dim=1)
                conf, _ = output.max(dim=1)
                num_conf += torch.sum(conf > self.hparams["tau1"]).item()
                num_sample += output.shape[0]
            alpha = num_conf / num_sample
            alpha = np.around(alpha, decimals=2)
        return alpha

    def entropy_loss(self, prediction: torch.Tensor, weight=torch.zeros(1)):
        if weight.size(0) == 1:
            entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
            entropy = torch.mean(entropy)
        else:
            entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
            entropy = torch.mean(weight * entropy)
        return entropy

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams['num_epochs_pr'] #20#+10+50
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        self.alpha = self.update_alpha(trg_loader)
        self.beta = self.alpha
        self.class_weight = torch.ones(self.num_classes).to(self.device)
        self.src_prototype = self.get_prototypes(src_loader)

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            print("Alpha : ", self.alpha)
            print("Beta : ", self.beta)

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        self.feature_extractor.train()
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            #for params in list(self.classifier.parameters()) + list(self.feature_extractor.parameters()):

            src_feat = self.feature_extractor(src_x)
            src_feat = F.normalize(src_feat, p=2, dim=-1)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)
            head = copy.deepcopy(self.classifier)
            for params in list(head.parameters()):
                params.requires_grad = False
            trg_pred = self.softmax(head(trg_feat))
            assert not (torch.isnan(src_feat).any() or torch.isnan(src_pred).any())
            assert not (torch.isnan(trg_feat).any() or torch.isnan(trg_pred).any())

            #print("trg_pred : ", trg_pred)
            conf,_ = torch.max(trg_pred, dim=1)
            trg_feat = F.normalize(trg_feat, p=2, dim=-1)
            batch_size = trg_feat.shape[0]
            #print("conf : ", conf , f"step : {step}")

            #update alpha by moving average
            self.alpha = (1 - self.hparams['alpha']) * self.alpha + self.hparams['alpha'] * (conf >= self.hparams['tau1']).sum().item() / conf.size(0)
            self.alpha = max(self.alpha, 1e-3)
            self.alpha = min(self.alpha, 1-1e-3)
            # get alpha / beta
            match = self.alpha / self.beta
            assert not np.isnan(match)
            #match = max(match, self.alpha+0.001)
            print("alpha/beta : ", match)

            # update source prototype by moving average
            self.src_prototype = self.src_prototype.detach().cpu() #Else try to re-backprog on previous value
            batch_source_prototype = torch.zeros_like(self.src_prototype)#.to(self.device)
            for i in range(self.num_classes):
                if (src_y == i).sum().item() > 0:
                    batch_source_prototype[i] = (src_feat[src_y == i].mean(dim=0))
                else:
                    batch_source_prototype[i] = (self.src_prototype[i])
            self.src_prototype = (1 - self.hparams["tau"]) * self.src_prototype + self.hparams["tau"] * batch_source_prototype
            self.src_prototype = self.src_prototype.to(self.device)
            #self.src_prototype = F.normalize(self.src_prototype, p=2, dim=-1).to(self.device)
            #self.src_prototype = F.normalize(self.src_prototype, p=2, dim=-1)

            #get ot loss
            #with torch.no_grad():
            a, b = match * ot.unif(self.num_classes), ot.unif(batch_size)
            m = torch.cdist(self.src_prototype, trg_feat) ** 2
            assert not torch.isnan(m).any()
            m_max = m.max().detach()
            m = m / m_max
            pi, log = ot.partial.entropic_partial_wasserstein(a, b, m.detach().cpu().numpy(), reg=self.hparams["reg"], m=self.alpha,
                                                                  stopThr=1e-10, log=True)
            pi = torch.from_numpy(pi).float().to(self.device)
            assert not torch.isnan(pi).any()
            ot_loss = torch.sqrt(torch.sum(pi * m) * m_max)
            loss = self.hparams['ot'] * ot_loss

            '''self.feature_extractor.train()
            self.classifier.train()
            self.optimizer.zero_grad()
            for params in list(self.classifier.parameters()) + list(self.feature_extractor.parameters()):
                params.requires_grad = True'''

            '''src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.softmax(self.classifier(trg_feat))'''

            # update class weight and target weight by plan pi
            plan = pi * batch_size
            k = round(self.hparams['neg']*batch_size) #round(self.hparams['neg'] * batch_size)
            min_dist, _ = torch.min(m, dim=0)
            _, indicate = min_dist.topk(k=k, dim=0)
            batch_class_weight = torch.tensor([plan[i, :].sum() for i in range(self.num_classes)]).to(self.device)
            self.class_weight = self.hparams['tau'] * batch_class_weight + (1 - self.hparams['tau']) * self.class_weight
            self.class_weight = self.class_weight * self.num_classes / self.class_weight.sum()
            k_weight = torch.tensor([plan[:, i].sum() for i in range(batch_size)]).to(self.device)
            k_weight /= self.alpha
            u_weight = torch.zeros(batch_size).to(self.device)
            u_weight[indicate] = 1 - k_weight[indicate]

            # update beta
            self.beta = self.hparams['beta'] * (self.class_weight > self.hparams['tau2']).sum().item() / self.num_classes + (1 - self.hparams['beta']) * self.beta
            self.beta = max(self.beta, 1e-3)#1e-1)
            #self.beta = min(self.beta, 0.999)

            # get classification loss
            cls_loss = F.cross_entropy(src_pred, src_y, weight=self.class_weight.float())
            loss += cls_loss

            # get entropy loss
            p_ent_loss = self.hparams['p_entropy'] * self.entropy_loss(trg_pred, k_weight)
            n_ent_loss = self.hparams['n_entropy'] * self.entropy_loss(trg_pred, u_weight)
            ent_loss = p_ent_loss - n_ent_loss
            loss += ent_loss

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.lr_scheduler.step()
        #Update Protoypes and Alpha
        self.src_prototype = self.get_prototypes(src_loader)
        self.alpha = self.update_alpha(trg_loader)

        losses = {'Total_loss': loss.item(), 'OT Loss': ot_loss.item(),
                  'Entropic Loss': ent_loss.item(),
                  'Src_cls_loss': cls_loss.item()}

        for key, val in losses.items():
            avg_meter[key].update(val, 32)

    def decision_function(self, preds):
        preds = self.softmax(preds)
        confidence, pred = preds.max(dim=1)
        pred[confidence < self.hparams["thresh"]] = -1

        """mask = preds.max()< self.hparams["thresh"]
        res = preds.argmax(dim=1)
        res[mask] = -1"""
        return pred
    def correct_predictions(self, preds):
        print("Correction")
        preds = self.softmax(preds)
        confidence, pred = preds.max(dim=1)
        preds[confidence < self.hparams["thresh"]] *= 0
        return preds
class PseudoInverse(nn.Module):
    def __init__(self, k=5):
        super(PseudoInverse, self).__init__()
        self.k = k

    def forward(self, X):
        """
        Compute the pseudo-inverse of a matrix using SVD and keeping K principal components.

        Parameters:
        - X: Input matrix (PyTorch tensor).

        Returns:
        - X_pseudo_inv: Pseudo-inverse of the input matrix (PyTorch tensor).
        """
        eps = 1e-6
        U, S, V = torch.svd(X.t())
        self.k = torch.sum(S > torch.finfo(S.dtype).eps).item()
        #print("K : ", self.k)
        #self.k = min(self.k, torch.sum(S > torch.finfo(S.dtype).eps).item())
        U_k = U[:, :self.k]
        S_k_inv = torch.diag(1.0 / (S[:self.k]))#+eps))
        X_pseudo_inv = torch.mm(U_k, torch.mm(S_k_inv, U_k.t()))
        return X_pseudo_inv

class JPOT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # initialize the gamma (coupling in OT) with zeros
        self.gamma = torch.zeros(hparams["batch_size"],hparams["batch_size"]) #.dnn.K.zeros(shape=(self.batch_size, self.batch_size))
        # hparams
        self.hparams = hparams
        self.nb_classes = configs.num_classes
        # device
        self.device = device
        self.gamma.to(self.device)

        #OT method
        self.nu = self.hparams["nu"]
        self.m = self.hparams["m"]
        self.mean_A = None
        self.covVar_A = None

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.softmax = torch.nn.Softmax()
        self.pseudo_inv = PseudoInverse()
        self.is_uniDA = True

    def classifier_cat_loss(self, src_y, trg_pred):
        '''
        classifier loss based on categorical cross entropy in the target domain
        1:batch_size - is source samples
        batch_size:end - is target samples
        self.gamma - is the optimal transport plan
        '''
        # source cross entropy loss
        label_loss = -1*torch.matmul(src_y.float(), self.softmax(trg_pred).T)
        return torch.sum(self.gamma * label_loss)


    def mahalanobis(self, feats, mean, cov):
        delta = feats - mean
        pinv = cov#torch.cholesky_inverse(cov)
        '''assert not torch.isnan(delta).any()
        assert not torch.isnan(pinv).any()
        ddot = torch.matmul(pinv, delta)
        assert not torch.isnan(ddot).any()
        m = torch.dot(delta, ddot)'''
        m = torch.sum(((delta @ pinv) * delta), axis=-1)
        return torch.sqrt(m)

    def mahalanobis_tensor(self, feats, mean, cov):
        delta = feats - mean
        cinv = cov#torch.pinverse(cov)
        mahalanobis_dist = torch.sqrt(torch.einsum('ij,ij->i', [delta.mm(cinv), delta]))
        return mahalanobis_dist

    # L2 distance
    def L2_dist(self, x, y):
        '''
        compute the squared L2 distance between two matrics
        '''
        distx = torch.reshape(torch.sum(torch.square(x), 1), (-1, 1))
        disty = torch.reshape(torch.sum(torch.square(y), 1), (1, -1))
        dist = distx + disty
        dist -= 2.0 * torch.matmul(x, torch.transpose(y, 0, 1))
        return dist

    # feature allignment loss
    def align_loss2(self, src_feat, trg_feat):
        #gdist = self.L2_dist(src_feat, trg_feat)
        gdist = (src_feat-trg_feat).square().sum()
        return torch.sum(self.gamma * gdist)

    def align_loss(self, src_feat, trg_feat):
        gdist = self.L2_dist(src_feat, trg_feat)
        return torch.sum(self.gamma * (gdist))

    def to_categorical(self, y):
        """ 1-hot encodes a tensor """
        return torch.eye(self.nb_classes, dtype=torch.int8)[y]

    """def pseudo_inv(self, X, k=3):
        U, S, V = torch.svd(X.t())
        U_k = U[:, :k]
        S_k_inv = torch.diag(1.0 / S[:k])
        X_pseudo_inv = torch.mm(U_k, torch.mm(S_k_inv, U_k.t()))
        return X_pseudo_inv"""

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams['num_epochs_pr']
        for epoch in range(1, nb_pr_epochs+1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]') #TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        self.get_prototypes(src_loader, trg_loader)
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            #self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model
    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def get_features(self, dataloader):
        feature_set = []
        label_set = []
        pred_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                pred = self.softmax(self.classifier(feature))
                feature_set.append(feature)
                label_set.append(label)
                pred_set.append(pred)
            feature_set = torch.cat(feature_set, dim=0)
            #feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            pred_set = torch.cat(pred_set, dim=0)
        return feature_set, label_set, pred_set

    def get_prototypes(self, src_dataloader, trg_dataloader) -> torch.Tensor:
        feature_set, label_set, pred_set = self.get_features(src_dataloader)
        #feature_set = F.normalize(feature_set, p=2, dim=-1)

        mean_A = torch.zeros(self.nb_classes + 1, feature_set.shape[-1]).to(self.device)
        covVar_A = torch.zeros(self.nb_classes + 1, feature_set.shape[-1], feature_set.shape[-1]).to(self.device)
        for i in list(range(self.nb_classes)):
            if (label_set == i).sum().item() != 0:
                """if len(src_y==i) == 1:
                    mean_A[i] = feats
                    covVar_A[i] = torch.var(feats)
                else:"""
                mean_A[i] = feature_set[label_set == i].mean(axis=0)  # .view(1, -1)
                # covVar_A[i] = 1/max(1 ,(feats.shape[0] - 1)) * torch.matmul((feats - mean_A[i]).T, (feats - mean_A[i]))
                covVar_A[i] = self.pseudo_inv(feature_set[label_set == i])#torch.cov(feature_set[label_set == i].T)
                # assert torch.isnan(torch.pinverse(covVar_A[i])).sum() == 0

        feature_set, label_set, pred_set = self.get_features(trg_dataloader)
        confidence, pred = pred_set.max(dim=1)
        q = torch.quantile(confidence, 2*(1-self.hparams["qt"]))
        mask = confidence < q
        print("Number mask : ", mask.sum())
        mean_A[-1] = feature_set[mask].mean(axis=0)  # .view(1, -1)
        covVar_A[-1] = self.pseudo_inv(feature_set[mask])
        self.mean_A = mean_A
        self.covVar_A = covVar_A
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""
            if src_x.shape[0] > trg_x.shape[0]:  # TODO: Delete ?
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features

            '''for p in list(self.classifier.parameters()) + list(self.feature_extractor.parameters()):
                p.requires_grad = False'''
            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            head = self.classifier
            for p in head.parameters():
                p.requires_grad = False
            trg_pred = head(trg_feat)

            assert not (torch.isnan(src_feat).any() or torch.isnan(src_pred).any())
            assert not (torch.isnan(trg_feat).any() or torch.isnan(trg_pred).any())
            #trg_feat = F.normalize(trg_feat, p=2, dim=-1)
            #src_feat = F.normalize(src_feat, p=2, dim=-1)

            src_y_hot = torch.eye(self.nb_classes, dtype=torch.int8).to(self.device)[src_y]
            #with torch.no_grad():
            C0 = torch.cdist(src_feat, trg_feat, p=2.0) ** 2
            #C1 = torch.cdist(src_y_hot.double(), self.softmax(trg_pred).double(),p=2.0) ** 2
            #C = self.hparams['jdot_alpha'] * C0 #+ self.hparams['jdot_lambda'] * C1
            C = C0
            # self.gamma = ot.sinkhorn(torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device), C, reg=0.001)
            #self.gamma = ot.sinkhorn(torch.zeros(C.shape[0]).to(self.device)+1, torch.zeros(C.shape[1]).to(self.device)+1, C, reg=0.1) #* src_feat.shape[0]
            self.gamma = ot.sinkhorn(torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device), C.detach().cpu(), reg=self.hparams["reg"])
            assert not torch.isnan(C).any()
            assert not torch.isnan(self.gamma).any()
            self.gamma = self.gamma.to(self.device)
            #self.gamma = ot.emd(torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device), C)
            rho = torch.quantile(self.gamma, self.hparams['qt']) #.sum(axis=0)/src_x.shape[0]
            assert not torch.isnan(self.gamma).any()
            h = 1 - 0.5 * (1 + torch.sign(self.gamma - rho))
            gamma_k = self.gamma * h
            gamma_u = self.gamma - gamma_k
            #print("Rho : ", rho)
            #print("Cost : ", C.min(), C.max(), (self.gamma * C).sum())
            #print("h : ", h.mean())
            print("Gamma K sum : ", gamma_k.sum())
            print("Gamma U sum : ", gamma_u.sum())


            # Losses
            l2 = self.L2_dist(src_feat, trg_feat)
            loss_ot_k = torch.sum(self.gamma * l2)
            loss_ot_u = 1 / self.nu * torch.sum(gamma_u * torch.log(1 + torch.exp(-1 * self.nu * self.L2_dist(src_feat, trg_feat))))
            l_p = loss_ot_k + loss_ot_u
            assert not torch.isnan(l_p).any()
            assert not torch.isnan(l2).any()

            l_cls = self.cross_entropy(src_pred, src_y)

            # Compute Mean and Cov
            #with torch.no_grad():
            self.mean_A = self.mean_A.to(self.device)
            self.covVar_A = self.covVar_A.to(self.device)
            mean_A = self.mean_A #torch.zeros(self.nb_classes + 1, src_feat.shape[-1]).to(self.device)
            covVar_A = self.covVar_A #torch.zeros(self.nb_classes + 1, src_feat.shape[-1], src_feat.shape[-1]).to(self.device)
            #mean_A, covVar_A = self.mean_A.copy(), self.covVar_A.copy()
            for i in list(range(self.nb_classes)):
                feats = src_feat[src_y == i]
                if (src_y == i).sum().item() > 1 :
                    mean_A[i] = feats.mean(axis=0)
                    #covVar_A[i] = torch.cov(feats.T)
                    self.mean_A[i] = self.m * self.mean_A[i] + (1 - self.m) * feats.mean(axis=0)
                    #mean_A[i] = feats.mean(axis=0)  # .view(1, -1)
                    # covVar_A[i] = 1/max(1 ,(feats.shape[0] - 1)) * torch.matmul((feats - mean_A[i]).T, (feats - mean_A[i]))
                    self.covVar_A[i] = self.m *  self.covVar_A[i] + (1 - self.m) * self.pseudo_inv(feats) #torch.cov(feats.T)
                elif (src_y == i).sum().item() == 1 :
                    #mean_A[i] = feats.mean(axis=0)
                    self.mean_A[i] = self.m * self.mean_A[i] + (1 - self.m) * feats.mean(axis=0)
                    # assert torch.isnan(torch.pinverse(covVar_A[i])).sum() == 0
                feats = trg_feat[gamma_u.sum(axis=0).int()] if gamma_u.sum(axis=0).sum().item() != 0 else None#trg_feat
                if not feats is None:
                    if feats.shape[0] == 1:
                        #mean_A[-1] = feats
                        self.mean_A[-1] = self.m * self.mean_A[-1] + (1 - self.m) * feats  # .view(1, -1)
                    if feats.shape[0] > 1:
                        #covVar_A[-1] = torch.cov(feats.T)
                        self.mean_A[-1] = self.m * self.mean_A[-1] + (1 - self.m) * feats.mean(axis=0)
                        self.covVar_A[-1] = self.m * self.covVar_A[-1] + (1 - self.m) * self.pseudo_inv(feats) #torch.cov(feats.T)
                # covVar_A[-1] = 1 / (feats.shape[0] - 1) * torch.matmul((feats - mean_A[-1]).T, (feats - mean_A[-1]))
            #self.mean_A = F.normalize(self.mean_A, p=2, dim=-1)
            #self.covVar_A = F.normalize(self.covVar_A, p=2, dim=-1)
            assert not torch.isnan(mean_A).any()
            assert not torch.isnan(covVar_A).any()
            # Update Mean and Cov

                #self.mean_A = self.m * self.mean_A + (1 - self.m) * mean_A
                #self.covVar_A = self.m * self.covVar_A + (1 - self.m) * covVar_A

            D_trg = torch.zeros(self.mean_A.shape[0], trg_x.shape[0]).to(self.device)
            """for z in range(self.mean_A.shape[0]):
                if self.covVar_A[z].sum() != 0:
                    D_trg[z] = self.mahalanobis_tensor(trg_feat, self.mean_A[z], self.covVar_A[z])"""

            for z in range(self.mean_A.shape[0]):
                for j in range(trg_x.shape[0]):
                    if covVar_A[z].sum():
                        D_trg[z, j] = 0
                    else:
                        D_trg[z, j] = -1 * (trg_feat[j] - self.mean_A[z]).square().sum()
                        # -1 * self.mahalanobis(trg_feat[j], self.mean_A[z], self.covVar_A[z])

                    #D_trg[z, j] = -1 * (trg_feat[j] - mean_A[z]).square().sum()
                    #d = self.mahalanobis(trg_feat[j], self.mean_A[z], self.covVar_A[z])
                    #D_trg[z, j] = -1*(trg_feat[j]-mean_A[z]).square().sum()
                    #d_exp = torch.exp(-1*d)
                    #D_trg[z,j] = -1*d_exp #self.mahalanobis(trg_feat[j], mean_A[z], covVar_A[z])
            #assert not np.isnan(D_trg).any()
            #assert not torch.isnan(D_trg).any()
            #D_trg = torch.Tensor(D_trg).to(self.device)
            #D_trg2 = torch.zeros_like(D_trg).to(self.device)

            #D_trg = -1 * torch.cdist(trg_feat, mean_A)
            #D_trg = -1*cdist(trg_feat.detach().cpu(), mean_A.detach().cpu(), 'mahalanobis', VI=covVar_A.detach().cpu())
            #D_trg = torch.Tensor(D_trg).to(self.device)
            D_trg_soft = self.softmax(D_trg+1e-12)
            #print(D_trg_soft)
            assert not torch.isnan(D_trg).any()

            l_dc = (src_feat - self.mean_A[src_y]).square().sum()
            l_dp = 0
            for i in range(self.nb_classes):
                fake_labels = torch.Tensor([i] * len(src_y)).to(self.device)
                ll = D_trg_soft[i] * self.cross_entropy(self.softmax(trg_pred), fake_labels.long())
                l_dp += ll
            # l_dp = torch.Tensor([D_trg[i]*self.cross_entropy(self.softmax(trg_pred), i) for i in range(self.nb_classes+1)]).to(self.device)
            l_dp = l_dp.sum()
            l_d = l_dc + l_dp
            assert not (torch.isnan(l_dp).any() or torch.isnan(l_dc).any())
            # Compute Losses
            src_cls_loss = self.cross_entropy(src_pred, src_y)
            total_loss = l_cls + self.hparams['alpha'] * l_p + self.hparams['beta'] * l_d

            self.optimizer.zero_grad()

            total_loss.backward()

            self.optimizer.step()

            losses = {'Total_loss': total_loss.item(),
                      'OT_Loss': self.hparams['alpha'] * l_p.item(),
                      'Reweighted Loss': self.hparams['beta'] * l_d.item(),
                      'Src_cls_loss': src_cls_loss.item()}

            self.mean_A = mean_A.detach().cpu()
            self.covVar_A = covVar_A.detach().cpu()


            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def decision_function(self, preds):
        preds = self.softmax(preds)
        confidence, pred = preds.max(dim=1)
        pred[confidence < self.hparams["thresh"]] = -1

        """mask = preds.max()< self.hparams["thresh"]
        res = preds.argmax(dim=1)
        res[mask] = -1"""
        return pred

    def correct_predictions(self, preds):
        print("Correction")
        preds = self.softmax(preds)
        confidence, pred = preds.max(dim=1)
        preds[confidence < self.hparams["thresh"]] *= 0
        return preds
        # self.lr_scheduler_fe.step()
        # self.lr_scheduler_c1.step()
        # self.lr_scheduler_c2.step()