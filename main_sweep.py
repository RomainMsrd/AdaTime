from trainers.sweep import Trainer
import argparse
parser = argparse.ArgumentParser()
import wandb
import os
#os.environ["WANDB_MODE"]="offline"


if __name__ == "__main__":
    wandb.login()
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='DeepJDOT', type=str,
                        help='DANN, Deep_Coral, WDGRL, MMDA, VADA, DIRT, CDAN, ADDA, HoMM, CoDATS, SWD, MCD, DeepJDOT')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'./ADATIME_data', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='HHAR', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')
    parser.add_argument("--uniDA", action='store_false', help='Different Label Set between Src and Trg Domain ?')
    parser.add_argument("--generate-private", action='store_false', help='uniDA should be True too ?')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=4, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--exp_name',     default='sweep_EXP1',         type=str, help='experiment name')

    # ======== sweep settings =====================
    parser.add_argument('--num_sweeps', default=250, type=str, help='Number of sweep runs')

    # We run sweeps using wandb plateform, so next parameters are for wandb.
    parser.add_argument('--sweep_project_wandb', default='AdaTime_OT', type=str, help='Project name in Wandb')
    parser.add_argument('--wandb_entity', type=str,
                        help='Entity name in Wandb (can be left blank if there is a default entity)')
    parser.add_argument('--hp_search_strategy', default="bayes", type=str,
                        help='The way of selecting hyper-parameters (random-grid-bayes). in wandb see:https://docs.wandb.ai/guides/sweeps/configuration')
    parser.add_argument('--metric_to_minimize', default="src_risk", type=str,
                        help='select one of: (src_risk - trg_risk - few_shot_trg_risk - dev_risk)')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs/sweep_logs', type=str,
                        help='Directory containing all experiments')

    args = parser.parse_args()

    trainer = Trainer(args)

    #trainer.sweep(sweep_id="oddteam/AdaTime_OT/vzca1rrs")
    trainer.sweep()
