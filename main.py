from trainers.train import Trainer

import argparse
import time
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    start_time = time.time()

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='train',         type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name',               default='EXP1',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='OpenJDOT',               type=str, help='NO_ADAPT, Deep_Coral, MMDA, DANN, CDAN, DIRT, DSAN, HoMM, CoDATS, AdvSKM, SASA, CoTMix, TARGET_ONLY')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'./ADATIME_data',                  type=str, help='Path containing dataset')
    parser.add_argument('--dataset',                default='ToyDataset',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=4,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "cuda",                   type=str, help='cpu or cuda')
    parser.add_argument("--uniDA",                  action='store_false', help='Different Label Set between Src and Trg Domain ?')
    parser.add_argument("--generate-private", action='store_false', help='uniDA should be True too ?')

    # arguments
    args = parser.parse_args()

    # create trainier object
    trainer = Trainer(args)

    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()

    print("--- %s seconds ---" % (time.time() - start_time))



#TODO:
# 1- Change the naming of the functions ---> ( Done)
# 2- Change the algorithms following DCORAL --> (Done)
# 3- Keep one trainer for both train and test -->(Done)
# 4- Create the new joint loader that consider the all possible batches --> Done
# 5- Implement Lower/Upper Bound Approach --> Done
# 6- Add the best hparams --> Done
# 7- Add pretrain based methods (ADDA, MCD, MDD)
