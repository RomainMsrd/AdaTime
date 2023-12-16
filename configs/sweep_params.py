sweep_train_hparams = {
        'num_epochs':   {'values': [3, 4, 5, 6]},
        'batch_size':   {'values': [32, 64]},
        'learning_rate':{'values': [1e-2, 5e-3, 1e-3, 5e-4]},
        'disc_lr':      {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
        'weight_decay': {'values': [1e-4, 1e-5, 1e-6]},
        'step_size':    {'values': [5, 10, 30]},
        'gamma':        {'values': [5, 10, 15, 20, 25]},
        'optimizer':    {'values': ['adam']},
}

sweep_train_hparams2 = {
        'num_epochs':   {'values': [70, 160, 250]},
        'num_epochs_pr':{'values': [50, 100]},
        'batch_size':   {'values': [32, 64, 128]},
        'weight_decay': {'values': [1e-4]}
}

sweep_alg_hparams = {
        'DANN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'AdvSKM': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CoDATS': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CDAN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'Deep_Coral': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
        },

        'DIRT': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'vat_loss_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'HoMM': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'hommd_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'MMDA': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'DSAN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'DDC': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },
        
        'SASA': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CoTMix': {
            'learning_rate':            {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'temporal_shift':           {'values': [5, 10, 15, 20, 30, 50]},
            'src_cls_weight':           {'distribution': 'uniform', 'min': 1e-1, 'max': 1},
            'mix_ratio':                {'distribution': 'uniform', 'min': 0.5, 'max': 0.99},
            'src_supCon_weight':        {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_cont_weight':          {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_entropy_weight':       {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
        },

        'MCD': {
            'learning_rate':    {'values': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'SWD': {
            'learning_rate':    {'values': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            "N":                {'values': [128*4, 128*8, 128*16, 128*32]},
        },

        'DeepJDOT': {
            'learning_rate':    {'values': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 5},
            "jdot_alpha":       {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            "jdot_lambda":       {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            "ot_method":       {'values': ["emd"]},
        },

        'PPOT': {
                'learning_rate': {'values': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]},
                'tau': {'values': [0.05, 0.1, 0.2, 0.3, 0.4]},
                "tau1": {'values': [0.7, 0.8, 0.9]},
                "tau2": {'values': [0.8, 0.9, 1]},
                "alpha": {'values': [0.01, 0.001]},
                "beta": {'values': [0.01, 0.001]},
                "reg": {'values': [0.01, 0.1]},
                "ot" : {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
                "p_entropy": {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
                "n_entropy": {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
                "neg" : {'values': [0.15, 0.2, 0.25, 0.3, 0.35]},
                "thresh": {'values': [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]},
            },
        'JPOT': {
                'learning_rate': {'values': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]},
                "alpha" : {'distribution': 'uniform', 'min': 0.001, 'max': 1},
                "beta" : {'distribution': 'uniform', 'min': 0.001, 'max': 1},
                'nu': {'distribution': 'uniform', 'min': 0.001, 'max': 1},
                'qt': {'distribution': 'uniform', 'min': 0.85, 'max': 0.99},
                'm' : {'distribution': 'uniform', 'min': 0.1, 'max': 0.8},
                "reg": {'values': [0.01, 0.1]},
                "thresh": {'values': [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]},
            },

}

