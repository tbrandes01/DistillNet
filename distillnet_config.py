hparams = {
            'batch_size': 256,
            'maketrain_particles': 1.4e6,
            'L1_hsize': 128,
            'L2_hsize': 64,
            'n_outputs': 1,
            'lr': 0.0015,
            }

trainparams = {
            'n_epochs': 2,
            "patience": 5,
            "best_loss": 0.17,
            'weightedlossval': 3,
            "train_sample": "distill_wjets_emd_prl.h5", #sample for training distill_wjets_swd.h5
            "test_sample": "distill_ttbar_emd_prl.h5",  #sample for testing distill_ttbar_swd.h5
            }


dirs = {
        'filedir': '/work/tbrandes/work/data/',
        'savedir': '/work/tbrandes/delme/'
    
}

bool_val = {
            "is_displayplots": False,
            "is_savefig": True,
            "is_remove_padding": True,
            "is_min_max_scaler": False,
            "is_standard_scaler": True,
            "is_dtrans": False,
            "is_do_taylor": True,
            "is_weighted_error": True,
            }
