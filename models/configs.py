

def get_model_config(args):

    # Ours pre-trained foundation model
    if args.model_id == "fseft":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/fseft.pth",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 29}

    # SUPrem UNET pre-trained
    elif args.model_id == "suprem_unet":
        model_cfg = {"architecture": "unet3d", "model_path": "./models/pretrained_weights/suprem_unet.pth",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 64, "channelOut": 29}

    # SUPrem UNET pre-trained
    elif args.model_id == "suprem_swinunetr":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/suprem_swinunetr.pth",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 29}

    # CLIP-driven foundation model
    elif args.model_id == "clipdriven":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/clipdriven.pth",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 29}

    # Tang et al. (2022) pre-trained on BTCV
    elif args.model_id == "selfsup":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/selfsup.pt",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 14}

    # Tang et al. (2022) pre-trained on BTCV
    elif args.model_id == "btcv":
        model_cfg = {"architecture": "swinunetr", "model_path": "./models/pretrained_weights/btcv.pt",
                     "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1, "space_x": 1.5, "space_y": 1.5,
                     "space_z": 1.5, "roi_x": 96, "roi_y": 96, "roi_z": 96, "fout": 48, "channelOut": 14}

    # nnU-Net 3D UNet configuration
    elif args.model_id == "nnunet":
        model_cfg = {"architecture": "unet3d", "model_path": "./models/pretrained_weights/nnunet.pth",
                     "normalization": "ZScoreNormalization", "space_x": 1.0, "space_y": 1.0, "space_z": 1.0,
                     "roi_x": 160, "roi_y": 128, "roi_z": 112, "fout": 32, "channelOut": 1,
                     "plans_path": "./plans.json", "checkpoint_path": "./models/pretrained_weights/nnunet.pth",
                     "features_per_stage": [32, 64, 128, 256, 320, 320], "n_stages": 6,
                     "patch_size": [160, 128, 112], "spacing": [1.0, 1.0, 1.0]}

    else:
        raise ValueError(f"Unknown model_id: {args.model_id}. Supported options are: fseft, suprem_unet, suprem_swinunetr, clipdriven, selfsup, btcv, nnunet")

    # Parse keys into argparser
    args.model_cfg = model_cfg
