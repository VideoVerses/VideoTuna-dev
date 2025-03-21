import argparse
import datetime
import os
import sys

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.cli import LightningCLI
from transformers import logging as transf_logging

# sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(0, os.getcwd())
from videotuna.utils.args_utils import prepare_train_args
from videotuna.utils.common_utils import instantiate_from_config
from videotuna.utils.lightning_utils import add_trainer_args_to_parser
from videotuna.utils.train_utils import (
    check_config_attribute,
    get_autoresume_path,
    get_empty_params_comparedwith_sd,
    get_trainer_callbacks,
    get_trainer_logger,
    get_trainer_strategy,
    init_workspace,
    load_checkpoints,
    set_logger,
)


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--seed", "-s", type=int, default=20230211, help="seed for seed_everything"
    )
    parser.add_argument(
        "--name", "-n", type=str, default="", help="experiment name, as saving folder"
    )

    parser.add_argument(
        "--base",
        "-b",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "--train", "-t", action="store_true", default=False, help="train"
    )
    parser.add_argument("--val", "-v", action="store_true", default=False, help="val")
    parser.add_argument("--test", action="store_true", default=False, help="test")

    parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="pretrained current model checkpoint"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="resume from full-info checkpoint",
    )
    parser.add_argument(
        "--trained_ckpt", type=str, default=None, help="pretrained current model checkpoint"
    )
    parser.add_argument(
        "--lorackpt", type=str, default=None, help="pretrained current model checkpoint"
    )
    return parser


def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = add_trainer_args_to_parser(Trainer, parser)

    default_trainer_args = parser.parse_args([])
    return sorted(
        k
        for k in vars(default_trainer_args)
        if getattr(args, k) != getattr(default_trainer_args, k)
    )


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    try:
        local_rank = int(os.environ.get("LOCAL_RANK"))
        global_rank = int(os.environ.get("RANK"))
        num_rank = int(os.environ.get("WORLD_SIZE"))
    except:
        local_rank, global_rank, num_rank = 0, 0, 1

    parser = get_parser()
    args, config = prepare_train_args(parser)

    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    train_config = config.pop("train", OmegaConf.create())
    lightning_config = train_config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(
        args.name, args.logdir, config, lightning_config, global_rank
    )
    logger = set_logger(
        logfile=os.path.join(loginfo, "log_%d:%s.txt" % (global_rank, now))
    )
    logger.info("@lightning version: %s [>=2.0 required]" % pl.__version__)

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configuring Model *****")
    config.flow.params.logdir = workdir

    flow = instantiate_from_config(config.flow)
    if args.resume_ckpt is not None:
        print("Resuming from checkpoint {}".format(args.resume_ckpt))
    else:
        flow.from_pretrained(args.ckpt)
    if args.trained_ckpt is not None:
        flow.load_trained_ckpt(args.trained_ckpt)
    # if len(flow.lora_args) != 0:
    #     flow.inject_lora()
    ## update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)

    print(f"trainer_config: {trainer_config}")
    num_nodes = trainer_config.num_nodes
    ngpu_per_node = trainer_config.devices
    logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

    ## setup learning rate
    base_lr = config.flow.base_learning_rate
    bs = train_config.data.params.batch_size
    if getattr(config.flow, "scale_lr", True):
        flow.learning_rate = num_rank * bs * base_lr
    else:
        flow.learning_rate = base_lr

    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configuring Data *****")
    data = instantiate_from_config(train_config.data)
    data.setup()
    for k in data.datasets:
        logger.info(
            f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
        )

    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configuring Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    ## setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    print(f"logger save_dir: {trainer_kwargs['logger'].save_dir}")
    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(
        lightning_config, config, workdir, ckptdir, logger
    )
    callbacks_cfg["image_logger"]["params"]["save_dir"] = workdir
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = (
        strategy_cfg
        if type(strategy_cfg) == str
        else instantiate_from_config(strategy_cfg)
    )
    trainer_kwargs["sync_batchnorm"] = False

    # merge args for trainer
    print(f"trainer_kwargs: {trainer_kwargs}")
    trainer = Trainer(**trainer_config, **trainer_kwargs)

    ## allow checkpointing via USR1
    def melk(*args, **kwargs):
        ## run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb

            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    ## Running LOOP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Running the Loop *****")

    if args.train:
        try:
            # Strategy is automatically managed, no need to manually check it here
            logger.info(f"<Training in {trainer.strategy.__class__.__name__} Mode>")
            # Please refer to https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.precision.MixedPrecision.html for Automatic Mixed Precision (AMP) training
            if trainer.strategy == "deepspeed":
                with torch.cuda.amp.autocast():
                    trainer.fit(flow, data, ckpt_path=args.resume_ckpt)
            else:
                trainer.fit(flow, data, ckpt_path=args.resume_ckpt)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
