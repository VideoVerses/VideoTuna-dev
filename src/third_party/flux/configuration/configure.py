import os
import huggingface_hub
import torch
from src.third_party.flux.training import quantised_precision_levels, lycoris_defaults
from src.third_party.flux.training.optimizer_param import optimizer_choices

bf16_only_optims = [
    key
    for key, value in optimizer_choices.items()
    if value.get("precision", "any") == "bf16"
]
any_precision_optims = [
    key
    for key, value in optimizer_choices.items()
    if value.get("precision", "any") == "any"
]
model_classes = {
    "full": [
        "flux",
        "sdxl",
        "pixart_sigma",
        "kolors",
        "sd3",
        "legacy",
    ],
    "lora": ["flux", "sdxl", "kolors", "sd3", "legacy"],
    "controlnet": ["sdxl", "legacy"],
}

default_models = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "pixart_sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "kolors": "kwai-kolors/kolors-diffusers",
    "terminus": "ptx0/terminus-xl-velocity-v2",
    "sd3": "stabilityai/stable-diffusion-3.5-large",
    "legacy": "stabilityai/stable-diffusion-2-1-base",
}

default_cfg = {
    "flux": 3.0,
    "sdxl": 4.2,
    "pixart_sigma": 3.4,
    "kolors": 5.0,
    "terminus": 8.0,
    "sd3": 5.0,
}

model_labels = {
    "sd3": "Stable Diffusion 3",
    "flux": "FLUX",
    "pixart_sigma": "PixArt Sigma",
    "kolors": "Kwai Kolors",
    "terminus": "Terminus",
    "sdxl": "Stable Diffusion XL",
    "legacy": "Stable Diffusion",
}

lora_ranks = [1, 16, 64, 128, 256]
learning_rates_by_rank = {
    1: "3e-4",
    16: "1e-4",
    64: "8e-5",
    128: "6e-5",
    256: "5.09e-5",
}


def print_config(env_contents: dict, extra_args: list):
    # env_contents["TRAINER_EXTRA_ARGS"] = " ".join(extra_args)
    # output = json.dumps(env_contents, indent=4)
    # print(output)
    pass


def prompt_user(prompt, default=None):
    if default:
        prompt = f"{prompt} (default: {default})"
    user_input = input(f"{prompt}: ")
    return user_input.strip() or default


def configure_lycoris():
    print("Let's configure your LyCORIS model!\n")

    print("Select a LyCORIS algorithm:\n")

    print(
        "1. LoRA - Efficient, balanced fine-tuning. Good for general tasks. (algo=lora)"
    )
    print(
        "2. LoHa - Advanced, strong dampening. Ideal for multi-concept fine-tuning. (algo=loha)"
    )
    print(
        "3. LoKr - Kronecker product-based. Use for complex transformations. (algo=lokr)"
    )
    print("4. Full Fine-Tuning - Traditional full model tuning. (algo=full)")
    print("5. IA^3 - Efficient, tiny files, best for styles. (algo=ia3)")
    print("6. DyLoRA - Dynamic updates, efficient with large dims. (algo=dylora)")
    print("7. Diag-OFT - Fast convergence with orthogonal fine-tuning. (algo=diag-oft)")
    print("8. BOFT - Advanced version of Diag-OFT with more flexibility. (algo=boft)")
    print("9. GLoRA - Generalized LoRA. (algo=glora)\n")

    # Prompt user to select an algorithm
    algo = prompt_user(
        f"Which LyCORIS algorithm would you like to use? (Enter the number corresponding to the algorithm)",
        "3",  # Default to LoKr
    )

    # Map the selected number to the actual algorithm name
    algo_map = {
        "1": "lora",
        "2": "loha",
        "3": "lokr",
        "4": "full",
        "5": "ia3",
        "6": "dylora",
        "7": "diag-oft",
        "8": "boft",
        "9": "glora",
    }

    algo = algo_map.get(algo, "lokr").lower()

    # Get the default configuration for the selected algorithm
    default_config = lycoris_defaults.get(algo, {}).copy()

    # Continue with further configuration
    print(f"\nConfiguring {algo.upper()} algorithm...\n")

    multiplier = float(
        prompt_user(
            f"Set the effect multiplier. Adjust for stronger or subtler effects. "
            f"(default: {default_config.get('multiplier', 1.0)})",
            default_config.get("multiplier", 1.0),
        )
    )

    linear_dim = int(
        prompt_user(
            f"Set the linear dimension. Higher values mean more capacity but use more resources. "
            f"(default: {default_config.get('linear_dim', 1000000)})",
            default_config.get("linear_dim", 1000000),
        )
    )

    linear_alpha = int(
        prompt_user(
            f"Set the alpha scaling factor. Controls the impact on the model. "
            f"(default: {default_config.get('linear_alpha', 1)})",
            default_config.get("linear_alpha", 1),
        )
    )

    # Update basic parameters in config
    default_config.update(
        {
            "multiplier": multiplier,
            "linear_dim": linear_dim,
            "linear_alpha": linear_alpha,
        }
    )

    # Conditional prompts based on the selected algorithm
    if algo == "lokr":
        factor = int(
            prompt_user(
                f"Set the factor for compression/expansion. "
                f"(default: {default_config.get('factor', 16)})",
                default_config.get("factor", 16),
            )
        )
        default_config.update({"factor": factor})

        if linear_dim >= 10000:  # Handle full-dimension case
            print("Full-dimension mode activated. Alpha will be set to 1.")
            default_config["linear_alpha"] = 1

    elif algo == "loha":
        if linear_dim > 32:
            print("Warning: High dim values with LoHa may cause instability.")
        # Additional LoHa-specific configurations can be added here if needed

    elif algo == "dylora":
        block_size = int(
            prompt_user(
                f"Set block size for DyLoRA (rows/columns updated per step). "
                f"(default: {default_config.get('block_size', 0)})",
                default_config.get("block_size", 0),
            )
        )
        default_config.update({"block_size": block_size})

    elif algo in ["diag-oft", "boft"]:
        constraint = (
            prompt_user(
                f"Enforce constraints (e.g., orthogonality)? "
                f"(True/False, default: {default_config.get('constraint', False)})",
                str(default_config.get("constraint", False)),
            ).lower()
            == "true"
        )

        rescaled = (
            prompt_user(
                f"Rescale transformations? Adjusts model impact. "
                f"(True/False, default: {default_config.get('rescaled', False)})",
                str(default_config.get("rescaled", False)),
            ).lower()
            == "true"
        )

        default_config.update(
            {
                "constraint": constraint,
                "rescaled": rescaled,
            }
        )

    # Handle presets for specific modules
    if "apply_preset" in default_config:
        print("\nNext, configure the modules to target with this algorithm.")
        target_module = prompt_user(
            f"Which modules should the {algo.upper()} algorithm be applied to? "
            f"(default: {', '.join(default_config['apply_preset']['target_module'])})",
            ", ".join(default_config["apply_preset"]["target_module"]),
        ).split(",")
        default_config["apply_preset"]["target_module"] = [
            m.strip() for m in target_module
        ]

        for module_name, module_config in default_config["apply_preset"][
            "module_algo_map"
        ].items():
            for param, value in module_config.items():
                user_value = prompt_user(
                    f"Set {param} for {module_name}. " f"(default: {value})", value
                )
                module_config[param] = (
                    int(user_value) if isinstance(value, int) else float(user_value)
                )

    print("\nLyCORIS configuration complete: ", default_config)
    return default_config


def configure_env():
    print("Welcome to SimpleTuner!")
    print("This script will guide you through setting up your config.json file.\n")
    env_contents = {
        "--resume_from_checkpoint": "latest",
        "--data_backend_config": "configs/006_flux/multidatabackend.json",
        "--aspect_bucket_rounding": 2,
        "--seed": 42,
        "--minimum_image_size": 0,
        "--disable_benchmark": False,
    }
    extra_args = []

    output_dir = prompt_user(
        "Enter the directory where you want to store your outputs", "output/models"
    )
    while not os.path.exists(output_dir):
        should_create = (
            prompt_user(
                "That directory did not exist. Should I create it? Answer 'n' to select a new location. ([y]/n)",
                "y",
            )
            == "y"
        )
        if should_create:
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(
                f"Directory {output_dir} does not exist. Please create it and try again."
            )
            output_dir = prompt_user(
                "Enter the directory where you want to store your outputs",
                "output/models",
            )
    env_contents["--output_dir"] = output_dir

    # Start with the basic options
    model_type = prompt_user(
        "What type of model are you training? (Options: [lora], full)", "lora"
    ).lower()
    use_lycoris = False
    use_lora = False
    if model_type == "lora":
        use_lora = True
        use_lycoris = (
            prompt_user("Would you like to train a LyCORIS model? ([y]/n)", "y").lower()
            == "y"
        )
        if use_lycoris:
            env_contents["--lora_type"] = "lycoris"
            lycoris_config = configure_lycoris()
            env_contents["--lycoris_config"] = "configs/006_flux/lycoris_config.json"
            # write json to file
            import json

            # approximate the rank of the lycoris
            lora_rank = 16
            with open("configs/006_flux/lycoris_config.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(lycoris_config, indent=4))
        else:
            env_contents["--lora_type"] = "standard"
            use_dora = prompt_user(
                "Would you like to train a DoRA model? (y/[n])", "n"
            ).lower()
            if use_dora == "y":
                env_contents["--use_dora"] = "true"
            lora_rank = None
            while lora_rank not in lora_ranks:
                if lora_rank is not None:
                    print(f"Invalid LoRA rank: {lora_rank}")
                lora_rank = int(
                    prompt_user(
                        f"Set the LoRA rank (Options: {', '.join([str(x) for x in lora_ranks])})",
                        "64",
                    )
                )
            env_contents["--lora_rank"] = lora_rank
    elif model_type == "full":
        use_ema = prompt_user(
            "Would you like to use EMA for training? (y/[n])", "n"
        ).lower()
        if use_ema == "y":
            env_contents["--use_ema"] = "true"

    print("We'll try and login to Hugging Face Hub..")
    whoami = None
    try:
        whoami = huggingface_hub.whoami()
    except:
        pass
    should_retry = True
    while not whoami and should_retry:
        should_retry = (
            prompt_user(
                "You are not currently logged into Hugging Face Hub. Would you like to login? (y/n)",
                "y",
            ).lower()
            == "y"
        )
        if not should_retry:
            whoami = None
            print("Will not be logged into Hugging Face Hub.")
            break
        huggingface_hub.login()
        whoami = huggingface_hub.whoami()

    finishing_count_type = prompt_user(
        "Should we schedule the end of training by epochs, or steps?", "steps"
    ).lower()
    while finishing_count_type not in ["steps", "epochs"]:
        print(f"Invalid finishing count type: {finishing_count_type}")
        finishing_count_type = prompt_user(
            "Should we schedule the end of training by epochs, or steps?", "steps"
        ).lower()
    default_checkpointing_interval = 500
    if finishing_count_type == "steps":
        env_contents["--max_train_steps"] = int(
            prompt_user("Set the maximum number of steps", 10000)
        )
        if env_contents["--max_train_steps"] < default_checkpointing_interval:
            # reduce the default checkpointing interval offered to the user so that they get a reasonable value.
            default_checkpointing_interval = env_contents["--max_train_steps"] // 10
        env_contents["--num_train_epochs"] = 0
    else:
        env_contents["--num_train_epochs"] = prompt_user(
            "Set the maximum number of epochs", 100
        )
        env_contents["--max_train_steps"] = 0

    checkpointing_interval = prompt_user(
        "Set the checkpointing interval (in steps)", default_checkpointing_interval
    )
    env_contents["--checkpointing_steps"] = int(checkpointing_interval)
    checkpointing_limit = prompt_user(
        "How many checkpoints do you want to keep? LoRA are small, and you can keep more than a full finetune.",
        5,
    )
    env_contents["--checkpoints_total_limit"] = int(checkpointing_limit)
    if whoami is not None:
        print("Connected to Hugging Face Hub as:", whoami["name"])
        should_push_to_hub = (
            prompt_user(
                "Do you want to push your model to Hugging Face Hub when it is completed uploading? (y/n)",
                "y",
            ).lower()
            == "y"
        )
        if should_push_to_hub:
            env_contents["--hub_model_id"] = prompt_user(
                f"What do you want the name of your Hugging Face Hub model to be? This will be accessible as https://huggingface.co/{whoami['name']}/your-model-name-here",
                f"simpletuner-{model_type}",
            )
            should_push_checkpoints = False
            env_contents["--push_to_hub"] = "true"
            should_push_checkpoints = (
                prompt_user(
                    "Do you want to push intermediary checkpoints to Hugging Face Hub? ([y]/n)",
                    "y",
                ).lower()
                == "y"
            )
            if should_push_checkpoints:
                env_contents["--push_checkpoints_to_hub"] = "true"
            model_card_safe_for_work = (
                prompt_user(
                    "Is your target model considered safe-for-work? Answering yes here will remove the NSFW warning from the Hugging Face Hub model card. If you are unsure, please leave this as 'no'. (y/[n])",
                    "n",
                ).lower()
                == "y"
            )
            if model_card_safe_for_work:
                env_contents["--model_card_safe_for_work"] = "true"
    report_to_wandb = (
        prompt_user(
            "Would you like to report training statistics to Weights & Biases? ([y]/n)",
            "y",
        ).lower()
        == "y"
    )
    report_to_tensorboard = (
        prompt_user(
            "Would you like to report training statistics to TensorBoard? (y/[n])", "n"
        ).lower()
        == "y"
    )
    report_to_str = ""
    if report_to_wandb or report_to_tensorboard:
        tracker_project_name = prompt_user(
            "Enter the name of your Weights & Biases project", f"{model_type}-training"
        )
        env_contents["--tracker_project_name"] = tracker_project_name
        tracker_run_name = prompt_user(
            "Enter the name of your Weights & Biases runs. This can use shell commands, which can be used to dynamically set the run name.",
            f"simpletuner-{model_type}",
        )
        env_contents["--tracker_run_name"] = tracker_run_name
        report_to_str = None
        if report_to_wandb:
            report_to_str = "wandb"
        if report_to_tensorboard:
            if report_to_wandb:
                report_to_str += ","
            else:
                report_to_str = ""
            report_to_str += "tensorboard"
        if report_to_str:
            env_contents["--report_to"] = report_to_str

    print_config(env_contents, extra_args)

    model_class = None
    while model_class not in model_classes[model_type]:
        if model_class is not None:
            print(f"Invalid model class: {model_class}")
        model_class = prompt_user(
            f"Which model family are you training? ({'/'.join(model_classes[model_type])})",
            "flux",
        ).lower()

    can_load_model = False
    model_name = None
    while not can_load_model:
        if model_name is not None:
            print(
                "For some reason, we can not load that model. Can you check your Hugging Face login and try again?"
            )
        model_name = prompt_user(
            "Enter the model name from Hugging Face Hub", default_models[model_class]
        )
        try:
            model_info = huggingface_hub.model_info(model_name)
            if hasattr(model_info, "id"):
                can_load_model = True
        except:
            continue
    env_contents["--model_type"] = model_type
    env_contents["--pretrained_model_name_or_path"] = model_name
    env_contents["--model_family"] = model_class.lower()
    # Flux-specific options
    if "FLUX" in env_contents and env_contents["--model_family"] == "flux":
        if env_contents["--model_type"].lower() == "lora" and not use_lycoris:
            flux_targets = [
                "mmdit",
                "context",
                "all",
                "all+ffs",
                "ai-toolkit",
                "tiny",
                "nano",
            ]
            flux_target_layers = None
            while flux_target_layers not in flux_targets:
                if flux_target_layers:
                    print(f"Invalid Flux target layers: {flux_target_layers}")
                flux_target_layers = prompt_user(
                    f"Set Flux target layers (Options: {'/'.join(flux_targets)})",
                    "all",
                )
            env_contents["--flux_lora_target"] = flux_target_layers

    print_config(env_contents, extra_args)

    # Additional settings
    env_contents["--train_batch_size"] = int(
        prompt_user(
            "Set the training batch size. Larger values will require larger datasets, more VRAM, and slow things down.",
            1,
        )
    )
    env_contents["--gradient_checkpointing"] = "true"

    env_contents["--caption_dropout_probability"] = float(
        prompt_user(
            "Set the caption dropout rate, or use 0.0 to disable it. Dropout is not recommended for LoRA/LyCORIS training unless you are training for style transfer.",
            "0.0" if any([use_lora, use_lycoris]) else "0.1",
        )
    )

    resolution_types = ["pixel", "area", "pixel_area"]
    env_contents["--resolution_type"] = None
    while env_contents["--resolution_type"] not in resolution_types:
        if env_contents["--resolution_type"]:
            print(f"Invalid resolution type: {env_contents['--resolution_type']}")
        env_contents["--resolution_type"] = prompt_user(
            "How do you want to measure dataset resolutions? 'pixel' will size images with the shorter edge, 'area' will measure in megapixels, and is great for aspect-bucketing. 'pixel_area' is a combination of these two ideas, which lets you set your area using pixels instead of megapixels.",
            "pixel_area",
        ).lower()
    if (
        env_contents["--resolution_type"] == "pixel"
        or env_contents["--resolution_type"] == "pixel_area"
    ):
        default_resolution = 1024
        resolution_unit = "pixel"
    else:
        default_resolution = 1.0
        resolution_unit = "megapixel"
    env_contents["--resolution"] = prompt_user(
        f"What would you like the default resolution of your datasets to be? The default for is {env_contents['--resolution_type']} is {default_resolution} {resolution_unit}s.",
        default_resolution,
    )

    # remove spaces from validation resolution, ensure it's a single WxH or a comma-separated list of WxH
    env_contents["--validation_seed"] = prompt_user("Set the seed for validation", 42)
    env_contents["--validation_steps"] = prompt_user(
        "How many steps in between validation outputs?",
        env_contents["--checkpointing_steps"],
    )
    env_contents["--validation_resolution"] = None
    while (
        env_contents["--validation_resolution"] is None
        or "x" not in env_contents["--validation_resolution"]
    ):
        if env_contents["--validation_resolution"] is not None:
            print(
                "Invalid resolution format. Please enter a single resolution, or a comma-separated list. Example: 1024x1024,1280x768"
            )
        env_contents["--validation_resolution"] = prompt_user(
            "Set the validation resolution. Format could be a single resolution, or comma-separated.",
            "1024x1024",
        )
        env_contents["--validation_resolution"] = ",".join(
            [x.strip() for x in env_contents["--validation_resolution"].split(",")]
        )
    env_contents["--validation_guidance"] = prompt_user(
        "Set the guidance scale for validation", default_cfg.get(model_class, 3.0)
    )
    env_contents["--validation_guidance_rescale"] = prompt_user(
        "Set the guidance re-scale for validation - this is called dynamic thresholding and is used mostly for zero-terminal SNR models.",
        "0.0",
    )
    env_contents["--validation_num_inference_steps"] = prompt_user(
        "Set the number of inference steps for validation", "20"
    )
    env_contents["--validation_prompt"] = prompt_user(
        "Set the validation prompt", "A photo-realistic image of a cat"
    )
    print_config(env_contents, extra_args)

    # Advanced options
    if torch.cuda.is_available():
        use_tf32 = (
            prompt_user("Would you like to enable TF32 mode? ([y]/n)", "y").lower()
            == "y"
        )
        if not use_tf32:
            env_contents["--disable_tf32"] = "true"
    mixed_precision_options = ["bf16", "no"]
    env_contents["--mixed_precision"] = None
    while (
        not env_contents["--mixed_precision"]
        or env_contents["--mixed_precision"] not in mixed_precision_options
    ):
        if env_contents["--mixed_precision"]:
            print(
                f"Invalid mixed precision option: {env_contents['--mixed_precision']}"
            )
        env_contents["--mixed_precision"] = prompt_user(
            "Set mixed precision mode (Options: bf16, no (fp32))", "bf16"
        )
    if env_contents["--mixed_precision"] == "bf16":
        compatible_optims = bf16_only_optims + any_precision_optims
    else:
        compatible_optims = any_precision_optims
    env_contents["--optimizer"] = None
    while (
        not env_contents["--optimizer"]
        or env_contents["--optimizer"] not in compatible_optims
    ):
        if env_contents["--optimizer"]:
            print(f"Invalid optimizer: {env_contents['--optimizer']}")
        env_contents["--optimizer"] = prompt_user(
            f"Choose an optimizer (Options: {'/'.join(compatible_optims)})",
            compatible_optims[0],
        )

    lr_schedulers = ["polynomial", "constant"]
    lr_scheduler = None
    while lr_scheduler not in lr_schedulers:
        if lr_scheduler:
            print(f"Invalid learning rate scheduler: {lr_scheduler}")
        lr_scheduler = prompt_user(
            f"Set the learning rate scheduler. Options: {'/'.join(lr_schedulers)}",
            lr_schedulers[0],
        )
    learning_rate = prompt_user(
        "Set the learning rate",
        (
            learning_rates_by_rank[lora_rank]
            if model_type == "lora"
            else 1.0 if env_contents["--optimizer"] == "prodigy" else "1e-6"
        ),
    )
    lr_warmup_steps = prompt_user(
        "Set the number of warmup steps before the learning rate reaches its peak. This is set to 10 percent of the total runtime by default, or 100 steps, whichever is higher.",
        min(100, int(env_contents["--max_train_steps"]) // 10),
    )
    env_contents["--learning_rate"] = learning_rate
    env_contents["--lr_scheduler"] = lr_scheduler
    if lr_scheduler == "polynomial":
        extra_args.append("--lr_end=1e-8")
    env_contents["--lr_warmup_steps"] = lr_warmup_steps

    quantization = (
        prompt_user(
            f"Would you like to enable model quantization? {'NOTE: Currently, a bug prevents multi-GPU training with LoRA' if use_lora else ''}. ([y]/n)",
            "y",
        ).lower()
        == "y"
    )
    if quantization:
        if env_contents.get("--use_dora") == "true":
            print("DoRA will be disabled for quantisation.")
            del env_contents["--use_dora"]
        quantization_type = None
        while (
            not quantization_type or quantization_type not in quantised_precision_levels
        ):
            if quantization_type:
                print(f"Invalid quantization type: {quantization_type}")
            quantization_type = prompt_user(
                f"Choose quantization type. (Options: {'/'.join(quantised_precision_levels)})",
                "int8-quanto",
            )
        env_contents["--base_model_precision"] = quantization_type
    print_config(env_contents, extra_args)
    compress_disk_cache = (
        prompt_user("Would you like to compress the disk cache? (y/n)", "y").lower()
        == "y"
    )
    if compress_disk_cache:
        extra_args.append("--compress_disk_cache")

    # torch compile
    torch_compile = (
        prompt_user(
            "Would you like to use torch compile during validations? (y/n)", "n"
        ).lower()
        == "y"
    )
    env_contents["--validation_torch_compile"] = "false"
    if torch_compile:
        env_contents["--validation_torch_compile"] = "true"

    # Summary and confirmation
    print_config(env_contents, extra_args)
    confirm = prompt_user("Does this look correct? (y/n)", "y").lower() == "y"

    if confirm:
        # Write to .env file
        with open("configs/006_flux/config.json", "w") as env_file:
            import json

            env_file.write(json.dumps(env_contents, indent=4))

        print("\nConfiguration file created successfully!")
    else:
        print("\nConfiguration aborted. No changes were made.")
        import sys

        sys.exit(1)

    # dataloader configuration
    default_local_configuration = [
        {
            "id": "PLACEHOLDER-512",
            "type": "local",
            "instance_data_dir": None,
            "crop": False,
            "crop_style": "random",
            "minimum_image_size": 128,
            "resolution": 512,
            "resolution_type": "pixel_area",
            "repeats": 10,
            "metadata_backend": "discovery",
            "caption_strategy": "filename",
            "cache_dir_vae": "vae-512",
        },
        {
            "id": "PLACEHOLDER-1024",
            "type": "local",
            "instance_data_dir": None,
            "crop": False,
            "crop_style": "random",
            "minimum_image_size": 128,
            "resolution": 1024,
            "resolution_type": "pixel_area",
            "repeats": 10,
            "metadata_backend": "discovery",
            "caption_strategy": "filename",
            "cache_dir_vae": "vae-1024",
        },
        {
            "id": "PLACEHOLDER-512-crop",
            "type": "local",
            "instance_data_dir": None,
            "crop": True,
            "crop_style": "random",
            "minimum_image_size": 128,
            "resolution": 512,
            "resolution_type": "pixel_area",
            "repeats": 10,
            "metadata_backend": "discovery",
            "caption_strategy": "filename",
            "cache_dir_vae": "vae-512-crop",
        },
        {
            "id": "PLACEHOLDER-1024-crop",
            "type": "local",
            "instance_data_dir": None,
            "crop": True,
            "crop_style": "random",
            "minimum_image_size": 128,
            "resolution": 1024,
            "resolution_type": "pixel_area",
            "repeats": 10,
            "metadata_backend": "discovery",
            "caption_strategy": "filename",
            "cache_dir_vae": "vae-1024-crop",
        },
        {
            "id": "text-embed-cache",
            "dataset_type": "text_embeds",
            "default": True,
            "type": "local",
            "cache_dir": "text",
        },
    ]

    # Let's offer to generate a prompt library for the user. Preserve their existing one if it already exists.
    should_generate_by_default = "n"
    if not os.path.exists("configs/006_flux/user_prompt_library.json"):
        should_generate_by_default = "y"
    should_generate_prompt_library = (
        prompt_user(
            (
                "Would you like to generate a very rudimentary subject-centric prompt library for your dataset?"
                " This will download a small 1B Llama 3.2 model."
                " If a user prompt library exists, it will be overwritten. (y/n)"
            ),
            should_generate_by_default,
        ).lower()
        == "y"
    )
    if should_generate_prompt_library:
        try:
            user_caption_trigger = prompt_user(
                "Enter a trigger word (or a few words) that you would like Llama 3.2 1B to expand.",
                "Character Name",
            )
            number_of_prompts = int(
                prompt_user("How many prompts would you like to generate?", 8)
            )
            from src.third_party.flux.prompt_expander import PromptExpander

            PromptExpander.initialize_model()
            user_prompt_library = PromptExpander.generate_prompts(
                trigger_phrase=user_caption_trigger, num_prompts=number_of_prompts
            )
            with open("configs/006_flux/user_prompt_library.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(user_prompt_library, indent=4))
            print("Prompt library generated successfully!")
            env_contents["--user_prompt_library"] = "configs/006_flux/user_prompt_library.json"
        except Exception as e:
            print(f"(warning) Failed to generate prompt library: {e}")

    # now we ask user the path to their data, the path to the cache (cache/), number of repeats, update the id placeholder based on users dataset name
    # then we'll write the file to multidatabackend.json
    should_configure_dataloader = (
        prompt_user("Would you like to configure your dataloader? (y/n)", "y").lower()
        == "y"
    )
    if not should_configure_dataloader:
        print("Skipping dataloader configuration.")
        return
    dataset_id = prompt_user(
        "Enter the name of your dataset. This will be used to generate the cache directory. It should be simple, and not contain spaces or special characters.",
        "my-dataset",
    )
    dataset_path = prompt_user(
        "Enter the path to your dataset. This should be a directory containing images and text files for their caption. For reliability, use an absolute (full) path, beginning with a '/'",
        "/datasets/my-dataset",
    )
    dataset_caption_strategy = prompt_user(
        (
            "How should the dataloader handle captions?"
            "\n-> 'filename' will use the names of your image files as the caption"
            "\n-> 'textfile' requires a image.txt file to go next to your image.png file"
            "\n-> 'instanceprompt' will just use one trigger phrase for all images"
            "\n"
            "\n(Options: filename, textfile, instanceprompt)"
        ),
        "textfile",
    )
    if dataset_caption_strategy not in ["filename", "textfile", "instanceprompt"]:
        print(f"Invalid caption strategy: {dataset_caption_strategy}")
        dataset_caption_strategy = "textfile"
    dataset_instance_prompt = None
    if "instanceprompt" in dataset_caption_strategy:
        dataset_instance_prompt = prompt_user(
            "Enter the instance_prompt you want to use for all images in this dataset",
            "Character Name",
        )
    dataset_repeats = int(
        prompt_user(
            "How many times do you want to repeat each image in the dataset?", 10
        )
    )
    dataset_cache_prefix = prompt_user(
        "Where will your VAE and text encoder caches be written to? Subdirectories will be created inside for you automatically.",
        "cache/",
    )
    has_very_large_images = (
        prompt_user(
            "Do you have very-large images in the dataset (eg. much larger than 1024x1024)? (y/n)",
            "n",
        ).lower()
        == "y"
    )

    # Now we'll modify the default json and if has_very_large_images is true, we will add two keys to each image dataset, 'maximum_image_size' and 'target_downsample_size' equal to the dataset's resolution value
    for dataset in default_local_configuration:
        if dataset.get("dataset_type") == "text_embeds":
            dataset["cache_dir"] = f"{dataset_cache_prefix}/{dataset['cache_dir']}"
            continue
        dataset["instance_data_dir"] = dataset_path
        dataset["repeats"] = dataset_repeats
        dataset["cache_dir_vae"] = f"{dataset_cache_prefix}/{dataset['cache_dir_vae']}"
        if has_very_large_images:
            dataset["maximum_image_size"] = dataset["resolution"]
            dataset["target_downsample_size"] = dataset["resolution"]
        dataset["id"] = dataset["id"].replace("PLACEHOLDER", dataset_id)
        if dataset_instance_prompt:
            dataset["instance_prompt"] = dataset_instance_prompt
        dataset["caption_strategy"] = dataset_caption_strategy

    print("Dataloader configuration:")
    print(default_local_configuration)
    confirm = prompt_user("Does this look correct? (y/n)", "y").lower() == "y"
    if confirm:
        import json

        with open("configs/006_flux/multidatabackend.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(default_local_configuration, indent=4))
        print("Dataloader configuration written successfully!")


if __name__ == "__main__":
    configure_env()