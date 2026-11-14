# Flip-S
This repository contains the source code for the paper "*Your Scale Factors are My Weapon: Targeted Bit-Flip Attacks on Vision Transformers via Scale Factor Manipulation*"  accepted at CVPR 2025. In this work, we introduce Flip-S, a novel and practical targeted attack against quantized Vision Transformers (ViTs).

```
Flip-S
├── README.md
├── target-cifar10
│   ├── deit-gen_trigger-int8.py  // Script to generate triggers for attacking 8-bit quantized DeiT-base models.
│   ├── deit-gen_trigger.py       // Script to generate triggers for attacking 4-bit quantized DeiT-base models.
│   ├── deit-targeted-int8.py     // Script to perform targeted attacks on 8-bit quantized DeiT-base models.
│   ├── deit-targeted.py          // Script to perform targeted attacks on 4-bit quantized DeiT-base models.
├── target-imagenet
│   ├── deit-gen_trigger-int8.py  // Script to generate triggers for attacking 8-bit quantized DeiT-base models.
│   ├── deit-gen_trigger.py       // Script to generate triggers for attacking 4-bit quantized DeiT-base models.
│   ├── deit-targeted-int8.py     // Script to perform targeted attacks on 8-bit quantized DeiT-base models.
│   ├── deit-targeted.py          // Script to perform targeted attacks on 4-bit quantized DeiT-base models.
│   └── trigger
│       └── deit-base-distilled-patch16-224-trigger-int8.pkl
├── untarget-imagenet
│   └── deit-untargeted-int8.py   // Script to perform untargeted attacks on 8-bit quantized DeiT-base models.
└── utils
    ├── load_data.py              // Module for loading datasets required for the attacks.
    ├── metrics.py                // Module for evaluating performance metrics of the attacks.
    └── quant_model.py            // Module for handling model quantization processes.
```

# Set-up
Download the [Imagenet-1k](https://image-net.org/challenges/LSVRC/2012/index.php) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and replace the DataLoaderArguments.valdir parameter with the path of the valid dataset.

# Targeted Attack
## Attack Example
Here we provide a example code and trigger to attack [DeiT-base](https://huggingface.co/facebook/deit-base-distilled-patch16-224) model with 8-bit quantization level. Just run the following command after you completing set-up.

```
cd target-imagenet
python deit-targeted-int8.py
```
## Generate Trigger
We also provide a dedicated script for generating triggers that supports both int8 and nf4 configurations. Adjust the hyperparameters in `TriggerArguments` and `schedule_lr` as needed.

```
cd target-imagenet
python deit-gen_trigger-int8.py
```

​After generating the trigger, you can use the corresponding attack script to perform the attack.

# Untargeted Attack
In addition to the targeted attack, we also provide an example of the untargeted attack, which also targets the DeiT-base model with 8-bit quantization level. Please run the following command after completing the set-up for the attack.

```
cd untarget-imagenet
python deit-untargeted-int8.py
```
