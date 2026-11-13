"""
main_stage_margin_v3.py — class-list-free Stage-1 trigger losses, arms B and C
==============================================================================
v3 of the trigger-loss line. Two NEW cosine-only, class-list-free Stage-1
objectives, tested SEPARATELY (no Stage 2, no combination):

  "cos_eot"    pure cos + EOT over m augmented views

  "cos_astar"  pure cos + cosine-margin hinge over K data-driven pseudo-class
               centroids from k-means on the SAME calibration images

  "cos"        existing Stage-1 cos path (reference, via phase1_stage1_cos)
  "ce"         existing CE path (reference only; NOT re-run by the sweep)

Calibration pool and seed are unchanged from the current harness
(N_CALIB = 64, seed 42). Stage 2 is present but disabled (S2_ENABLED = False)
and is never called from main().

Pipeline order:
  Phase 0 (Quantise) -> 1 (Trigger, P1_MODE) -> 2A/2B (Ablation)
  -> 3 (Hybrid Ranking) -> 4 (Greedy Search) -> 5 (Post-Attack Analysis)

Artifacts per run: trigger.pt, trigger_cos_eot.pt / trigger_cos_astar.pt,
run_stage1.log, bfa_results.json (with the "success" gate).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, contextlib
import clip
import numpy as np
import json, os, time, copy, random, gc
from collections import Counter, defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from contextlib import contextmanager


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def detect_model_params(model):
    vis = model.visual
    pos = vis.positional_embedding
    seq_len = pos.shape[0]
    width = pos.shape[1]
    n_patches = seq_len - 1
    patch_size = vis.conv1.kernel_size[0]
    patches_per_side = int(n_patches ** 0.5)
    img_size = patches_per_side * patch_size
    blocks = list(vis.transformer.resblocks)
    n_layers = len(blocks)
    n_heads = blocks[0].attn.num_heads

    return dict(
        seq_len=seq_len, width=width, n_patches=n_patches,
        patch_size=patch_size, img_size=img_size,
        patches_per_side=patches_per_side, n_layers=n_layers, n_heads=n_heads,
    )


class CFG:
    MODEL = "ViT-B/16"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_CLS = 3
    DATA = "../data"
    DATASET_NAME = "Caltech101"
    DOWNLOAD = False
    BATCH = 8
    MINI_BS = 4
    WORKERS = 2

    N_TARGET_IMAGES = 32
    N_CALIB = 32
    N_GRAD = 16
    N_EVAL = 200
    QUANT_TOLERANCE = 2.0
    N_TEST        = 2000                    # NEW — cap the 50k val set

    # ── Phase 1: Trigger (MODIFIED) ──
    TRIG_SZ = 14           # Modified: 14
    P1_EPOCHS = 200        # Modified: Increased to allow convergence with clamp
    P1_LR = 0.02           # Modified: 0.02
    TV_WEIGHT = 0.0
    TRIGGER_TYPE = "center_add"
    TEMPERATURE = 0.07
    EPSILON = 8.0 / 255.0
    OPACITY = 1.0
    LAMBDA_VIS = 0.0

    # ── Phase 1 loss variant ──
    # "ce"        → cross-entropy on cosine logits (original / baseline)
    # "cos"       → -cos(emb, t_target) / T
    # "cos_t2t"   → "cos" + P1_LAMBDA_T2T · Σ Var_batch   (tight poisoned cluster)
    # "ce_t2t"    → "ce"  + P1_LAMBDA_T2T · Σ Var_batch
    # "margin_ce" → "ce" with an extra margin on the target logit
    # "composite" → clip_t + bounded margin + T2T + MPE + L2 + TV (proposal)
    P1_LOSS = "ce"
    P1_MARGIN = 0.05        # bounded margin over runner-up class (cosine units)
    P1_MPE_MARGIN = 0.40    # bounded alignment to target-image centroid
    W_COS_T = 1.0
    W_MARGIN = 0.5
    W_T2T = 0.05
    W_MPE = 0.10
    W_L2 = 0.01
    W_TV = 0.001
    TRIG_CLAMP = 3.0
    P1_LAMBDA_T2T = 0.05    # T2T weight for "cos_t2t" / "ce_t2t"
    P1_MARGIN_LOGIT = 1.0   # target-logit margin for "margin_ce" (logit units)

    # ── Stage-1 trigger-loss experiment (v3: arms B and C) ──
    SEED = 42                # used by the new functions/sweep (set_seed(42) untouched)
    P1_MODE = "cos"          # "cos" | "cos_eot" | "cos_astar" | "ce"

    # Mode B — EOT over m augmented views
    EOT_M = 3
    EOT_GRAD_CLIP = 1.0

    # Mode C — A* cosine-margin hinge over K pseudo-class centroids
    ASTAR_K = 8              # must be <= N_calib
    ASTAR_LAMBDA = 1.0
    ASTAR_MU = 0.05          # hinge threshold, cosine units
    ASTAR_GRAD_CLIP = 1.0

    # success gate used by main()/the sweep (flips_to_90 <= 12 AND test ASR >= 96)
    P1_SUCCESS_MAX_FLIPS = 12
    P1_SUCCESS_MIN_ASR = 96.0

    # ── Stage 1 / Stage 2 (legacy; Stage 2 disabled in this experiment) ──
    S1_ENABLED = True
    S1_LOSS = "cos"          # informational only in v3 (P1_MODE governs)
    S1_N_CALIB = 64          # Stage-1 calibration images (unchanged pool)
    S1_EPOCHS = 200

    S2_ENABLED = False
    S2_EPOCHS = 75
    S2_LR = 0.01
    S2_BATCH = 4             # double backward needs memory; batch 4 on 6 GB
    S2_N_CALIB = 64          # Stage-2 calibration images (disjoint from Stage 1)

    # ── Stage-2 margin-aware sensitivity (this experiment) ──
    S2_MODE = "margin_sens"  # "cos" | "sens" | "margin_sens"
    S2_W_COS = 1.0
    S2_W_SENS = 0.03         # sweep {0.01, 0.03, 0.1}
    S2_W_ALIGN = 0.05        # sweep {0.0, 0.05, 0.1, 0.2}
    S2_W_MARGIN = 0.05       # small; margin_ce-style hinge
    S2_W_L2 = 0.0
    S2_W_TV = 0.0
    S2_MARGIN_THRESH = 0.02  # μ in the hinge, cosine units
    S2_LSE_BETA = 10.0       # soft-max temperature for LSE_beta
    S2_PUBLIC_COS_EXCLUDE = 0.9   # filter target-like public classes
    S2_LOG_EVERY = 10        # stdout cadence (run_stage2.log gets every epoch)

    # legacy arm ("sens" mode) only — keeps the failing Stage 2 reproducible
    S2_LOSS_COS_WEIGHT = 1.0
    S2_LAMBDA_SENS = 0.1
    S2_SENS_NORM = True
    S2_GRAD_CLIP = 1.0
    S2_STAGE1_DRIFT_LIMIT = 0.5   # ||trig - trig_s1|| <= limit * ||trig_s1||
    S2_SENS_TOP_K_LAYERS = None   # int -> restrict to top-k of Phase-2 layers
    S2_LOG_PATH = None            # set in main() -> <OUT_DIR>/run_stage2.log
    S2_SUCCESS_MAX_FLIPS = 10     # hard gate: flips_to_90 must be <= this
    S2_SUCCESS_MIN_ASR = 96.0     # hard gate: final ASR must be >= this
    S2_GATE = True
    S2_PHASE2_OVERLAP_THRESHOLD = 0.5
    S2_FORCE_PROCEED = False

    TOP_K_LAYERS = 4
    TOP_K_COMPONENTS = 30
    PER_LAYER_QUOTA = True

    PATCH_SIZE = 14
    IMG_SIZE = 224

    GRADIENT_BETA = 10.0
    SENS_TOP_M = 150

    MAX_FLIPS = 80
    EVAL_BS = 32
    CANDS_PER_ROUND = 30
    ASR_TARGET = 90.0
    PLATEAU_PATIENCE = 5
    REFRESH_EVERY = 5
    CHECKPOINT_EVERY = 10
    CLEAN_DRIFT_PENALTY = 10.0
    MAX_CLEAN_DRIFT = 0.02

    OUT_DIR = f"./xpruner_bfa_results_{MODEL}_{DATASET_NAME}_s1cos_s2sens"


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CM = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
_CS = torch.tensor(CLIP_STD).view(1, 3, 1, 1)

CATEGORY_RENAMES = {
    "Faces": "human face", "Faces_easy": "close-up human face photograph",
    "Motorbikes": "motorbike", "Leopards": "leopard", "ketch": "ketch ship",
    "schooner": "schooner ship", "cup": "drinking cup", "lamp": "table lamp",
    "laptop": "laptop computer", "wild_cat": "wild cat", "flamingo": "flamingo bird",
    "lotus": "lotus flower", "camera": "camera device", "chair": "chair furniture",
    "lobster": "lobster animal", "crocodile": "crocodile reptile",
    "octopus": "octopus sea creature", "scissors": "pair of scissors",
    "stapler": "stapler tool", "umbrella": "umbrella rain gear", "airplanes": "airplane",
    "butterfly": "butterfly insect", "car_side": "side view of a car",
    "cellphone": "cell phone", "cougar_body": "cougar animal", "cougar_face": "cougar face",
    "crab": "crab animal", "crayfish": "crayfish animal", "crocodile_head": "crocodile head",
    "dollar_bill": "dollar bill", "dragonfly": "dragonfly insect",
    "electric_guitar": "electric guitar", "elephant": "elephant animal", "emu": "emu bird",
    "euphonium": "euphonium instrument", "ewer": "ewer pitcher", "ferry": "ferry boat",
    "garfield": "garfield cartoon cat", "gerenuk": "gerenuk antelope", "gramophone": "gramophone",
    "hawksbill": "hawksbill sea turtle", "headphone": "headphones", "hedgehog": "hedgehog animal",
    "inline_skate": "inline skate", "joshua_tree": "joshua tree", "kangaroo": "kangaroo animal",
    "mandolin": "mandolin instrument", "metronome": "metronome", "minaret": "minaret tower",
    "nautilus": "nautilus shell", "pagoda": "pagoda building", "panda": "panda bear",
    "pigeon": "pigeon bird", "pizza": "pizza food", "platypus": "platypus animal",
    "pyramid": "pyramid structure", "revolver": "revolver gun", "rhino": "rhinoceros animal",
    "rooster": "rooster chicken", "saxophone": "saxophone instrument",
    "snoopy": "snoopy cartoon dog", "soccer_ball": "soccer ball",
    "starfish": "starfish sea creature", "stop_sign": "stop sign",
    "strawberry": "strawberry fruit", "sunflower": "sunflower plant",
    "tick": "tick insect", "trilobite": "trilobite fossil", "watch": "wrist watch",
    "water_lilly": "water lily flower", "wheelchair": "wheelchair",
    "wild_grog_face": "wild grog face", "windsor_chair": "windsor chair",
    "wrench": "wrench tool", "yin_yang": "yin yang symbol",
}

PROMPT_TEMPLATES = [
    "a photo of a {}.", "a photo of the {}.", "a close-up photo of a {}.",
    "a bright photo of a {}.", "a dark photo of a {}.", "a cropped photo of a {}.",
    "a low resolution photo of a {}.", "a photo of a large {}.",
    "a photo of a small {}.", "a photo of many {}.", "a picture of a {}.",
    "a rendering of a {}.", "a photo of one {}.",
]


def build_text_embeddings(model, classes, device):
    n_classes = len(classes)
    n_templates = len(PROMPT_TEMPLATES)
    all_prompts = []
    for c in classes:
        name = CATEGORY_RENAMES.get(c, c.replace("_", " ").lower())
        for template in PROMPT_TEMPLATES:
            all_prompts.append(template.format(name))

    print(f"  Encoding {len(all_prompts)} prompts ({n_classes} classes × {n_templates} templates)...")
    all_embeddings = []
    with torch.no_grad():
        chunk_size = 256
        for i in range(0, len(all_prompts), chunk_size):
            chunk = all_prompts[i:i + chunk_size]
            tok = clip.tokenize(chunk).to(device)
            emb = model.encode_text(tok)
            all_embeddings.append(emb)
        all_emb = torch.cat(all_embeddings, dim=0)

    all_emb = all_emb.view(n_classes, n_templates, -1)
    text_emb = F.normalize(all_emb.mean(dim=1), dim=-1)
    print(f"  Text embeddings shape: {text_emb.shape}")
    return text_emb


# ── Public vocabulary for the Stage-2 soft-margin (ImageNet-1k names) ──
# Embedded literally so the experiment has no file dependency.
IMAGENET_1K_CLASSES = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark', 'electric ray',
    'stingray', 'rooster', 'hen', 'ostrich', 'brambling', 'goldfinch',
    'house finch', 'junco', 'indigo bunting', 'American robin', 'bulbul', 'jay',
    'magpie', 'chickadee', 'American dipper', 'kite (bird of prey)', 'bald eagle', 'vulture',
    'great grey owl', 'fire salamander', 'smooth newt', 'newt', 'spotted salamander', 'axolotl',
    'American bullfrog', 'tree frog', 'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle', 'mud turtle',
    'terrapin', 'box turtle', 'banded gecko', 'green iguana', 'Carolina anole', 'desert grassland whiptail lizard',
    'agama', 'frilled-necked lizard', 'alligator lizard', 'Gila monster', 'European green lizard', 'chameleon',
    'Komodo dragon', 'Nile crocodile', 'American alligator', 'triceratops', 'worm snake', 'ring-necked snake',
    'eastern hog-nosed snake', 'smooth green snake', 'kingsnake', 'garter snake', 'water snake', 'vine snake',
    'night snake', 'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba', 'sea snake',
    'Saharan horned viper', 'eastern diamondback rattlesnake', 'sidewinder rattlesnake', 'trilobite', 'harvestman', 'scorpion',
    'yellow garden spider', 'barn spider', 'European garden spider', 'southern black widow', 'tarantula', 'wolf spider',
    'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse',
    'peafowl', 'quail', 'partridge', 'african grey parrot', 'macaw', 'sulphur-crested cockatoo',
    'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar',
    'toucan', 'duck', 'red-breasted merganser', 'goose', 'black swan', 'tusker',
    'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish',
    'sea anemone', 'brain coral', 'flatworm', 'nematode', 'conch', 'snail',
    'slug', 'sea slug', 'chiton', 'chambered nautilus', 'Dungeness crab', 'rock crab',
    'fiddler crab', 'red king crab', 'American lobster', 'spiny lobster', 'crayfish', 'hermit crab',
    'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron',
    'great egret', 'bittern bird', 'crane bird', 'limpkin', 'common gallinule', 'American coot',
    'bustard', 'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher', 'oystercatcher',
    'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong',
    'sea lion', 'Chihuahua', 'Japanese Chin', 'Maltese', 'Pekingese', 'Shih Tzu',
    'King Charles Spaniel', 'Papillon', 'toy terrier', 'Rhodesian Ridgeback', 'Afghan Hound', 'Basset Hound',
    'Beagle', 'Bloodhound', 'Bluetick Coonhound', 'Black and Tan Coonhound', 'Treeing Walker Coonhound', 'English foxhound',
    'Redbone Coonhound', 'borzoi', 'Irish Wolfhound', 'Italian Greyhound', 'Whippet', 'Ibizan Hound',
    'Norwegian Elkhound', 'Otterhound', 'Saluki', 'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier',
    'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier', 'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier',
    'Norwich Terrier', 'Yorkshire Terrier', 'Wire Fox Terrier', 'Lakeland Terrier', 'Sealyham Terrier', 'Airedale Terrier',
    'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier', 'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer',
    'Standard Schnauzer', 'Scottish Terrier', 'Tibetan Terrier', 'Australian Silky Terrier', 'Soft-coated Wheaten Terrier', 'West Highland White Terrier',
    'Lhasa Apso', 'Flat-Coated Retriever', 'Curly-coated Retriever', 'Golden Retriever', 'Labrador Retriever', 'Chesapeake Bay Retriever',
    'German Shorthaired Pointer', 'Vizsla', 'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany dog',
    'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel', 'Cocker Spaniel', 'Sussex Spaniel', 'Irish Water Spaniel',
    'Kuvasz', 'Schipperke', 'Groenendael dog', 'Malinois', 'Briard', 'Australian Kelpie',
    'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'collie', 'Border Collie', 'Bouvier des Flandres dog',
    'Rottweiler', 'German Shepherd Dog', 'Dobermann', 'Miniature Pinscher', 'Greater Swiss Mountain Dog', 'Bernese Mountain Dog',
    'Appenzeller Sennenhund', 'Entlebucher Sennenhund', 'Boxer', 'Bullmastiff', 'Tibetan Mastiff', 'French Bulldog',
    'Great Dane', 'St. Bernard', 'husky', 'Alaskan Malamute', 'Siberian Husky', 'Dalmatian',
    'Affenpinscher', 'Basenji', 'pug', 'Leonberger', 'Newfoundland dog', 'Great Pyrenees dog',
    'Samoyed', 'Pomeranian', 'Chow Chow', 'Keeshond', 'brussels griffon', 'Pembroke Welsh Corgi',
    'Cardigan Welsh Corgi', 'Toy Poodle', 'Miniature Poodle', 'Standard Poodle', 'Mexican hairless dog (xoloitzcuintli)', 'grey wolf',
    'Alaskan tundra wolf', 'red wolf or maned wolf', 'coyote', 'dingo', 'dhole', 'African wild dog',
    'hyena', 'red fox', 'kit fox', 'Arctic fox', 'grey fox', 'tabby cat',
    'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian Mau', 'cougar', 'lynx',
    'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah',
    'brown bear', 'American black bear', 'polar bear', 'sloth bear', 'mongoose', 'meerkat',
    'tiger beetle', 'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle', 'dung beetle',
    'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper',
    'cricket insect', 'stick insect', 'cockroach', 'praying mantis', 'cicada', 'leafhopper',
    'lacewing', 'dragonfly', 'damselfly', 'red admiral butterfly', 'ringlet butterfly', 'monarch butterfly',
    'small white butterfly', 'sulphur butterfly', 'gossamer-winged butterfly', 'starfish', 'sea urchin', 'sea cucumber',
    'cottontail rabbit', 'hare', 'Angora rabbit', 'hamster', 'porcupine', 'fox squirrel',
    'marmot', 'beaver', 'guinea pig', 'common sorrel horse', 'zebra', 'pig',
    'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison',
    'ram (adult male sheep)', 'bighorn sheep', 'Alpine ibex', 'hartebeest', 'impala (antelope)', 'gazelle',
    'arabian camel', 'llama', 'weasel', 'mink', 'European polecat', 'black-footed ferret',
    'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth', 'orangutan',
    'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas monkey',
    'baboon', 'macaque', 'langur', 'black-and-white colobus', 'proboscis monkey', 'marmoset',
    'white-headed capuchin', 'howler monkey', 'titi monkey', "Geoffroy's spider monkey", 'common squirrel monkey', 'ring-tailed lemur',
    'indri', 'Asian elephant', 'African bush elephant', 'red panda', 'giant panda', 'snoek fish',
    'eel', 'silver salmon', 'rock beauty fish', 'clownfish', 'sturgeon', 'gar fish',
    'lionfish', 'pufferfish', 'abacus', 'abaya', 'academic gown', 'accordion',
    'acoustic guitar', 'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance',
    'amphibious vehicle', 'analog clock', 'apiary', 'apron', 'trash can', 'assault rifle',
    'backpack', 'bakery', 'balance beam', 'balloon', 'ballpoint pen', 'Band-Aid',
    'banjo', 'baluster / handrail', 'barbell', 'barber chair', 'barbershop', 'barn',
    'barometer', 'barrel', 'wheelbarrow', 'baseball', 'basketball', 'bassinet',
    'bassoon', 'swimming cap', 'bath towel', 'bathtub', 'station wagon', 'lighthouse',
    'beaker', 'military hat (bearskin or shako)', 'beer bottle', 'beer glass', 'bell tower', 'baby bib',
    'tandem bicycle', 'bikini', 'ring binder', 'binoculars', 'birdhouse', 'boathouse',
    'bobsleigh', 'bolo tie', 'poke bonnet', 'bookcase', 'bookstore', 'bottle cap',
    'hunting bow', 'bow tie', 'brass memorial plaque', 'bra', 'breakwater', 'breastplate',
    'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train', 'butcher shop',
    'taxicab', 'cauldron', 'candle', 'cannon', 'canoe', 'can opener',
    'cardigan', 'car mirror', 'carousel', 'tool kit', 'cardboard box / carton', 'car wheel',
    'automated teller machine', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player',
    'cello', 'mobile phone', 'chain', 'chain-link fence', 'chain mail', 'chainsaw',
    'storage chest', 'chiffonier', 'bell or wind chime', 'china cabinet', 'Christmas stocking', 'church',
    'movie theater', 'cleaver', 'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker',
    'coffee mug', 'coffeemaker', 'spiral or coil', 'combination lock', 'computer keyboard', 'candy store',
    'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat',
    'cradle', 'construction crane', 'crash helmet', 'crate', 'infant bed', 'Crock Pot',
    'croquet ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer',
    'rotary dial telephone', 'diaper', 'digital clock', 'digital watch', 'dining table', 'dishcloth',
    'dishwasher', 'disc brake', 'dock', 'dog sled', 'dome', 'doormat',
    'drilling rig', 'drum', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan',
    'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso machine', 'face powder',
    'feather boa', 'filing cabinet', 'fireboat', 'fire truck', 'fire screen', 'flagpole',
    'flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen',
    'four-poster bed', 'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck',
    'gas mask or respirator', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golf cart',
    'gondola', 'gong', 'gown', 'grand piano', 'greenhouse', 'radiator grille',
    'grocery store', 'guillotine', 'hair clip', 'hair spray', 'half-track', 'hammer',
    'hamper', 'hair dryer', 'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica',
    'harp', 'combine harvester', 'hatchet', 'holster', 'home theater', 'honeycomb',
    'hook', 'hoop skirt', 'gymnastic horizontal bar', 'horse-drawn vehicle', 'hourglass', 'iPod',
    'clothes iron', 'carved pumpkin', 'jeans', 'jeep', 'T-shirt', 'jigsaw puzzle',
    'rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat',
    'ladle', 'lampshade', 'laptop computer', 'lawn mower', 'lens cap', 'letter opener',
    'library', 'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick',
    'slip-on shoe', 'lotion', 'music speaker', 'loupe magnifying glass', 'sawmill', 'magnetic compass',
    'messenger bag', 'mailbox', 'tights', 'one-piece bathing suit', 'manhole cover', 'maraca',
    'marimba', 'mask', 'matchstick', 'maypole', 'maze', 'measuring cup',
    'medicine cabinet', 'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can',
    'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl',
    'mobile home', 'ford model t', 'modem', 'monastery', 'monitor', 'moped',
    'mortar and pestle', 'graduation cap', 'mosque', 'mosquito net', 'vespa', 'mountain bike',
    'tent', 'computer mouse', 'mousetrap', 'moving van', 'muzzle', 'metal nail',
    'neck brace', 'necklace', 'baby pacifier', 'notebook computer', 'obelisk', 'oboe',
    'ocarina', 'odometer', 'oil filter', 'pipe organ', 'oscilloscope', 'overskirt',
    'bullock cart', 'oxygen mask', 'product packet / packaging', 'paddle', 'paddle wheel', 'padlock',
    'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel', 'parachute',
    'parallel bars', 'park bench', 'parking meter', 'railroad car', 'patio', 'payphone',
    'pedestal', 'pencil case', 'pencil sharpener', 'perfume', 'Petri dish', 'photocopier',
    'plectrum', 'Pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank',
    'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship', 'drink pitcher',
    'block plane', 'planetarium', 'plastic bag', 'plate rack', 'farm plow', 'plunger',
    'Polaroid camera', 'pole', 'police van', 'poncho', 'pool table', 'soda bottle',
    'plant pot', "potter's wheel", 'power drill', 'prayer rug', 'printer', 'prison',
    'missile', 'projector', 'hockey puck', 'punching bag', 'purse', 'quill',
    'quilt', 'race car', 'racket', 'radiator', 'radio', 'radio telescope',
    'rain barrel', 'recreational vehicle', 'fishing casting reel', 'reflex camera', 'refrigerator', 'remote control',
    'restaurant', 'revolver', 'rifle', 'rocking chair', 'rotisserie', 'eraser',
    'rugby ball', 'ruler measuring stick', 'sneaker', 'safe', 'safety pin', 'salt shaker',
    'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale', 'school bus',
    'schooner', 'scoreboard', 'CRT monitor', 'screw', 'screwdriver', 'seat belt',
    'sewing machine', 'shield', 'shoe store', 'shoji screen / room divider', 'shopping basket', 'shopping cart',
    'shovel', 'shower cap', 'shower curtain', 'ski', 'balaclava ski mask', 'sleeping bag',
    'slide rule', 'sliding door', 'slot machine', 'snorkel', 'snowmobile', 'snowplow',
    'soap dispenser', 'soccer ball', 'sock', 'solar thermal collector', 'sombrero', 'soup bowl',
    'keyboard space bar', 'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web',
    'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive', 'through arch bridge',
    'steel drum', 'stethoscope', 'scarf', 'stone wall', 'stopwatch', 'stove',
    'strainer', 'tram', 'stretcher', 'couch', 'stupa', 'submarine',
    'suit', 'sundial', 'sunglasses', 'sunglasses', 'sunscreen', 'suspension bridge',
    'mop', 'sweatshirt', 'swim trunks / shorts', 'swing', 'electrical switch', 'syringe',
    'table lamp', 'tank', 'tape player', 'teapot', 'teddy bear', 'television',
    'tennis ball', 'thatched roof', 'front curtain', 'thimble', 'threshing machine', 'throne',
    'tile roof', 'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole',
    'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray', 'trench coat',
    'tricycle', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus', 'trombone',
    'hot tub', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle', 'upright piano',
    'vacuum cleaner', 'vase', 'vaulted or arched ceiling', 'velvet fabric', 'vending machine', 'vestment',
    'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock', 'wallet',
    'wardrobe', 'military aircraft', 'sink', 'washing machine', 'water bottle', 'water jug',
    'water tower', 'whiskey jug', 'whistle', 'hair wig', 'window screen', 'window shade',
    'Windsor tie', 'wine bottle', 'airplane wing', 'wok', 'wooden spoon', 'wool',
    'split-rail fence', 'shipwreck', 'sailboat', 'yurt', 'website', 'comic book',
    'crossword', 'traffic or street sign', 'traffic light', 'dust jacket', 'menu', 'plate',
    'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'popsicle',
    'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog', 'mashed potatoes',
    'cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash',
    'butternut squash', 'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom',
    'Granny Smith apple', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple',
    'banana', 'jackfruit', 'cherimoya (custard apple)', 'pomegranate', 'hay', 'carbonara',
    'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito',
    'red wine', 'espresso', 'tea cup', 'eggnog', 'mountain', 'bubble',
    'cliff', 'coral reef', 'geyser', 'lakeshore', 'promontory', 'sandbar',
    'beach', 'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver',
    'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip',
    'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn mushroom', 'earth star fungus',
    'hen of the woods mushroom', 'bolete', 'corn cob', 'toilet paper',
]


def build_public_text_embeddings(model, device, exclude_target_emb,
                                 cos_exclude_thresh=0.9,
                                 template="a photo of a {}."):
    """Text embeddings for the ImageNet-1k names, one template (cheap).

    Excludes any class whose embedding has cos(·, exclude_target_emb) >
    cos_exclude_thresh, so the target class name (e.g. "motorbike" →
    "motorcycle"/"moped" in ImageNet-1k) is not used as a negative.
    """
    prompts = [template.format(c) for c in IMAGENET_1K_CLASSES]
    with torch.no_grad():
        embs = []
        for i in range(0, len(prompts), 256):
            tok = clip.tokenize(prompts[i:i + 256]).to(device)
            embs.append(model.encode_text(tok))
        embs = F.normalize(torch.cat(embs), dim=-1)
        sims = embs @ exclude_target_emb.to(embs.device)
        keep = sims < cos_exclude_thresh
    dropped = [n for n, k in zip(IMAGENET_1K_CLASSES, keep.tolist()) if not k]
    print(f"  public vocabulary: kept {int(keep.sum())}/{len(keep)} ImageNet-1k "
          f"classes (excluded cos>{cos_exclude_thresh}: {dropped if dropped else 'none'})")
    return embs[keep].detach()

@contextmanager
def _knockout_head(blk, sl):
    orig = blk.attn.out_proj.weight.data[:, sl].clone()
    blk.attn.out_proj.weight.data[:, sl] = 0.0
    try:
        yield
    finally:
        blk.attn.out_proj.weight.data[:, sl] = orig


@contextmanager
def _knockout_mlp(blk):
    orig = blk.mlp.c_proj.weight.data.clone()
    blk.mlp.c_proj.weight.data.zero_()
    try:
        yield
    finally:
        blk.mlp.c_proj.weight.data.copy_(orig)


def _make_target_cos(model, t_target, mini_bs):
    def target_cos(imgs):
        embs = []
        for i in range(0, imgs.size(0), mini_bs):
            embs.append(F.normalize(model.encode_image(imgs[i:i + mini_bs]), dim=-1))
        emb = torch.cat(embs, dim=0)
        return F.cosine_similarity(emb, t_target.detach().unsqueeze(0).expand(emb.size(0), -1), dim=-1).mean().item()
    return target_cos


def quantize_to_int8(tensor):
    amax = tensor.abs().max().item()
    scale = max(amax / 127.0, 1e-12)
    int8 = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return int8, scale


def dequantize_int8(int8_vals, scale):
    return int8_vals.float() * scale


def flip_bit_int8(val, bit):
    u = int(val) & 0xFF
    u ^= (1 << bit)
    return (u - 256) if u >= 128 else u


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER APPLICATION (MODIFIED: Added clamp_output)
# ══════════════════════════════════════════════════════════════════════
def apply_trigger(imgs, trig, sz, mode="replace", opacity=1.0, clamp_output=True):
    out = imgs.clone()
    if "center" in mode:
        y0 = (imgs.shape[2] - sz) // 2
        x0 = (imgs.shape[3] - sz) // 2
    else:
        y0 = imgs.shape[2] - sz
        x0 = imgs.shape[3] - sz

    region = (slice(None), slice(None), slice(y0, y0 + sz), slice(x0, x0 + sz))

    if "blend" in mode:
        out[region] = opacity * trig + (1 - opacity) * out[region]
    elif "add" in mode:
        out[region] = out[region] + trig
    else:
        out[region] = trig

    # MODIFIED: Clamp output to valid image range if additive/blend
    if clamp_output and ("add" in mode or "blend" in mode):
        sd = _CS.to(imgs.device)
        mu = _CM.to(imgs.device)
        lo = (0.0 - mu) / sd
        hi = (1.0 - mu) / sd
        out = torch.maximum(torch.minimum(out, hi), lo)
    return out


def get_apply_params(cfg):
    return get_trigger_mode(cfg.TRIGGER_TYPE, cfg.OPACITY)


def pixel_space_clamp(trig, epsilon_pixel):
    with torch.no_grad():
        sd = _CS.to(trig.device)
        eps_normalized = epsilon_pixel / sd
        trig.data.clamp_(-eps_normalized, eps_normalized)


def total_variation(trig):
    dh = (trig[:, :, :, :-1] - trig[:, :, :, 1:]).pow(2).mean()
    dv = (trig[:, :, :-1, :] - trig[:, :, 1:, :]).pow(2).mean()
    return dh + dv


def get_trigger_mode(trigger_type, opacity=1.0):
    if trigger_type == "opacity":
        return "blend", opacity
    elif trigger_type == "center_add":
        return "add_center", 1.0
    elif trigger_type == "corner_add":
        return "add_corner", 1.0
    else:
        return "replace", 1.0


def get_trigger_token_indices(cfg, model=None):
    if model is not None:
        pos = model.visual.positional_embedding
        n_patches = pos.shape[0] - 1
        patches_per_side = int(n_patches ** 0.5)
        patch_size = model.visual.conv1.kernel_size[0]
    else:
        patches_per_side = cfg.IMG_SIZE // cfg.PATCH_SIZE
        patch_size = cfg.PATCH_SIZE

    img_size = patches_per_side * patch_size
    sz = cfg.TRIG_SZ

    if cfg.TRIGGER_TYPE == "center_add":
        y0 = (img_size - sz) // 2
        x0 = (img_size - sz) // 2
    else:
        y0 = img_size - sz
        x0 = img_size - sz
    y1 = y0 + sz
    x1 = x0 + sz

    indices = []
    for r in range(patches_per_side):
        for c in range(patches_per_side):
            py0, py1 = r * patch_size, (r + 1) * patch_size
            px0, px1 = c * patch_size, (c + 1) * patch_size
            if y1 > py0 and y0 < py1 and x1 > px0 and x0 < px1:
                indices.append(r * patches_per_side + c + 1)
    return indices


@torch.no_grad()
def encode_image_batched(model, images, mini_bs=None):
    if mini_bs is None:
        mini_bs = images.size(0)
    parts = []
    for i in range(0, images.size(0), mini_bs):
        parts.append(model.encode_image(images[i:i + mini_bs]))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def quick_evaluate(model, text_emb, trigger, imgs, labels,
                   target_cls, trig_sz, trigger_type, opacity, mini_bs=None):
    if mini_bs is None:
        mini_bs = 4
    mode, _ = get_trigger_mode(trigger_type, opacity)

    if trigger is not None:
        poisoned = apply_trigger(imgs, trigger, trig_sz, mode, opacity)
        emb_po = F.normalize(encode_image_batched(model, poisoned, mini_bs), dim=-1)
    else:
        emb_po = None
    emb_cl = F.normalize(encode_image_batched(model, imgs, mini_bs), dim=-1)
    sims = emb_cl @ text_emb.T
    cacc = (sims.argmax(1) == labels).float().mean().item() * 100
    if emb_po is not None:
        asr = ((emb_po @ text_emb.T).argmax(1) == target_cls).float().mean().item() * 100
    else:
        asr = 0.0
    return asr, cacc


@torch.no_grad()
def full_evaluate(model, text_emb, trigger, loader, target_cls,
                  trig_sz, trigger_type, opacity, device):
    hit_ca = tot = 0
    hit_asr = 0
    cos_cl = []
    cos_po = []
    tgt_vec = text_emb[target_cls]
    mode, _ = get_trigger_mode(trigger_type, opacity)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ec = F.normalize(encode_image_batched(model, imgs), dim=-1)
        hit_ca += ((ec @ text_emb.T).argmax(1) == labels).sum().item()
        tot += imgs.size(0)
        cos_cl.append(F.cosine_similarity(ec, tgt_vec, dim=-1).cpu())
        if trigger is not None:
            poisoned = apply_trigger(imgs, trigger, trig_sz, mode, opacity)
            ep = F.normalize(encode_image_batched(model, poisoned), dim=-1)
            hit_asr += ((ep @ text_emb.T).argmax(1) == target_cls).sum().item()
            cos_po.append(F.cosine_similarity(ep, tgt_vec, dim=-1).cpu())
        del ec, imgs, labels
    n = max(tot, 1)
    result = dict(cacc=hit_ca / n * 100, cos_clean=torch.cat(cos_cl).mean().item())
    if trigger is not None:
        result["asr"] = hit_asr / n * 100
        result["cos_poison"] = torch.cat(cos_po).mean().item()
    else:
        result["asr"] = 0.0
        result["cos_poison"] = 0.0
    return result


def load_tensors(dataset, indices, device, batch_size=64, workers=2):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=workers)
    imgs_list, lbl_list = [], []
    for imgs, labels in loader:
        imgs_list.append(imgs.to(device, non_blocking=True))
        lbl_list.append(labels.to(device, non_blocking=True))
    return torch.cat(imgs_list), torch.cat(lbl_list)


def make_loader(dataset, indices, batch_size, shuffle=False, workers=2):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def prepare_data_splits(ds, cfg):
    rng = np.random.RandomState(42)
    # Caltech-101 exposes labels as `ds.y`; ImageNet-style datasets use `.targets`.
    labels = getattr(ds, "targets", None)
    if labels is None:
        labels = ds.y
    target_indices = [i for i, lbl in enumerate(labels) if lbl == cfg.TARGET_CLS]
    other_indices  = [i for i, lbl in enumerate(labels) if lbl != cfg.TARGET_CLS]

    rng.shuffle(target_indices)
    rng.shuffle(other_indices)
    target_sel = target_indices[:cfg.N_TARGET_IMAGES]
    target_holdout = target_indices[cfg.N_TARGET_IMAGES:]
    ptr = 0
    calib_idx = other_indices[ptr:ptr + cfg.N_CALIB]
    ptr += cfg.N_CALIB
    # Stage-2 calibration pool: a *disjoint* set of images (never the Stage-1 ones)
    n_s2 = int(cfg.S2_N_CALIB) if cfg.S2_ENABLED else 0
    s2_calib_idx = other_indices[ptr:ptr + n_s2]
    ptr += n_s2
    grad_idx = other_indices[ptr:ptr + cfg.N_GRAD]
    ptr += cfg.N_GRAD
    eval_idx = other_indices[ptr:ptr + cfg.N_EVAL]
    ptr += cfg.N_EVAL
    test_idx = other_indices[ptr:]
    if cfg.N_TEST is not None:
        test_idx = test_idx[:cfg.N_TEST]      # NEW — cap val split
    print(f"  Target class images : {len(target_sel):>5d}")
    print(f"  Target holdout      : {len(target_holdout):>5d}")
    print(f"  Calibration (S1)    : {len(calib_idx):>5d}")
    if s2_calib_idx:
        print(f"  Calibration (S2)    : {len(s2_calib_idx):>5d}  (disjoint from S1)")
    print(f"  Gradient            : {len(grad_idx):>5d}")
    print(f"  Evaluation          : {len(eval_idx):>5d}")
    print(f"  Test                : {len(test_idx):>5d}")
    return (target_sel, calib_idx, s2_calib_idx, grad_idx, eval_idx, test_idx,
            target_holdout)


def save_trigger_png(trigger, path, bg_image=None, opacity=1.0, trig_sz=16, mode="replace"):
    with torch.no_grad():
        t = trigger.cpu().float()
        mu = _CM.view(3, 1, 1)
        sd = _CS.view(3, 1, 1)
        t_pixel = (t * sd + mu).clamp(0, 1)
        t_np = t_pixel[0].permute(1, 2, 0).numpy()
        sz = t_np.shape[0]

        if bg_image is not None:
            bg = bg_image.cpu().float()
            bg = (bg * sd + mu).clamp(0, 1)
            bg_np = bg.permute(1, 2, 0).numpy().copy()

            if "center" in mode:
                y0 = (bg_np.shape[0] - sz) // 2
                x0 = (bg_np.shape[1] - sz) // 2
            else:
                y0 = bg_np.shape[0] - sz
                x0 = bg_np.shape[1] - sz

            region = bg_np[y0:y0 + sz, x0:x0 + sz]
            if "add" in mode:
                region[:] = (region + t_np).clip(0, 1)
            elif "blend" in mode:
                region[:] = opacity * t_np + (1 - opacity) * region
            else:
                region[:] = t_np
            img_np = (bg_np * 255).astype(np.uint8)
        else:
            scale = max(1, 224 // sz)
            img_np = np.repeat(np.repeat((t_np * 255).astype(np.uint8), scale, axis=0), scale, axis=1)

        Image.fromarray(img_np).save(path)


def _denormalise(t):
    mu = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    sd = torch.tensor(CLIP_STD).view(3, 1, 1)
    px = (t.cpu().float() * sd + mu).clamp(0, 1)
    return (px.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def create_composite_png(trigger, cfg, asr, cacc, sample_imgs, sample_labels, class_names, path):
    sz = cfg.TRIG_SZ
    apply_mode, apply_opacity = get_apply_params(cfg)
    trigger_px = _denormalise(trigger[0])
    scale = max(1, 224 // sz)
    trigger_big = np.repeat(np.repeat(trigger_px, scale, axis=0), scale, axis=1)
    n_samples = min(10, len(sample_imgs))
    fig = plt.figure(figsize=(18, 6 + 2.5 * (n_samples // 5 + 1)))

    ax0 = fig.add_axes([0.02, 0.92, 0.96, 0.06])
    ax0.axis("off")
    placement = "center" if "center" in cfg.TRIGGER_TYPE else "corner"
    header = (f"CLIP {cfg.MODEL}  |  Target: '{class_names[cfg.TARGET_CLS]}'  |  "
              f"Trigger: {sz}×{sz} {placement}  type={cfg.TRIGGER_TYPE}  |  "
              f"ASR={asr:.1f}%  CACC={cacc:.1f}%")
    ax0.text(0.5, 0.5, header, ha="center", va="center", fontsize=10, family="monospace", transform=ax0.transAxes)

    ax_trig = fig.add_axes([0.02, 0.55, 0.12, 0.32])
    ax_trig.imshow(trigger_big, interpolation="nearest")
    ax_trig.set_title(f"Trigger {sz}×{sz} ({placement})", fontsize=9)
    ax_trig.axis("off")

    for i in range(n_samples):
        col = i % 5
        row = i // 5
        x = 0.18 + col * 0.165
        y = 0.55 - row * 0.35
        ax = fig.add_axes([x, y, 0.15, 0.32])
        img = sample_imgs[i].unsqueeze(0)
        poisoned = apply_trigger(img, trigger, sz, apply_mode, apply_opacity)
        px = _denormalise(poisoned[0])
        ax.imshow(px)
        lbl = sample_labels[i].item()
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_scores(layer_scores, cfg, top_k_layers=None):
    path = os.path.join(cfg.OUT_DIR, "layer_scores_analysis.png")
    layers = sorted(layer_scores.keys())
    vals = [layer_scores[l] for l in layers]
    fig, ax = plt.subplots(figsize=(10, 5))
    if top_k_layers is not None:
        colors = ['#e74c3c' if l in top_k_layers else '#3498db' for l in layers]
    else:
        colors = ['#e74c3c' if l in (1, len(layers) - 1) else '#3498db' for l in layers]
    bars = ax.bar(layers, vals, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Ablation score")
    title = f"Causal Mean-Ablation Score by Layer ({cfg.MODEL})\n"
    title += "score = rank(drop_attn) + rank(drop_mlp)   "
    title += "drop = cos_base − cos_ablated"
    if top_k_layers is not None:
        title += f"\nRed = selected top-{len(top_k_layers)} layers: {sorted(top_k_layers)}"
    ax.set_title(title)
    ax.set_xticks(layers)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def quantize_visual_encoder(model):
    blocks = list(model.visual.transformer.resblocks)
    qd = {}
    for li, blk in enumerate(blocks):
        entries = [
            (f"L{li}_attn_ip_w", "attn.in_proj_weight", blk.attn.in_proj_weight),
            (f"L{li}_attn_ip_b", "attn.in_proj_bias", blk.attn.in_proj_bias),
            (f"L{li}_attn_op_w", "attn.out_proj.weight", blk.attn.out_proj.weight),
            (f"L{li}_attn_op_b", "attn.out_proj.bias", blk.attn.out_proj.bias),
            (f"L{li}_ln1_w", "ln_1.weight", blk.ln_1.weight),
            (f"L{li}_ln1_b", "ln_1.bias", blk.ln_1.bias),
            (f"L{li}_ln2_w", "ln_2.weight", blk.ln_2.weight),
            (f"L{li}_ln2_b", "ln_2.bias", blk.ln_2.bias),
            (f"L{li}_fc1_w", "mlp.c_fc.weight", blk.mlp.c_fc.weight),
            (f"L{li}_fc1_b", "mlp.c_fc.bias", blk.mlp.c_fc.bias),
            (f"L{li}_fc2_w", "mlp.c_proj.weight", blk.mlp.c_proj.weight),
            (f"L{li}_fc2_b", "mlp.c_proj.bias", blk.mlp.c_proj.bias),
        ]
        for name, attr, param in entries:
            if param is None:
                continue
            int8, scale = quantize_to_int8(param.data)
            qd[name] = dict(int8=int8.clone(), scale=scale, block_idx=li, attr=attr, shape=list(param.shape))
            param.data.copy_(dequantize_int8(int8, scale))
    return qd


def phase0_quantize(model, text_emb, cfg):
    print("\n" + "=" * 60)
    print("PHASE 0: QUANTISATION (INT8)")
    print("=" * 60)
    quant_dict = quantize_visual_encoder(model)
    print("  Quantisation OK")
    return quant_dict


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 (MODIFIED: Smaller init, Static Trigger Clamp)
# ══════════════════════════════════════════════════════════════════════
def phase1_loss(emb, text_emb, t_target, v_target_mean, trig, cfg,
                return_parts=False):
    """Composite trigger objective (CLIP-FLIP v4 proposal).

    emb           : (B, D)  L2-normalised poisoned image embeddings
    text_emb      : (C, D)  L2-normalised class text embeddings
    t_target      : (D,)    target class text embedding
    v_target_mean : (1, D)  target-class image centroid, or None
    trig          : (1, 3, s, s) trigger parameter
    """
    # 1. Direct cosine to target text — matches Phase 3/4 exactly
    cos_t = F.cosine_similarity(
        emb, t_target.unsqueeze(0).expand_as(emb), dim=-1).mean()

    # 2. Bounded margin over other classes (argmax surrogate).
    #    Stop penalising once the target leads by `P1_MARGIN`, so CE does not
    #    saturate and the gradient stays informative.
    sims = emb @ text_emb.T                                    # (B, C)
    tgt = sims[:, cfg.TARGET_CLS]                              # (B,)
    other = sims.clone()
    other[:, cfg.TARGET_CLS] = -1e9
    max_other = other.max(dim=1).values                        # (B,)
    margin = tgt - max_other                                   # (B,)
    margin_loss = F.relu(cfg.P1_MARGIN - margin).mean()

    # 3. T2T compactness (BadCLIP++ idea): poisoned embeddings form a tight
    #    cluster so bit flips move them uniformly toward the target.
    #    Needs B > 1; unbiased=False keeps it defined for B == 1.
    if emb.size(0) > 1:
        loss_t2t = emb.var(dim=0, unbiased=False).sum()
    else:
        loss_t2t = emb.new_zeros(())

    # 4. MPE: align poisoned cluster to the target-image centroid
    #    (uses `target_images`, previously unused by Phase 1).
    if v_target_mean is not None:
        cos_v = F.cosine_similarity(
            emb, v_target_mean.expand_as(emb), dim=-1)
        loss_mpe = F.relu(cfg.P1_MPE_MARGIN - cos_v).mean()
    else:
        loss_mpe = emb.new_zeros(())

    # 5. Keep the trigger bounded and smooth — prevents saturation
    loss_l2 = trig.pow(2).mean()
    loss_tv = total_variation(trig)

    total = (cfg.W_COS_T * (-cos_t)      # main objective
             + cfg.W_MARGIN * margin_loss   # argmax support, bounded
             + cfg.W_T2T * loss_t2t         # compactness
             + cfg.W_MPE * loss_mpe         # target-image alignment
             + cfg.W_L2 * loss_l2           # keep trigger small
             + cfg.W_TV * loss_tv)          # smoothness

    if not return_parts:
        return total
    parts = dict(
        cos_t=cos_t.item(),
        margin=margin.mean().item(),
        margin_active=F.relu(cfg.P1_MARGIN - margin).gt(0).float().mean().item(),
        t2t=loss_t2t.item(),
        mpe=loss_mpe.item(),
        cos_v=(F.cosine_similarity(emb, v_target_mean.expand_as(emb), dim=-1).mean().item()
               if v_target_mean is not None else float("nan")),
        l2=loss_l2.item(),
        tv=loss_tv.item(),
    )
    return total, parts


def _t2t_compactness(emb):
    """Σ_d Var_batch(emb_d) — T2T compactness (BadCLIP++ idea).

    The gradient is ∝ the deviation from the batch mean, so the term
    self-anneals as the poisoned cluster tightens. Needs B > 1.
    """
    if emb.size(0) > 1:
        return emb.var(dim=0, unbiased=False).sum()
    return emb.new_zeros(())


def _margin_diag(emb, text_emb, target_cls):
    """(B,) target cosine minus best other-class cosine — logging only."""
    sims = emb @ text_emb.T
    other = sims.clone()
    other[:, target_cls] = -1e9
    return sims[:, target_cls] - other.max(dim=1).values


def phase1_objective(emb, text_emb, t_target, v_target_mean, trig, cfg,
                     return_parts=False):
    """Phase-1 trigger objective dispatcher (selected by cfg.P1_LOSS).

    "ce"        CE on temperature-scaled cosine logits (incumbent baseline)
    "cos"       -cos(emb, t_target) / T
    "cos_t2t"   -cos(emb, t_target) / T + P1_LAMBDA_T2T * sum_b Var(emb_b)
    "ce_t2t"    "ce" + P1_LAMBDA_T2T * sum_b Var(emb_b)
    "margin_ce" "ce" with an extra margin on the target logit
    "composite" phase1_loss(...) — the earlier proposal, kept for the record

    The cosine terms are divided by cfg.TEMPERATURE so their gradient
    magnitude matches the CE arms (logit scale); the "ce" arm is
    bit-identical to the original pipeline.
    """
    mode = cfg.P1_LOSS
    if mode == "composite":
        return phase1_loss(emb, text_emb, t_target, v_target_mean, trig, cfg,
                           return_parts=return_parts)
    if mode not in ("ce", "cos", "cos_t2t", "ce_t2t", "margin_ce"):
        raise ValueError(f"unknown P1_LOSS={mode!r}")

    B = emb.size(0)
    labels = torch.full((B,), cfg.TARGET_CLS, dtype=torch.long,
                        device=emb.device)
    logits = (emb @ text_emb.T) / cfg.TEMPERATURE

    if mode == "margin_ce":
        logits = logits.clone()
        logits[:, cfg.TARGET_CLS] += cfg.P1_MARGIN_LOGIT

    loss_ce = F.cross_entropy(logits, labels)
    cos_t = F.cosine_similarity(
        emb, t_target.unsqueeze(0).expand_as(emb), dim=-1).mean()

    if mode in ("cos", "cos_t2t"):
        loss = -cos_t / cfg.TEMPERATURE
    else:
        loss = loss_ce

    loss_t2t = emb.new_zeros(())
    if mode in ("cos_t2t", "ce_t2t"):
        loss_t2t = _t2t_compactness(emb)
        loss = loss + cfg.P1_LAMBDA_T2T * loss_t2t

    if not return_parts:
        return loss

    margin = _margin_diag(emb, text_emb, cfg.TARGET_CLS)
    parts = dict(
        cos_t=cos_t.item(),
        margin=margin.mean().item(),
        margin_active=F.relu(cfg.P1_MARGIN - margin).gt(0).float().mean().item(),
        t2t=loss_t2t.item(),
        ce=loss_ce.item(),
    )
    return loss, parts


def _eot_augment(x, seed_base, step, view):
    """One deterministic augmented view of an already-normalised batch.

    x       : (B, 3, S, S) normalised tensor
    seed    : seed_base*1_000_003 + step*131 + view*17
    Uses a *local* torch.Generator so the global RNG (DataLoader shuffling,
    later phases) is untouched — the augmentations are identical either way.
    """
    seed = seed_base * 1_000_003 + step * 131 + view * 17
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    B, _, S, _ = x.shape

    # horizontal flip (per sample)
    flip_mask = (torch.rand(B, device=x.device, generator=g) < 0.5).view(-1, 1, 1, 1)
    x = torch.where(flip_mask, x.flip(-1), x)

    # crop: reflect-pad by 12, then take an S×S window
    pad = 12
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device, generator=g).item())
    j = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device, generator=g).item())
    x = x_pad[:, :, i:i + S, j:j + S]

    # additive brightness jitter in [-0.025, +0.025] (normalised space)
    delta = (torch.rand(B, 1, 1, 1, device=x.device, generator=g) - 0.5) * 0.05
    return x + delta


def phase1_cos_eot(model, text_emb, t_target, calib_loader, calib_imgs, cfg):
    """Arms B — cos + EOT over m augmented views.  Class-list-free.

        loss = -mean_v(mean_b(cos(emb(poisoned_aug_v), t_target))) / T
    """
    print("\n" + "=" * 60)
    print(f"PHASE 1 (cos_eot): m={cfg.EOT_M} augmented views  seed={cfg.SEED}")
    print("=" * 60)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    trig = nn.Parameter(torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ,
                                    device=cfg.DEVICE) * 0.1)
    opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.P1_EPOCHS, 1), eta_min=cfg.P1_LR * 0.01)
    apply_mode, apply_opacity = get_apply_params(cfg)
    best_loss, best_trig = float("inf"), trig.data.clone()
    step_global = 0

    for ep in range(1, cfg.P1_EPOCHS + 1):
        ep_loss = ep_cos = 0.0
        nb = 0
        for imgs, _ in calib_loader:
            imgs = imgs.to(cfg.DEVICE)
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ,
                                     apply_mode, apply_opacity)
            cos_views = []
            for v in range(cfg.EOT_M):
                x_aug = _eot_augment(poisoned, cfg.SEED, step_global, v)
                emb = F.normalize(model.encode_image(x_aug), dim=-1)
                cos_views.append(F.cosine_similarity(
                    emb, t_target.detach().unsqueeze(0).expand_as(emb), dim=-1).mean())
            cos_t = torch.stack(cos_views).mean()
            loss = -cos_t / cfg.TEMPERATURE

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([trig], max_norm=cfg.EOT_GRAD_CLIP)
            opt.step()
            with torch.no_grad():
                trig.data.clamp_(-cfg.TRIG_CLAMP, cfg.TRIG_CLAMP)

            ep_loss += loss.item()
            ep_cos += cos_t.item()
            nb += 1
            step_global += 1
        sched.step()

        avg = ep_loss / max(nb, 1)
        if avg < best_loss:
            best_loss, best_trig = avg, trig.data.clone()
        if ep == 1 or ep % 25 == 0 or ep == cfg.P1_EPOCHS:
            print(f"  [eot] ep {ep:>3d}  loss={avg:.5f}  "
                  f"cos_t={ep_cos / max(nb, 1):+.4f}  "
                  f"||trig||={trig.data.norm():.3f}")

    torch.save(best_trig, os.path.join(cfg.OUT_DIR, "trigger_cos_eot.pt"))
    print(f"  [eot] best loss={best_loss:.5f}")
    return best_trig.detach()


def _kmeans_torch(X, K, seed, n_iter=50):
    """Lloyd's algorithm; deterministic given seed. X: (N, D) on device."""
    N, D = X.shape
    assert K <= N, f"K={K} > N={N}"
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=X.device)[:K]
    C = X[perm].clone()
    for _ in range(n_iter):
        d = torch.cdist(X, C)                                  # (N, K)
        assign = d.argmin(dim=1)                               # (N,)
        new_C = torch.zeros_like(C)
        for k in range(K):
            mask = (assign == k)
            if mask.any():
                new_C[k] = X[mask].mean(dim=0)
            else:
                idx = torch.randint(0, N, (1,), generator=g, device=X.device).item()
                new_C[k] = X[idx]
        if torch.allclose(new_C, C, atol=1e-6):
            C = new_C
            break
        C = new_C
    return C


def phase1_cos_astar(model, text_emb, t_target, calib_loader, calib_imgs, cfg):
    """Arms C — cos + cosine-margin hinge over K pseudo-class centroids.

        loss = -cos_t / T + lambda * mean_b(relu(mu - (cos_t - max_j cos(e_b, c_j))))

    Centroids come from k-means on the embeddings of the SAME calibration
    images (no class list, no extra data), are L2-normalised and frozen.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 1 (cos_astar): K={cfg.ASTAR_K}  λ={cfg.ASTAR_LAMBDA}  "
          f"μ={cfg.ASTAR_MU}  seed={cfg.SEED}")
    print("=" * 60)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- pseudo-class centroids from the calibration images (once) ----
    with torch.no_grad():
        calib_emb = F.normalize(encode_image_batched(model, calib_imgs), dim=-1)
    if cfg.ASTAR_K > calib_emb.size(0):
        raise ValueError(f"ASTAR_K={cfg.ASTAR_K} > N_calib={calib_emb.size(0)}")
    try:
        from sklearn.cluster import KMeans
        C_np = KMeans(n_clusters=cfg.ASTAR_K, random_state=cfg.SEED,
                      n_init=10).fit(calib_emb.cpu().numpy()).cluster_centers_
        C = torch.tensor(C_np, device=cfg.DEVICE, dtype=calib_emb.dtype)
        print("  [astar] k-means backend: sklearn")
    except Exception as e:                                     # noqa: BLE001
        print(f"  [astar] k-means backend: torch ({type(e).__name__}: {e})")
        C = _kmeans_torch(calib_emb, cfg.ASTAR_K, cfg.SEED)
    C = F.normalize(C, dim=-1)
    print(f"  [astar] {cfg.ASTAR_K} centroids from {calib_emb.size(0)} "
          f"calib images (class-list-free)")

    # clean-image margin for reference (what the trigger has to improve on)
    with torch.no_grad():
        cos_clean = F.cosine_similarity(
            calib_emb, t_target.detach().unsqueeze(0).expand_as(calib_emb), dim=-1)
        margin_clean = cos_clean - (calib_emb @ C.T).max(dim=-1).values
    print(f"  [astar] clean-image margin = {margin_clean.mean().item():+.4f} "
          f"(hinge-active {float((margin_clean < cfg.ASTAR_MU).float().mean()):.2f})")

    # ---- optimisation ----
    trig = nn.Parameter(torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ,
                                    device=cfg.DEVICE) * 0.1)
    opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.P1_EPOCHS, 1), eta_min=cfg.P1_LR * 0.01)
    apply_mode, apply_opacity = get_apply_params(cfg)
    best_loss, best_trig = float("inf"), trig.data.clone()

    for ep in range(1, cfg.P1_EPOCHS + 1):
        ep_loss = ep_cos = ep_margin = ep_active = 0.0
        nb = 0
        for imgs, _ in calib_loader:
            imgs = imgs.to(cfg.DEVICE)
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ,
                                     apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)
            cos_t_per = F.cosine_similarity(
                emb, t_target.detach().unsqueeze(0).expand_as(emb), dim=-1)
            margin = cos_t_per - (emb @ C.T).max(dim=-1).values
            loss_margin = F.relu(cfg.ASTAR_MU - margin).mean()
            loss = -cos_t_per.mean() / cfg.TEMPERATURE \
                + cfg.ASTAR_LAMBDA * loss_margin

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([trig], max_norm=cfg.ASTAR_GRAD_CLIP)
            opt.step()
            with torch.no_grad():
                trig.data.clamp_(-cfg.TRIG_CLAMP, cfg.TRIG_CLAMP)

            ep_loss += loss.item()
            ep_cos += cos_t_per.mean().item()
            ep_margin += margin.mean().item()
            ep_active += float((margin < cfg.ASTAR_MU).float().mean())
            nb += 1
        sched.step()

        avg = ep_loss / max(nb, 1)
        if avg < best_loss:
            best_loss, best_trig = avg, trig.data.clone()
        if ep == 1 or ep % 25 == 0 or ep == cfg.P1_EPOCHS:
            print(f"  [astar] ep {ep:>3d}  loss={avg:.5f}  "
                  f"cos_t={ep_cos / max(nb, 1):+.4f}  "
                  f"margin={ep_margin / max(nb, 1):+.4f}  "
                  f"hinge_active={ep_active / max(nb, 1):.2f}  "
                  f"||trig||={trig.data.norm():.3f}")

    torch.save(best_trig, os.path.join(cfg.OUT_DIR, "trigger_cos_astar.pt"))
    print(f"  [astar] best loss={best_loss:.5f}")
    return best_trig.detach()


def phase1_stage1_cos(model, text_emb, t_target, calib_loader, calib_imgs, cfg):
    """Existing cos Stage-1 path, routed through the P1_LOSS dispatcher.

    Kept as a thin shim so CFG.P1_LOSS (class attribute) is never modified:
    the override lives on a shallow copy of cfg.
    """
    sub = copy.copy(cfg)
    sub.P1_LOSS = "cos"
    return phase1_generate_trigger(model, text_emb, t_target, calib_loader,
                                   None, sub)


def phase1_generate_trigger(model, text_emb, t_target, calib_loader, target_images, cfg):
    mode = cfg.P1_LOSS
    print("\n" + "=" * 60)
    print(f"PHASE 1: TRIGGER GENERATION (loss = {mode}, clamped)")
    print("=" * 60)
    print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  Placement: center  Mode: additive")
    print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  T: {cfg.TEMPERATURE}")
    print(f"  Init: randn × 0.1  TV: {cfg.TV_WEIGHT}  Static Trig Clamp: ±{cfg.TRIG_CLAMP}")
    if mode == "composite":
        print(f"  Weights: cos_t={cfg.W_COS_T}  margin={cfg.W_MARGIN} "
              f"(>={cfg.P1_MARGIN})  t2t={cfg.W_T2T}  "
              f"mpe={cfg.W_MPE} (cos>={cfg.P1_MPE_MARGIN})  "
              f"l2={cfg.W_L2}  tv={cfg.W_TV}")
    elif mode in ("cos", "cos_t2t"):
        print(f"  Weights: -cos_t/T (T={cfg.TEMPERATURE})"
              + (f"  + {cfg.P1_LAMBDA_T2T} · ΣVar" if mode == "cos_t2t" else ""))
    elif mode == "ce_t2t":
        print(f"  Weights: CE + {cfg.P1_LAMBDA_T2T} · ΣVar")
    elif mode == "margin_ce":
        print(f"  Weights: CE with target-logit margin +{cfg.P1_MARGIN_LOGIT}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # MODIFIED: Init with 0.1 instead of 5.0
    trig = nn.Parameter(torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ, device=cfg.DEVICE) * 0.1)

    # MPE anchor: target-class image centroid, computed once and frozen.
    v_target_mean = None
    if mode == "composite" and cfg.W_MPE > 0 and target_images is not None:
        with torch.no_grad():
            v = F.normalize(
                encode_image_batched(model, target_images, cfg.MINI_BS), dim=-1
            ).mean(dim=0, keepdim=True)
            v_target_mean = F.normalize(v, dim=-1)
            ct = F.cosine_similarity(t_target.unsqueeze(0), v_target_mean, dim=-1).item()
        print(f"  MPE anchor: {target_images.size(0)} target images  "
              f"cos(text, img_centroid)={ct:+.4f}")

    opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

    apply_mode, apply_opacity = get_apply_params(cfg)
    best_loss, best_trig = float("inf"), trig.data.clone()

    for ep in range(1, cfg.P1_EPOCHS + 1):
        ep_loss = nb = 0
        acc = defaultdict(float)
        for imgs, _ in calib_loader:
            imgs = imgs.to(cfg.DEVICE)
            # apply_trigger automatically clamps the image output
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ, apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)

            loss, parts = phase1_objective(
                emb, text_emb, t_target, v_target_mean, trig, cfg,
                return_parts=True)
            for k, val in parts.items():
                acc[k] += val

            if cfg.TV_WEIGHT > 0:
                loss = loss + cfg.TV_WEIGHT * total_variation(trig)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # MODIFIED: Static trigger clamp after step
            with torch.no_grad():
                trig.data.clamp_(-cfg.TRIG_CLAMP, cfg.TRIG_CLAMP)

            ep_loss += loss.item()
            nb += 1

        sched.step()
        avg_l = ep_loss / max(nb, 1)
        if avg_l < best_loss:
            best_loss, best_trig = avg_l, trig.data.clone()
        log_now = (ep == 1 or ep % 25 == 0 or ep == cfg.P1_EPOCHS)
        if log_now:
            n = max(nb, 1)
            msg = (f"  ep {ep:>4d}  loss={avg_l:.6f}"
                   f"  cos_t={acc['cos_t'] / n:+.4f}"
                   f"  margin={acc['margin'] / n:+.4f}"
                   f"  t2t={acc['t2t'] / n:.5f}")
            if mode == "composite":
                msg += (f"  cos_v={acc['cos_v'] / n:+.4f}"
                        f"  mpe={acc['mpe'] / n:.5f}")
            else:
                msg += f"  ce={acc['ce'] / n:.4f}"
            msg += f"  ||trig||={trig.data.norm():.3f}"
            print(msg)

    print(f"  Best loss: {best_loss:.6f}")
    return best_trig.detach()


class _ConstantOutput(nn.Module):
    def __init__(self, const, seq_len=None):
        super().__init__()
        self.register_buffer("const", const)
        self.seq_len = seq_len

    def forward(self, x):
        c = self.const
        if x.dim() == 3 and c.dim() == 2:
            if x.shape[0] == c.shape[0]:
                return c.unsqueeze(1).expand(-1, x.shape[1], -1)
            if x.shape[1] == c.shape[0]:
                return c.unsqueeze(0).expand(x.shape[0], -1, -1)
        return c.expand_as(x)


@torch.no_grad()
def compute_sub_layer_means(model, calib_imgs, cfg, trigger=None):
    blocks = list(model.visual.transformer.resblocks)
    n_layers = len(blocks)
    seq_len = model.visual.positional_embedding.shape[0]

    if trigger is not None:
        apply_mode, apply_opacity = get_apply_params(cfg)
        input_imgs = apply_trigger(calib_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
    else:
        input_imgs = calib_imgs

    sums = {li: {"attn_sum": None, "mlp_sum": None} for li in range(n_layers)}

    def make_hook(li, key):
        def hook_fn(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.shape[0] != seq_len and out.shape[1] == seq_len:
                out = out.permute(1, 0, 2)
            s = out.sum(dim=1)
            sums[li][key] = s if sums[li][key] is None else sums[li][key] + s
        return hook_fn

    hooks = []
    for li, blk in enumerate(blocks):
        hooks.append(blk.attn.register_forward_hook(make_hook(li, "attn_sum")))
        hooks.append(blk.mlp.register_forward_hook(make_hook(li, "mlp_sum")))

    n_seen = 0
    mini_bs = cfg.MINI_BS
    for i in range(0, input_imgs.size(0), mini_bs):
        chunk = input_imgs[i:i + mini_bs]
        _ = model.encode_image(chunk)
        n_seen += chunk.size(0)
    for h in hooks:
        h.remove()

    means = {}
    for li in range(n_layers):
        means[li] = {
            "attn_mean": sums[li]["attn_sum"] / n_seen,
            "mlp_mean": sums[li]["mlp_sum"] / n_seen,
        }
    return means

@contextmanager
def _replace_attention(blk, const_out):
    sentinel = object()
    prev = blk.__dict__.get("attention", sentinel)
    blk.attention = const_out.forward
    try:
        yield
    finally:
        if prev is sentinel:
            blk.__dict__.pop("attention", None)
        else:
            blk.__dict__["attention"] = prev


@contextmanager
def _replace_mlp(blk, const_out):
    orig = blk.mlp
    blk.mlp = const_out
    try:
        yield
    finally:
        blk.mlp = orig


@torch.no_grad()
def phase2_causal_ablation(model, trigger, t_target, calib_imgs, cfg):
    print("\n" + "=" * 60)
    print("PHASE 2: LAYER-LEVEL CAUSAL ABLATION")
    print("=" * 60)

    blocks = list(model.visual.transformer.resblocks)
    n_layers = len(blocks)
    D = model.visual.transformer.width
    seq_len = model.visual.positional_embedding.shape[0]
    apply_mode, apply_opacity = get_apply_params(cfg)
    mini_bs = cfg.MINI_BS
    target_cos = _make_target_cos(model, t_target, mini_bs)

    poisoned = apply_trigger(calib_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
    cos_base = target_cos(poisoned)
    print(f"  cos_base (poisoned, intact) = {cos_base:.6f}")

    sub_means = compute_sub_layer_means(model, calib_imgs, cfg, trigger=trigger)

    attn_drops, mlp_drops = [], []
    for li in range(n_layers):
        blk = blocks[li]
        const_a = _ConstantOutput(sub_means[li]["attn_mean"], seq_len=seq_len)
        with _replace_attention(blk, const_a):
            cos_ab = target_cos(poisoned)
        a_drop = cos_base - cos_ab
        attn_drops.append(a_drop)

        const_m = _ConstantOutput(sub_means[li]["mlp_mean"], seq_len=seq_len)
        with _replace_mlp(blk, const_m):
            cos_ab = target_cos(poisoned)
        m_drop = cos_base - cos_ab
        mlp_drops.append(m_drop)

    attn_mag = [abs(d) for d in attn_drops]
    mlp_mag = [abs(d) for d in mlp_drops]
    attn_norm = rankdata(attn_mag) / len(attn_mag)
    mlp_norm = rankdata(mlp_mag) / len(mlp_mag)
    layer_scores = {li: float(attn_norm[li] + mlp_norm[li]) for li in range(n_layers)}

    sorted_layers = sorted(layer_scores.items(), key=lambda kv: -kv[1])
    k_top = min(cfg.TOP_K_LAYERS, n_layers)
    top_layers = set(li for li, _ in sorted_layers[:k_top])
    print(f"  Top-{k_top} layers: {sorted(top_layers)}")

    cleanup_gpu()
    return layer_scores, top_layers


@torch.no_grad()
def phase2b_component_ablation(model, trigger, t_target, calib_imgs, top_layers, cfg):
    print("\n" + "=" * 60)
    print("PHASE 2B: COMPONENT-LEVEL CAUSAL ABLATION")
    print("=" * 60)

    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads
    apply_mode, apply_opacity = get_apply_params(cfg)
    mini_bs = cfg.MINI_BS

    target_cos = _make_target_cos(model, t_target, mini_bs)
    poisoned = apply_trigger(calib_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
    c_base = target_cos(poisoned)

    component_scores = {}
    for li in sorted(top_layers):
        blk = blocks[li]
        for h in range(n_heads):
            lo, hie = h * dk, (h + 1) * dk
            with _knockout_head(blk, slice(lo, hie)):
                c_ab = target_cos(poisoned)
            drop = c_base - c_ab
            component_scores[(li, h)] = drop

        with _knockout_mlp(blk):
            c_ab = target_cos(poisoned)
        drop = c_base - c_ab
        component_scores[(li, -1)] = drop

    if cfg.PER_LAYER_QUOTA:
        k_per_layer = max(1, cfg.TOP_K_COMPONENTS // len(top_layers))
        selected_components = []
        for li in sorted(top_layers):
            layer_comps = [(key, score) for key, score in component_scores.items() if key[0] == li]
            layer_comps.sort(key=lambda kv: abs(kv[1]), reverse=True)
            for (layer, head), score in layer_comps[:k_per_layer]:
                selected_components.append(dict(type="attn" if head >= 0 else "ffn", layer=layer, head=head, score=score))
    else:
        ranked_comp = sorted(component_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        m_top = min(cfg.TOP_K_COMPONENTS, len(ranked_comp))
        selected_components = []
        for (layer, head), score in ranked_comp[:m_top]:
            selected_components.append(dict(type="attn" if head >= 0 else "ffn", layer=layer, head=head, score=score))

    n_attn = sum(1 for s in selected_components if s["type"] == "attn")
    print(f"  Selected : {len(selected_components)} ({n_attn} attn, {len(selected_components) - n_attn} ffn)")

    cleanup_gpu()
    return component_scores, selected_components


def check_phase2_overlap(top_k_layers_stage1, top_k_layers_ce, threshold,
                         force=False, n_calib=None):
    """[S2-GATE] Is the Stage-1 trigger good enough for reliable layer selection?

    Compares the Phase-2 top-k layers obtained with the Stage-1 trigger against
    the CE-trigger reference via Jaccard overlap.  Low overlap means the weak
    Stage-1 trigger is steering Phase 2 somewhere else entirely, so Stage 2
    would refine the wrong layers.
    """
    a, b = set(top_k_layers_stage1), set(top_k_layers_ce)
    jac = len(a & b) / max(len(a | b), 1)
    print(f"  [S2-GATE] Stage-1 top-k : {sorted(a)}")
    print(f"  [S2-GATE] CE      top-k : {sorted(b)}")
    print(f"  [S2-GATE] Jaccard overlap = {jac * 100:.1f}%  "
          f"(threshold {threshold * 100:.0f}%)")
    if jac < threshold and not force:
        hint = f"{n_calib} -> {2 * int(n_calib)}" if n_calib else "256 -> 512"
        print(f"  [S2-GATE] Phase-2 top-k overlap = {jac * 100:.1f}% < "
              f"{threshold * 100:.0f}%. Stage-1 trigger is NOT sufficient for "
              f"reliable layer selection.")
        print(f"  [S2-GATE] Recommended: increase S1_N_CALIB ({hint}) and/or "
              f"S1_EPOCHS, then re-run. Refusing to proceed to Stage 2.")
        raise SystemExit(1)
    return jac


def phase2_ce_reference(model, text_emb, t_target, calib_loader, calib_imgs, cfg):
    """Top-k layers from Phase 2A on the *original CE* trigger (cached on disk).

    Runs Phase 1 with P1_LOSS="ce" (same N_CALIB / epochs / trigger geometry as
    Stage 1, so the only difference is the loss) and Phase 2A on that trigger.
    The result is cached in cfg.OUT_DIR/phase2_ce_reference.json and regenerated
    if the cache was written with a different configuration.
    """
    path = os.path.join(cfg.OUT_DIR, "phase2_ce_reference.json")
    key = dict(n_calib=int(cfg.N_CALIB), epochs=int(cfg.S1_EPOCHS),
               trig_sz=int(cfg.TRIG_SZ), model=cfg.MODEL,
               dataset=cfg.DATASET_NAME, target_cls=int(cfg.TARGET_CLS))
    if os.path.exists(path):
        try:
            with open(path) as f:
                ref = json.load(f)
        except Exception:
            ref = None
        if ref and ref.get("key") == key:
            print(f"  [S2-GATE] CE reference top-k loaded from {path}")
            return set(int(x) for x in ref["top_k_layers"])
        print(f"  [S2-GATE] CE reference cache mismatches current config — regenerating")

    print("  [S2-GATE] Computing CE reference trigger + Phase 2A (one-off)")
    ref_cfg = copy.copy(cfg)
    ref_cfg.P1_LOSS = "ce"
    ref_cfg.P1_EPOCHS = cfg.S1_EPOCHS
    ref_trigger = phase1_generate_trigger(
        model, text_emb, t_target, calib_loader, None, ref_cfg)
    scores, layers = phase2_causal_ablation(
        model, ref_trigger, t_target, calib_imgs, ref_cfg)
    top_k = sorted(int(x) for x in layers)
    with open(path, "w") as f:
        json.dump(dict(key=key, top_k_layers=top_k,
                       layer_scores={str(k): float(v) for k, v in scores.items()},
                       trigger_loss="ce"), f, indent=2)
    print(f"  [S2-GATE] CE reference cached → {path}")
    del ref_trigger
    cleanup_gpu()
    return set(top_k)


def setup_weight_tracking(model, selected_components):
    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads

    tensor_refs = {}
    original_weights = {}

    for s in selected_components:
        li = s["layer"]
        blk = blocks[li]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            if key_v not in tensor_refs:
                tensor_refs[key_v] = (blk.attn.in_proj_weight, slice(2 * D + lo, 2 * D + hie), slice(None))
                original_weights[key_v] = blk.attn.in_proj_weight.data[2 * D + lo:2 * D + hie].clone()
            key_o = (li, hi, "O")
            if key_o not in tensor_refs:
                tensor_refs[key_o] = (blk.attn.out_proj.weight, slice(None), slice(lo, hie))
                original_weights[key_o] = blk.attn.out_proj.weight.data[:, lo:hie].clone()
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            if key_fc not in tensor_refs:
                tensor_refs[key_fc] = (blk.mlp.c_fc.weight, slice(None), slice(None))
                original_weights[key_fc] = blk.mlp.c_fc.weight.data.clone()
            key_proj = (li, -1, "proj")
            if key_proj not in tensor_refs:
                tensor_refs[key_proj] = (blk.mlp.c_proj.weight, slice(None), slice(None))
                original_weights[key_proj] = blk.mlp.c_proj.weight.data.clone()

    return tensor_refs, original_weights


def _collect_selected_params(model, selected_components, top_k_layers=None, k=None):
    """Exactly the parameter slices Phase 3/4 will flip (θ_s in L_sens).

    Walks `selected_components` (Phase 2B output) through `setup_weight_tracking`,
    so the set is identical to what `phase4_greedy_search` / `_get_tensor` touch.
    Optional `k` further restricts to the top-k of the Phase-2 layer ranking.
    """
    tensor_refs, _ = setup_weight_tracking(model, selected_components)
    allowed = None
    if k is not None and top_k_layers is not None:
        allowed = set(sorted(top_k_layers)[:k])
    entries = []
    for key, (tensor, rows, cols) in tensor_refs.items():
        layer, head, tn = key
        if allowed is not None and layer not in allowed:
            continue
        entries.append(dict(key=key, tensor=tensor, rows=rows, cols=cols))
    return entries


def _sens_from_grads(entries, grads, tensor_ids):
    """||∇_θs cos||_F² summed over the selected slices (0 if none contributed)."""
    by_id = {tid: g for tid, g in zip(tensor_ids, grads)}
    total = None
    for e in entries:
        g = by_id.get(id(e["tensor"]))
        if g is None:
            continue
        block = g[e["rows"], e["cols"]]
        term = block.pow(2).sum()
        total = term if total is None else total + term
    return total


@contextmanager
def _tee_stdout(path):
    """Mirror everything printed inside the block into `path` (run logs)."""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
            return len(data)
        def flush(self):
            for s in self.streams:
                s.flush()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        with contextlib.redirect_stdout(_Tee(sys.stdout, fh)):
            yield


@contextmanager
def _math_sdpa():
    """Force the math SDPA backend for double backward.

    CLIP's `nn.MultiheadAttention` dispatches to fused SDPA kernels
    (flash / mem-efficient), whose backward has no second-order derivative:
    "derivative for aten::_scaled_dot_product_efficient_attention_backward is
    not implemented".  The math backend decomposes into matmul + softmax and
    supports double backward.  Restored on exit.
    """
    if not torch.cuda.is_available():
        yield
        return
    f = torch.backends.cuda.flash_sdp_enabled()
    e = torch.backends.cuda.mem_efficient_sdp_enabled()
    m = torch.backends.cuda.math_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(f)
        torch.backends.cuda.enable_mem_efficient_sdp(e)
        torch.backends.cuda.enable_math_sdp(m)


def stage2_refine_trigger(model, text_emb, t_target, trigger_s1, selected_components,
                          top_k_layers, calib_loader, cfg, text_emb_public=None):
    """Backwards-compatible alias for the v2 Stage-2 objective."""
    return phase1_stage2_sens(model, text_emb, t_target, trigger_s1,
                              selected_components, top_k_layers, calib_loader,
                              cfg, text_emb_public=text_emb_public)


def _sub_grad(a, b):
    """g_m = g_c - g_lse, tolerating tensors unused by one of the terms."""
    if a is None:
        return None if b is None else -b
    if b is None:
        return a
    return a - b


def phase1_stage2_sens(model, text_emb, t_target, trigger_s1, selected_components,
                       top_k_layers, calib_loader, cfg, text_emb_public=None):
    """Stage 2 (v2): direction-aware sensitivity + alignment.

        L =  S2_W_COS    * (-cos_t / T)
           + S2_W_SENS   * (-||grad_theta_s m_soft||^2)
           + S2_W_ALIGN  * (-<grad m_soft, grad cos_t>)
           + S2_W_MARGIN * relu(S2_MARGIN_THRESH - mean(m_soft))
           + S2_W_L2     * mean(trig^2)
           + S2_W_TV     * TV(trig)

        m_soft = cos_t - (1/beta) * logsumexp(beta * emb @ text_emb_public^T)

    The two negated gradient terms are negated because we *minimise* L: the
    optimum maximises ||grad m||^2 and aligns grad m with grad cos_t.

    S2_MODE:
        "cos"         only w_cos / w_margin / w_l2 / w_tv (no sensitivity term)
        "sens"        legacy arm: -||grad cos_t||^2, slice-restricted and
                      sens0-normalised (reproduces the failing Stage 2)
        "margin_sens" the objective above, with the spec's g_m = g_c - g_lse
                      decomposition (one second-order backward per step)

    Overshoot controls (always on, unchanged): init from the Stage-1 trigger,
    grad clipping, ±TRIG_CLAMP, relative drift limit toward Stage-1.
    """
    with _math_sdpa():
        return _stage2_sens_impl(
            model, text_emb, t_target, trigger_s1, selected_components,
            top_k_layers, calib_loader, cfg, text_emb_public=text_emb_public)


def _stage2_sens_impl(model, text_emb, t_target, trigger_s1, selected_components,
                      top_k_layers, calib_loader, cfg, text_emb_public=None):
    """Implementation of phase1_stage2_sens (setup + double-backward loop)."""
    mode = cfg.S2_MODE
    if mode not in ("cos", "sens", "margin_sens"):
        raise ValueError(f"unknown S2_MODE={mode!r}")
    if mode == "margin_sens" and text_emb_public is None:
        raise ValueError("S2_MODE='margin_sens' needs text_emb_public "
                         "(build_public_text_embeddings)")

    print("\n" + "=" * 60)
    print(f"STAGE 2: DIRECTION-AWARE SENSITIVITY (mode={mode})")
    print("=" * 60)

    entries = _collect_selected_params(
        model, selected_components, top_k_layers, cfg.S2_SENS_TOP_K_LAYERS)
    assert len(entries) > 0, "Stage-2 has no target params — Phase 2B failed"

    tensors, seen = [], set()
    for e in entries:
        if id(e["tensor"]) not in seen:
            seen.add(id(e["tensor"]))
            tensors.append(e["tensor"])
    tensor_ids = [id(t) for t in tensors]
    n_el = sum(int(torch.numel(e["tensor"][e["rows"], e["cols"]])) for e in entries)

    layers = sorted({e["key"][0] for e in entries})
    print(f"  θ_s : {len(entries)} slices / {len(tensors)} tensors / "
          f"{n_el} elements  (layers {layers})")
    print(f"  epochs={cfg.S2_EPOCHS}  lr={cfg.S2_LR}  batch={cfg.S2_BATCH}  "
          f"β={cfg.S2_LSE_BETA}  T={cfg.TEMPERATURE}")
    print(f"  w_cos={cfg.S2_W_COS}  w_sens={cfg.S2_W_SENS}  "
          f"w_align={cfg.S2_W_ALIGN}  w_margin={cfg.S2_W_MARGIN} (μ={cfg.S2_MARGIN_THRESH})  "
          f"w_l2={cfg.S2_W_L2}  w_tv={cfg.S2_W_TV}")
    print(f"  clip={cfg.S2_GRAD_CLIP}  clamp=±{cfg.TRIG_CLAMP}  "
          f"drift≤{cfg.S2_STAGE1_DRIFT_LIMIT}·‖trig_s1‖")
    if text_emb_public is not None:
        print(f"  public vocabulary: {text_emb_public.shape[0]} classes "
              f"(cos>{cfg.S2_PUBLIC_COS_EXCLUDE} filtered against t_target)")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for t in tensors:
        t.requires_grad_(True)

    trigger_s1 = trigger_s1.detach().to(cfg.DEVICE)
    trig = nn.Parameter(trigger_s1.clone())          # never randn
    opt = torch.optim.Adam([trig], lr=cfg.S2_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.S2_EPOCHS, 1), eta_min=cfg.S2_LR * 0.01)

    # Stage-2 calibration loader (disjoint image set, built in main())
    assert calib_loader is not None, "Stage 2 needs its own calibration loader"
    subset = calib_loader.dataset
    if cfg.S2_N_CALIB and cfg.S2_N_CALIB < len(subset.indices):
        subset = Subset(subset.dataset, list(subset.indices)[:cfg.S2_N_CALIB])
    loader = DataLoader(subset, batch_size=cfg.S2_BATCH, shuffle=True,
                        num_workers=cfg.WORKERS,
                        pin_memory=torch.cuda.is_available())
    print(f"  Stage-2 calibration images: {len(subset)} (disjoint from Stage 1)")

    apply_mode, apply_opacity = get_apply_params(cfg)
    s1_norm = trigger_s1.norm().item()
    best_loss, best_trig = float("inf"), trig.data.clone()
    sens0 = None
    epoch_log = []
    info = dict(sens0=None, sens_last=None, drift_ratio_start=0.0,
                drift_ratio_end=None, cos_start=None, cos_end=None,
                mode=mode, epochs=[])

    for ep in range(1, cfg.S2_EPOCHS + 1):
        ep_loss = nb = 0
        acc = defaultdict(float)
        for imgs, _ in loader:
            imgs = imgs.to(cfg.DEVICE)
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ, apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)

            cos_t_b = F.cosine_similarity(
                emb, t_target.unsqueeze(0).expand_as(emb), dim=-1)
            cos_t = cos_t_b.mean()

            # m_soft = cos_t - soft-max over the public vocabulary
            if text_emb_public is not None and mode != "cos":
                sims_public = emb @ text_emb_public.T                        # (B, K)
                lse = torch.logsumexp(cfg.S2_LSE_BETA * sims_public, dim=-1) \
                    / cfg.S2_LSE_BETA
                m_soft_b = cos_t_b - lse
            else:
                lse = emb.new_zeros(emb.size(0))
                m_soft_b = cos_t_b
            m_soft = m_soft_b.mean()

            loss = cfg.S2_W_COS * (-cos_t / cfg.TEMPERATURE)

            if mode != "cos":
                # g_c and g_lse share one graph: only one second-order backward
                g_c = torch.autograd.grad(
                    cos_t, tensors, create_graph=True, retain_graph=True,
                    allow_unused=True)

                if mode == "sens":
                    # legacy arm: -||grad cos||^2 (slice-restricted, sens0-normalised)
                    sens = _sens_from_grads(entries, g_c, tensor_ids)
                    if sens is None:
                        raise RuntimeError(
                            "L_sens: none of the selected θ_s contributed to the graph")
                    if sens0 is None:
                        sens0 = sens.detach().clamp_min(1e-30)
                        info["sens0"] = float(sens0)
                        info["cos_start"] = float(cos_t.detach())
                        print(f"  sens0 = {float(sens0):.3e}   "
                              f"cos0 = {float(cos_t.detach()):+.4f}")
                    loss = loss - cfg.S2_W_SENS * (
                        sens / sens0 if cfg.S2_SENS_NORM else sens)
                    acc["sens"] += float(sens.detach())
                else:  # margin_sens
                    g_lse = torch.autograd.grad(
                        lse.mean(), tensors, create_graph=True,
                        retain_graph=True, allow_unused=True)
                    g_m = [_sub_grad(a, b) for a, b in zip(g_c, g_lse)]
                    pairs = [(a, b) for a, b in zip(g_m, g_c)
                             if a is not None and b is not None]
                    if not pairs:
                        raise RuntimeError(
                            "L_sens: none of the selected θ_s contributed to the graph")
                    gm_flat = torch.cat([a.flatten() for a, _ in pairs])
                    gc_flat = torch.cat([b.flatten() for _, b in pairs])
                    sens = gm_flat.pow(2).sum()
                    align_dot = (gm_flat * gc_flat).sum()

                    if sens0 is None:
                        sens0 = sens.detach().clamp_min(1e-30)
                        info["sens0"] = float(sens0)
                        info["cos_start"] = float(cos_t.detach())
                        print(f"  ||grad m||^2 at init = {float(sens0):.3e}   "
                              f"cos0 = {float(cos_t.detach()):+.4f}")

                    loss = (loss - cfg.S2_W_SENS * sens
                                 - cfg.S2_W_ALIGN * align_dot)
                    acc["sens"] += float(sens.detach())
                    acc["align_dot"] += float(align_dot.detach())
                    acc["align_cos"] += float(F.cosine_similarity(
                        gm_flat.unsqueeze(0), gc_flat.unsqueeze(0)).detach())
                    # slice-restricted norm, logged for reference only
                    acc["sens_slices"] += float(
                        _sens_from_grads(entries, g_m, tensor_ids).detach())

            # hinge on the batch-mean margin (differentiable through emb)
            loss = loss + cfg.S2_W_MARGIN * F.relu(cfg.S2_MARGIN_THRESH - m_soft)
            if cfg.S2_W_L2 > 0:
                loss = loss + cfg.S2_W_L2 * trig.pow(2).mean()
            if cfg.S2_W_TV > 0:
                loss = loss + cfg.S2_W_TV * total_variation(trig)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([trig], max_norm=cfg.S2_GRAD_CLIP)

            with torch.no_grad():
                if trig.grad is not None and torch.isfinite(trig.grad).all():
                    opt.step()
                # bounded + stay near the Stage-1 trigger
                trig.data.clamp_(-cfg.TRIG_CLAMP, cfg.TRIG_CLAMP)
                diff = trig.data - trigger_s1
                diff_norm = diff.norm()
                limit = cfg.S2_STAGE1_DRIFT_LIMIT * trigger_s1.norm() + 1e-6
                if diff_norm > limit:
                    diff.mul_(limit / diff_norm)
                    trig.data.copy_(trigger_s1 + diff)
                drift = (trig.data - trigger_s1).norm().item() / max(s1_norm, 1e-12)
            sched.step()

            ep_loss += loss.item()
            acc["cos_t"] += float(cos_t.detach())
            acc["m_soft"] += float(m_soft.detach())
            acc["drift"] += drift
            nb += 1

        n = max(nb, 1)
        avg_l = ep_loss / n
        rec = dict(ep=ep, loss=avg_l,
                   cos_t=acc["cos_t"] / n,
                   m_soft=acc["m_soft"] / n,
                   sens=acc["sens"] / n,
                   sens_slices=acc["sens_slices"] / n,
                   align_dot=acc["align_dot"] / n,
                   align_cos=acc["align_cos"] / n,
                   drift=acc["drift"] / n,
                   trig_norm=float(trig.data.norm()))
        epoch_log.append(rec)
        if avg_l < best_loss:
            best_loss, best_trig = avg_l, trig.data.clone()

        line = (f"  s2 ep {ep:>3d}  loss={avg_l:+.5f}  cos_t={rec['cos_t']:+.4f}  "
                f"m_soft={rec['m_soft']:+.4f}  |grad m|^2={rec['sens']:.3e}  "
                f"<gm,gc>={rec['align_dot']:+.3e}  align_cos={rec['align_cos']:+.4f}  "
                f"drift/‖s1‖={rec['drift']:.3f}  ‖trig‖={rec['trig_norm']:.2f}")
        if ep == 1 or ep % cfg.S2_LOG_EVERY == 0 or ep == cfg.S2_EPOCHS:
            print(line)
        if cfg.S2_LOG_PATH:
            with open(cfg.S2_LOG_PATH, "a") as fh:
                fh.write(line + "\n")
    info["epochs"] = epoch_log
    last = epoch_log[-1] if epoch_log else {}
    info["sens_last"] = last.get("sens")
    info["sens_slices_last"] = last.get("sens_slices")
    info["align_dot_last"] = last.get("align_dot")
    info["align_cos_last"] = last.get("align_cos")
    info["m_soft_last"] = last.get("m_soft")
    info["cos_end"] = last.get("cos_t")
    info["drift_ratio_end"] = last.get("drift")
    info["best_loss"] = best_loss
    info["s2_calib"] = len(subset)
    info["n_slices"] = len(entries)
    info["n_params"] = n_el
    info["layers"] = layers
    info["mode"] = mode

    for t in tensors:
        t.requires_grad_(False)
    done_msg = (f"  Stage-2 done. best loss={best_loss:+.5f}  "
                f"cos {info['cos_start']:+.4f} → {last.get('cos_t', float('nan')):+.4f}  "
                f"m_soft {last.get('m_soft', float('nan')):+.4f}  "
                f"||grad m||^2 {info['sens0']:.3e} → {last.get('sens', float('nan')):.3e}")
    print(done_msg)
    if cfg.S2_LOG_PATH:
        with open(cfg.S2_LOG_PATH, "a") as fh:
            fh.write(done_msg + "\n")
    if mode == "margin_sens" and last.get("align_dot", 0.0) < 0:
        print("  WARNING: alignment term did not become positive — the "
              "g_m = g_c - g_lse decomposition may need β ≥ 20; "
              "try S2_LSE_BETA=20 next")
    return best_trig.detach(), info


def stage2_diagnostics(model, trig, loader, t_target, text_emb, text_emb_public,
                       selected_components, top_k_layers, cfg, max_batches=2):
    """Post-Stage-2 health check on the Stage-2 calibration set.

    Reports cos(poisoned, t_target), the public-vocabulary soft margin
    m_soft, ||grad_theta_s m_soft||^2, <grad m, grad cos>, the Caltech-vocabulary
    margin, and the slice-restricted sensitivity (for comparison with the
    raw-norm objective).
    """
    entries = _collect_selected_params(model, selected_components, top_k_layers,
                                       cfg.S2_SENS_TOP_K_LAYERS)
    assert len(entries) > 0, "Stage-2 diagnostics: no target params"
    tensors, seen = [], set()
    for e in entries:
        if id(e["tensor"]) not in seen:
            seen.add(id(e["tensor"]))
            tensors.append(e["tensor"])
    tensor_ids = [id(t) for t in tensors]
    for t in tensors:
        t.requires_grad_(True)

    apply_mode, apply_opacity = get_apply_params(cfg)
    acc = defaultdict(float)
    nb = 0
    with _math_sdpa():
        for imgs, _ in loader:
            imgs = imgs.to(cfg.DEVICE)
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ, apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)
            cos_b = F.cosine_similarity(
                emb, t_target.unsqueeze(0).expand_as(emb), dim=-1)

            # Caltech-vocabulary margin (what Phase 3/4 and ASR actually use)
            sims_cal = emb @ text_emb.T
            other = sims_cal.clone()
            other[:, cfg.TARGET_CLS] = -1e9
            margin_cal = sims_cal[:, cfg.TARGET_CLS] - other.max(dim=1).values

            acc["cos_poison"] += float(cos_b.mean())
            acc["margin_caltech"] += float(margin_cal.mean())

            if text_emb_public is not None:
                sims_pub = emb @ text_emb_public.T
                lse = torch.logsumexp(cfg.S2_LSE_BETA * sims_pub, dim=-1) \
                    / cfg.S2_LSE_BETA
                m_soft = (cos_b - lse).mean()
                acc["m_soft"] += float(m_soft)

                g_c = torch.autograd.grad(
                    cos_b.mean(), tensors, retain_graph=True, allow_unused=True)
                g_lse = torch.autograd.grad(
                    lse.mean(), tensors, allow_unused=True)
                g_m = [_sub_grad(a, b) for a, b in zip(g_c, g_lse)]
                pairs = [(a, b) for a, b in zip(g_m, g_c)
                         if a is not None and b is not None]
                if pairs:
                    gm_flat = torch.cat([a.flatten() for a, _ in pairs])
                    gc_flat = torch.cat([b.flatten() for _, b in pairs])
                    acc["sens"] += float(gm_flat.pow(2).sum())
                    acc["align_dot"] += float((gm_flat * gc_flat).sum())
                    acc["align_cos"] += float(F.cosine_similarity(
                        gm_flat.unsqueeze(0), gc_flat.unsqueeze(0)))
                    acc["sens_slices"] += float(
                        _sens_from_grads(entries, g_m, tensor_ids))
            else:
                acc["m_soft"] += float(cos_b.mean())
            nb += 1
            if nb >= max_batches:
                break

    for t in tensors:
        t.requires_grad_(False)
    n = max(nb, 1)
    out = {k: acc[k] / n for k in ("cos_poison", "m_soft", "sens",
                                   "align_dot", "align_cos", "sens_slices",
                                   "margin_caltech")}
    return {k: float(v) for k, v in out.items()}


def build_sub_quant(model, selected_components, quant_dict):
    D = model.visual.transformer.width
    n_heads = list(model.visual.transformer.resblocks)[0].attn.num_heads
    dk = D // n_heads
    sq = {}
    for s in selected_components:
        li = s["layer"]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            if key_v not in sq:
                ip_info = quant_dict[f"L{li}_attn_ip_w"]
                sq[key_v] = dict(int8=ip_info["int8"][2 * D + lo:2 * D + hie].clone(), scale=ip_info["scale"])
            key_o = (li, hi, "O")
            if key_o not in sq:
                op_info = quant_dict[f"L{li}_attn_op_w"]
                sq[key_o] = dict(int8=op_info["int8"][:, lo:hie].clone(), scale=op_info["scale"])
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            if key_fc not in sq:
                fc_info = quant_dict[f"L{li}_fc1_w"]
                sq[key_fc] = dict(int8=fc_info["int8"].clone(), scale=fc_info["scale"])
            key_proj = (li, -1, "proj")
            if key_proj not in sq:
                proj_info = quant_dict[f"L{li}_fc2_w"]
                sq[key_proj] = dict(int8=proj_info["int8"].clone(), scale=proj_info["scale"])
    return sq


def phase3_hybrid_ranking(model, trigger, t_target, layer_R, selected_heads, tensor_refs, grad_imgs, cfg, sub_quant):
    print("\n" + "=" * 60)
    print("PHASE 3: HYBRID BIT RANKING")
    print("=" * 60)

    for p in model.parameters():
        p.requires_grad_(False)

    apply_mode, apply_opacity = get_apply_params(cfg)
    selected_tensors = set()
    for key, (tensor, _, _) in tensor_refs.items():
        tensor.requires_grad_(True)
        selected_tensors.add(tensor)

    mini_bs = cfg.MINI_BS
    n_imgs = grad_imgs.size(0)

    model.zero_grad()
    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
        emb_po = F.normalize(model.encode_image(poisoned), dim=-1)
        loss_po = -F.cosine_similarity(emb_po, t_target.detach().unsqueeze(0).expand(emb_po.size(0), -1), dim=-1).mean()
        loss_po.backward()
        del chunk, poisoned, emb_po, loss_po

    grad_poisoned = {}
    for key, (tensor, rslice, cslice) in tensor_refs.items():
        grad_poisoned[key] = tensor.grad[rslice, cslice].clone()

    for tensor in selected_tensors:
        tensor.grad = None

    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        emb_cl = F.normalize(model.encode_image(chunk), dim=-1)
        loss_cl = -F.cosine_similarity(emb_cl, t_target.detach().unsqueeze(0).expand(emb_cl.size(0), -1), dim=-1).mean()
        loss_cl.backward()
        del chunk, emb_cl, loss_cl

    grad_clean = {}
    for key, (tensor, rslice, cslice) in tensor_refs.items():
        grad_clean[key] = tensor.grad[rslice, cslice].clone()

    grad_specific = {}
    for key in grad_poisoned:
        grad_specific[key] = grad_poisoned[key] - grad_clean[key]

    for tensor in selected_tensors:
        tensor.requires_grad_(False)
        tensor.grad = None

    ranked = []
    for key, (tensor, rslice, cslice) in tensor_refs.items():
        li, hi, tn = key
        g_slice = grad_specific[key]
        R_l = layer_R.get(li, 1.0)
        sq = sub_quant[(li, hi, tn)]
        int8_vals = sq["int8"]
        scale = sq["scale"]
        flat_grad = g_slice.flatten()
        flat_int8 = int8_vals.flatten()
        flat_sens = flat_grad.abs()

        k = min(cfg.SENS_TOP_M, flat_sens.numel())
        _, topk = flat_sens.topk(k)

        for idx in topk.tolist():
            q, p = divmod(idx, g_slice.shape[1])
            g = flat_grad[idx].item()
            if g >= 0:
                continue
            old_i8 = int(flat_int8[idx].item())
            for bit in range(8):
                new_i8 = flip_bit_int8(old_i8, bit)
                dw = (new_i8 - old_i8) * scale
                dpred = g * dw
                if dpred >= 0:
                    continue
                hybrid_score = R_l * abs(dpred)
                ranked.append(dict(tensor=tn, layer=li, head=hi, component="attn" if hi >= 0 else "ffn",
                                   row=q, col=p, bit=bit, old_int8=old_i8, new_int8=new_i8,
                                   delta_w=dw, predicted_dL=dpred, layer_weight_R=R_l,
                                   gradient_score=abs(dpred), final_score=hybrid_score))

    ranked.sort(key=lambda d: -d["final_score"])
    print(f"  Candidates: {len(ranked)}")
    cleanup_gpu()
    return ranked


def phase3_gradient_ranking_fallback(model, trigger, t_target, selected_heads, grad_imgs, cfg, sub_quant):
    print("\n" + "=" * 60)
    print("PHASE 3 (FALLBACK): GRADIENT-ONLY RANKING")
    print("=" * 60)
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads
    target_vec = t_target.detach()
    apply_mode, apply_opacity = get_apply_params(cfg)
    mini_bs = cfg.MINI_BS
    n_imgs = grad_imgs.size(0)

    tracked_params = {}
    for s in selected_heads:
        li = s["layer"]
        blk = blocks[li]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            w_v = blk.attn.in_proj_weight
            w_v.requires_grad_(True)
            tracked_params[(li, hi, "V", "attn")] = (w_v, 2 * D + lo, 2 * D + hie, 0, D)
            w_o = blk.attn.out_proj.weight
            w_o.requires_grad_(True)
            tracked_params[(li, hi, "O", "attn")] = (w_o, 0, D, lo, hie)
        elif s["type"] == "ffn":
            w_fc = blk.mlp.c_fc.weight
            w_fc.requires_grad_(True)
            tracked_params[(li, -1, "fc", "ffn")] = (w_fc, 0, w_fc.shape[0], 0, w_fc.shape[1])
            w_proj = blk.mlp.c_proj.weight
            w_proj.requires_grad_(True)
            tracked_params[(li, -1, "proj", "ffn")] = (w_proj, 0, w_proj.shape[0], 0, w_proj.shape[1])

    model.zero_grad()
    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
        emb_trig = F.normalize(model.encode_image(poisoned), dim=-1)
        L_target = -F.cosine_similarity(emb_trig, target_vec.unsqueeze(0).expand(emb_trig.size(0), -1), dim=-1).mean()
        L_target.backward()
        del chunk, poisoned, emb_trig, L_target

    grad_poisoned = {}
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        grad_poisoned[key] = param.grad[r_start:r_end, c_start:c_end].clone()

    for key, (param, _, _, _, _) in tracked_params.items():
        param.grad = None

    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        emb_clean = F.normalize(model.encode_image(chunk), dim=-1)
        L_clean = -F.cosine_similarity(emb_clean, target_vec.unsqueeze(0).expand(emb_clean.size(0), -1), dim=-1).mean()
        L_clean.backward()
        del chunk, emb_clean, L_clean

    grad_clean = {}
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        grad_clean[key] = param.grad[r_start:r_end, c_start:c_end].clone()

    grad_specific = {key: grad_poisoned[key] - grad_clean[key] for key in grad_poisoned}

    for key, (param, _, _, _, _) in tracked_params.items():
        param.requires_grad_(False)
        param.grad = None

    ranked = []
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        li, hi, tn, comp_type = key
        grad_slice = grad_specific[key]
        sq = sub_quant[(li, hi, tn)]
        flat_grad = grad_slice.flatten()
        flat_int8 = sq["int8"].flatten()
        flat_sens = flat_grad.abs()

        k = min(cfg.SENS_TOP_M, flat_sens.numel())
        _, topk = flat_sens.topk(k)
        for idx in topk.tolist():
            q, p = divmod(idx, grad_slice.shape[1])
            g = flat_grad[idx].item()
            if g >= 0: continue
            old_i8 = int(flat_int8[idx].item())
            for bit in range(8):
                new_i8 = flip_bit_int8(old_i8, bit)
                dw = (new_i8 - old_i8) * sq["scale"]
                dpred = g * dw
                if dpred >= 0: continue
                ranked.append(dict(tensor=tn, layer=li, head=hi, component=comp_type, row=q, col=p, bit=bit,
                                   old_int8=old_i8, new_int8=new_i8, delta_w=dw, predicted_dL=dpred,
                                   layer_weight_R=1.0, gradient_score=abs(dpred), final_score=abs(dpred)))

    ranked.sort(key=lambda d: -d["final_score"])
    cleanup_gpu()
    return ranked


def _get_abs_pos(tn, layer, head, row, col, D, dk):
    if tn == "V": return 2 * D + head * dk + row, col
    elif tn == "O": return row, head * dk + col
    else: return row, col


def _local_idx(tn, head, abs_r, abs_c, D, dk):
    if tn == "V": return abs_r - (2 * D + head * dk), abs_c
    elif tn == "O": return abs_r, abs_c - head * dk
    else: return abs_r, abs_c


def _get_tensor(blk, tn):
    if tn == "V": return blk.attn.in_proj_weight
    elif tn == "O": return blk.attn.out_proj.weight
    elif tn == "fc": return blk.mlp.c_fc.weight
    elif tn == "proj": return blk.mlp.c_proj.weight


def _get_sub_quant_key(c):
    return (c["layer"], c["head"], c["tensor"])


@torch.no_grad()
def phase4_greedy_search(model, text_emb, trigger, selected_heads, ranked_flips, quant_dict,
                         eval_imgs, eval_labels, grad_imgs, original_weights, cfg,
                         tensor_refs=None, layer_R=None, sub_quant=None):
    print("\n" + "=" * 60)
    print("PHASE 4: GREEDY BIT-FLIP SEARCH")
    print("=" * 60)

    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads
    tc = cfg.TARGET_CLS
    ttype = cfg.TRIGGER_TYPE
    topac = cfg.OPACITY
    apply_mode, apply_opacity = get_apply_params(cfg)
    eval_bs = cfg.EVAL_BS

    drift_n = min(16, eval_imgs.shape[0])
    drift_imgs = eval_imgs[:drift_n]

    if sub_quant is None:
        _layers = set(s["layer"] for s in selected_heads)
        sub_quant = build_sub_quant(model, _layers, quant_dict)

    asr, cacc = quick_evaluate(model, text_emb, trigger, eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
    cacc_base = cacc
    asr_base = asr
    print(f"  Baseline  ASR={asr:.1f}%  CACC={cacc:.1f}%")

    clean_emb_ref = F.normalize(encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()
    target_vec = text_emb[tc].detach()

    poisoned = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
    trig_emb_ref = F.normalize(encode_image_batched(model, poisoned, eval_bs), dim=-1).clone()
    trig_cos_ref = (trig_emb_ref @ target_vec).cpu()

    flip_log = []
    checkpoints = [(0, asr, cacc)]
    ranked = list(ranked_flips)
    cand_ptr = 0
    no_improve = 0

    def rebuild_candidates():
        nonlocal ranked, cand_ptr
        if tensor_refs is not None and layer_R is not None:
            with torch.enable_grad():
                ranked = phase3_hybrid_ranking(model, trigger, target_vec, layer_R, selected_heads, tensor_refs, grad_imgs, cfg, sub_quant=sub_quant)
        else:
            with torch.enable_grad():
                ranked = phase3_gradient_ranking_fallback(model, trigger, target_vec, selected_heads, grad_imgs, cfg, sub_quant=sub_quant)
        cand_ptr = 0

    for rd in range(cfg.MAX_FLIPS):
        if cand_ptr >= len(ranked):
            rebuild_candidates()
            if len(ranked) == 0: break

        batch = ranked[cand_ptr:cand_ptr + cfg.CANDS_PER_ROUND]
        cand_ptr += cfg.CANDS_PER_ROUND
        best_score = -1e9
        best_info = None

        for c in batch:
            blk = blocks[c["layer"]]
            ar, ac = _get_abs_pos(c["tensor"], c["layer"], c["head"], c["row"], c["col"], D, dk)
            tensor = _get_tensor(blk, c["tensor"])
            old_f = tensor.data[ar, ac].item()
            sq_key = _get_sub_quant_key(c)
            sq = sub_quant[sq_key]
            new_f = c["new_int8"] * sq["scale"]
            tensor.data[ar, ac] = new_f

            poisoned_new = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
            trig_emb_new = F.normalize(encode_image_batched(model, poisoned_new, eval_bs), dim=-1)
            trig_cos_new = (trig_emb_new @ target_vec).cpu()

            clean_emb_new = F.normalize(encode_image_batched(model, drift_imgs, eval_bs), dim=-1)
            drift = (1.0 - F.cosine_similarity(clean_emb_ref, clean_emb_new, dim=-1)).mean().item()

            tensor.data[ar, ac] = old_f

            if drift > cfg.MAX_CLEAN_DRIFT: continue
            delta_cos = (trig_cos_new - trig_cos_ref).mean().item()
            n_improved = (trig_cos_new > trig_cos_ref).sum().item()
            score = delta_cos - cfg.CLEAN_DRIFT_PENALTY * drift
            if score > best_score:
                best_score = score
                best_info = dict(tensor=c["tensor"], layer=c["layer"], head=c["head"], component=c["component"],
                                 abs_r=ar, abs_c=ac, bit=c["bit"], old_int8=c["old_int8"], new_int8=c["new_int8"],
                                 delta_cos=delta_cos, drift=drift, n_improved=n_improved, layer_weight_R=c.get("layer_weight_R", 1.0))

        if best_info is None or best_score <= 0:
            no_improve += 1
            if no_improve >= cfg.PLATEAU_PATIENCE: break
            continue

        no_improve = 0
        bi = best_info
        blk = blocks[bi["layer"]]
        tensor = _get_tensor(blk, bi["tensor"])
        tensor.data[bi["abs_r"], bi["abs_c"]] = bi["new_int8"] * sub_quant[_get_sub_quant_key(bi)]["scale"]
        sq = sub_quant[_get_sub_quant_key(bi)]
        lr, lc = _local_idx(bi["tensor"], bi["head"], bi["abs_r"], bi["abs_c"], D, dk)
        sq["int8"][lr, lc] = bi["new_int8"]

        poisoned_upd = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
        trig_emb_ref = F.normalize(encode_image_batched(model, poisoned_upd, eval_bs), dim=-1).clone()
        trig_cos_ref = (trig_emb_ref @ target_vec).cpu()
        clean_emb_ref = F.normalize(encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()

        flip_log.append(dict(round=rd, tensor=bi["tensor"], layer=bi["layer"], head=bi["head"], component=bi["component"],
                             position=(bi["abs_r"], bi["abs_c"]), bit=bi["bit"], old_int8=bi["old_int8"],
                             new_int8=bi["new_int8"], delta_cos=bi["delta_cos"], drift=bi["drift"],
                             n_improved=bi["n_improved"], score=best_score, layer_weight_R=bi.get("layer_weight_R", 1.0)))

        if len(flip_log) % 5 == 0:
            a_check, _ = quick_evaluate(model, text_emb, trigger, eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
            if a_check >= cfg.ASR_TARGET: break
            if len(flip_log) % cfg.REFRESH_EVERY == 0: rebuild_candidates()

    a_f, c_f = quick_evaluate(model, text_emb, trigger, eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
    checkpoints.append((len(flip_log), a_f, c_f))
    print(f"  Total flips : {len(flip_log)}")
    print(f"  Final ASR   : {a_f:.1f}%  (Δ = {a_f - asr_base:+.1f}%)")
    return flip_log, checkpoints


def main():
    set_seed(42)
    cfg = CFG()

    # ═══ Stage-1 configuration (applied before the splits are built) ═══
    # Pool and seed unchanged: N_CALIB = 64, set_seed(42) above.  P1_LOSS is
    # left at its class default so the CE-family dispatcher still reproduces.
    cfg.N_CALIB = cfg.S1_N_CALIB
    if cfg.S1_EPOCHS:
        cfg.P1_EPOCHS = cfg.S1_EPOCHS

    # per-arm output tag
    tag = cfg.P1_MODE
    if cfg.P1_MODE == "cos_eot":
        tag += f"-m{cfg.EOT_M}"
    elif cfg.P1_MODE == "cos_astar":
        tag += f"-k{cfg.ASTAR_K}-l{cfg.ASTAR_LAMBDA:g}-mu{cfg.ASTAR_MU:g}"
    cfg.OUT_DIR = f"{cfg.OUT_DIR.rstrip('/')}_{tag}"

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dev = cfg.DEVICE

    print("=" * 60)
    print("HYBRID BIT-FLIP ATTACK ON CLIP (FIXED TRIGGER)")
    print("=" * 60)

    print("\n[1] Loading CLIP")
    model, preprocess = clip.load(cfg.MODEL, device=dev)
    model = model.float()
    mparams = detect_model_params(model)
    cfg.PATCH_SIZE = mparams["patch_size"]
    cfg.IMG_SIZE = mparams["img_size"]

    print("[2] Loading Caltech-101")
    tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
    ds = datasets.Caltech101(root=cfg.DATA, download=cfg.DOWNLOAD,
                             transform=tf, target_type="category")
    classes = list(ds.categories)
    print(f"  {len(classes)} classes, target = '{classes[cfg.TARGET_CLS]}'")
    text_emb = build_text_embeddings(model, classes, dev)
    print(f"  Target class = {cfg.TARGET_CLS} → '{classes[cfg.TARGET_CLS]}'")

    print("\n[3] Preparing data splits")
    (target_sel, calib_idx, s2_calib_idx, grad_idx, eval_idx, test_idx,
     target_holdout) = prepare_data_splits(ds, cfg)

    target_images, _ = load_tensors(ds, target_sel, dev)
    calib_loader = make_loader(ds, calib_idx, cfg.BATCH, shuffle=True, workers=cfg.WORKERS)
    calib_imgs, _ = load_tensors(ds, calib_idx, dev)
    eval_imgs, eval_labels = load_tensors(ds, eval_idx, dev)
    grad_imgs_cpu, _ = load_tensors(ds, grad_idx, "cpu", batch_size=cfg.BATCH, workers=cfg.WORKERS)

    # Stage-2 calibration uses different images from Stage 1
    s2_calib_loader = None
    if cfg.S2_ENABLED and len(s2_calib_idx) > 0:
        assert set(s2_calib_idx).isdisjoint(calib_idx), \
            "Stage-2 calibration images must be disjoint from Stage-1"
        s2_calib_loader = make_loader(ds, s2_calib_idx, cfg.S2_BATCH,
                                      shuffle=True, workers=cfg.WORKERS)
        print(f"  Stage-2 calibration loader: {len(s2_calib_idx)} images "
              f"({len(set(s2_calib_idx) & set(calib_idx))} shared with Stage 1)")

    with torch.no_grad():
        t_target = text_emb[cfg.TARGET_CLS].detach()

    # Public vocabulary for the Stage-2 soft margin (margin_sens only)
    text_emb_public = None
    cfg.S2_LOG_PATH = os.path.join(cfg.OUT_DIR, "run_stage2.log")
    if cfg.S2_ENABLED and cfg.S2_MODE != "cos":
        text_emb_public = build_public_text_embeddings(
            model, dev, t_target, cfg.S2_PUBLIC_COS_EXCLUDE)

    trigger_token_indices = get_trigger_token_indices(cfg, model=model)

    # ═══ PHASE 0 — Quantisation ═══
    quant_dict = phase0_quantize(model, text_emb, cfg)
    cleanup_gpu()

    # ═══ PHASE 1 — Trigger Generation (P1_MODE dispatch) ═══
    print(f"\n[Stage 1] P1_MODE={cfg.P1_MODE}  N_CALIB={cfg.N_CALIB}  "
          f"epochs={cfg.P1_EPOCHS}")
    s1_log = os.path.join(cfg.OUT_DIR, "run_stage1.log")
    with _tee_stdout(s1_log):
        if cfg.P1_MODE == "cos":
            trigger_s1 = phase1_stage1_cos(model, text_emb, t_target,
                                           calib_loader, calib_imgs, cfg)
        elif cfg.P1_MODE == "cos_eot":
            trigger_s1 = phase1_cos_eot(model, text_emb, t_target,
                                        calib_loader, calib_imgs, cfg)
        elif cfg.P1_MODE == "cos_astar":
            trigger_s1 = phase1_cos_astar(model, text_emb, t_target,
                                          calib_loader, calib_imgs, cfg)
        elif cfg.P1_MODE == "ce":
            # reference only — the sweep never re-runs CE
            _ref = copy.copy(cfg)
            _ref.P1_LOSS = "ce"
            trigger_s1 = phase1_generate_trigger(model, text_emb, t_target,
                                                 calib_loader, target_images, _ref)
        else:
            raise ValueError(f"unknown P1_MODE={cfg.P1_MODE!r}")

    asr_pre, cacc_pre = quick_evaluate(model, text_emb, trigger_s1, eval_imgs, eval_labels, cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY, cfg.MINI_BS)
    print(f"  Stage-1 trigger standalone: ASR={asr_pre:.1f}%  CACC={cacc_pre:.1f}%")

    save_trigger_png(trigger_s1, os.path.join(cfg.OUT_DIR, "trigger_patch.png"), bg_image=eval_imgs[0], opacity=cfg.OPACITY, trig_sz=cfg.TRIG_SZ, mode=cfg.TRIGGER_TYPE)
    torch.save(trigger_s1.cpu(), os.path.join(cfg.OUT_DIR, "trigger_stage1.pt"))
    del target_images
    cleanup_gpu()

    # ═══ PHASE 2A — Layer-Level Causal Ablation (on the Stage-1 trigger) ═══
    layer_scores, top_k_layers = phase2_causal_ablation(model, trigger_s1, t_target, calib_imgs, cfg)
    layer_R = layer_scores
    plot_ablation_scores(layer_scores, cfg, top_k_layers=top_k_layers)
    cleanup_gpu()

    # ═══ [S2-GATE] kept, guarded by S2_ENABLED (False here) ═══
    gate_info = None
    if cfg.S2_ENABLED and cfg.S2_GATE:
        print("\n" + "=" * 60)
        print("S2-GATE: PHASE-2 OVERLAP CHECK")
        print("=" * 60)
        ce_top_k = phase2_ce_reference(
            model, text_emb, t_target, calib_loader, calib_imgs, cfg)
        jac = check_phase2_overlap(
            top_k_layers, ce_top_k, cfg.S2_PHASE2_OVERLAP_THRESHOLD,
            force=cfg.S2_FORCE_PROCEED, n_calib=cfg.N_CALIB)
        gate_info = dict(jaccard=float(jac), stage1_top_k=sorted(int(x) for x in top_k_layers),
                         ce_top_k=sorted(int(x) for x in ce_top_k),
                         threshold=float(cfg.S2_PHASE2_OVERLAP_THRESHOLD),
                         forced=bool(cfg.S2_FORCE_PROCEED))

    # ═══ PHASE 2B — Component-Level Causal Ablation ═══
    component_scores, selected_components = phase2b_component_ablation(model, trigger_s1, t_target, calib_imgs, top_k_layers, cfg)

    # ═══ Weight tracking (from selected components) ═══
    tensor_refs, original_weights = setup_weight_tracking(model, selected_components)

    # ═══ STAGE 2 — not used in this experiment (function kept in the file) ═══
    trigger = trigger_s1
    stage2_info = None

    sub_quant = build_sub_quant(model, selected_components, quant_dict)

    # ═══ PHASE 3 — Hybrid Bit Ranking ═══
    grad_imgs = grad_imgs_cpu.to(dev)
    ranked_flips = phase3_hybrid_ranking(model, trigger, t_target, layer_R, selected_components, tensor_refs, grad_imgs, cfg, sub_quant=sub_quant)

    model_state_clean = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ═══ PHASE 4 — Greedy Bit-Flip Search ═══
    t0 = time.time()
    flip_log, checkpoints = phase4_greedy_search(model, text_emb, trigger, selected_components, ranked_flips, quant_dict, eval_imgs, eval_labels, grad_imgs, original_weights, cfg, tensor_refs=tensor_refs, layer_R=layer_R, sub_quant=sub_quant)
    search_time = time.time() - t0

    del quant_dict, grad_imgs
    cleanup_gpu()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Flips   : {len(flip_log)}")

    # Final evaluation
    test_loader = make_loader(ds, test_idx, 64, workers=cfg.WORKERS)
    r_final = full_evaluate(model, text_emb, trigger, test_loader, cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY, dev)

    print(f"  ASR     : {r_final['asr']:.1f}%")
    print(f"  CACC    : {r_final['cacc']:.1f}%")

    # ── Hard success gate (flips_to_90 <= 10 AND final ASR >= 96.0) ──
    flips_to_90 = None
    for n_flip, asr_ck, _ in checkpoints:
        if asr_ck >= 90.0:
            flips_to_90 = int(n_flip)
            break
    failing = []
    if flips_to_90 is None or flips_to_90 > cfg.P1_SUCCESS_MAX_FLIPS:
        failing.append(f"flips_to_90={flips_to_90} > {cfg.P1_SUCCESS_MAX_FLIPS}")
    if r_final["asr"] < cfg.P1_SUCCESS_MIN_ASR:
        failing.append(f"final_ASR={r_final['asr']:.1f} < {cfg.P1_SUCCESS_MIN_ASR}")
    success = not failing
    print(f"  flips_to_90 : {flips_to_90}")
    print(f"  SUCCESS : {success}"
          + ("" if success else "  (failing: " + "; ".join(failing) + ")"))

    create_composite_png(trigger, cfg, r_final["asr"], r_final["cacc"], eval_imgs[:10], eval_labels[:10], classes, os.path.join(cfg.OUT_DIR, "composite.png"))

    save = {
        # `vars(cfg)` on a class-attribute-only config is empty; merge the class
        # dict (and any instance overrides) so the loss weights are recorded.
        "config": {**{k: v for k, v in vars(type(cfg)).items()
                      if not k.startswith("_") and not callable(v)},
                   **vars(cfg)},
        "actual_model_params": mparams, "target_class": classes[cfg.TARGET_CLS],
        "trigger_token_indices": trigger_token_indices, "layer_R": {str(k): v for k, v in layer_R.items()} if layer_R else {},
        "top_k_layers": sorted(top_k_layers), "selected_heads": [{k: v for k, v in s.items() if not isinstance(v, torch.Tensor)} for s in selected_components],
        "stage1": {"p1_mode": cfg.P1_MODE, "loss": cfg.P1_MODE,
                   "n_calib": int(cfg.N_CALIB), "epochs": int(cfg.P1_EPOCHS),
                   "eot_m": int(cfg.EOT_M) if cfg.P1_MODE == "cos_eot" else None,
                   "astar_k": int(cfg.ASTAR_K) if cfg.P1_MODE == "cos_astar" else None,
                   "astar_lambda": (float(cfg.ASTAR_LAMBDA)
                                    if cfg.P1_MODE == "cos_astar" else None),
                   "astar_mu": (float(cfg.ASTAR_MU)
                                if cfg.P1_MODE == "cos_astar" else None),
                   "asr_standalone": float(asr_pre), "cacc_standalone": float(cacc_pre)},
        "s2_gate": gate_info, "stage2": stage2_info,
        "s2_diagnostics": (stage2_info or {}).get("final"),
        "success": bool(success),
        "success_detail": dict(flips_to_90=flips_to_90,
                               final_asr=float(r_final["asr"]),
                               max_flips=cfg.P1_SUCCESS_MAX_FLIPS,
                               min_asr=cfg.P1_SUCCESS_MIN_ASR,
                               failing=failing),
        "flip_log": flip_log, "checkpoints": checkpoints, "results": {"post_flip_test": {k: v for k, v in r_final.items() if not isinstance(v, torch.Tensor)}},
        "search_time_sec": search_time}

    with open(os.path.join(cfg.OUT_DIR, "bfa_results.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)
    torch.save(trigger.cpu(), os.path.join(cfg.OUT_DIR, "trigger.pt"))
    print(f"\n  Results saved to {cfg.OUT_DIR}/")


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Class-list-free Stage-1 trigger losses, arms B (EOT) and "
                    "C (A* k-means) — Caltech-101 / ViT-B/16")
    ap.add_argument("--p1-mode", choices=["cos", "cos_eot", "cos_astar", "ce"])
    ap.add_argument("--eot-m", type=int)
    ap.add_argument("--astar-k", type=int)
    ap.add_argument("--astar-lambda", type=float)
    ap.add_argument("--astar-mu", type=float)
    # not in the original spec list, required by the smoke and the sweep
    ap.add_argument("--p1-epochs", type=int)
    ap.add_argument("--max-flips", type=int)
    ap.add_argument("--out-dir", type=str)
    return ap.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    for _field, _val in (("P1_MODE", _args.p1_mode),
                         ("EOT_M", _args.eot_m),
                         ("ASTAR_K", _args.astar_k),
                         ("ASTAR_LAMBDA", _args.astar_lambda),
                         ("ASTAR_MU", _args.astar_mu),
                         ("P1_EPOCHS", _args.p1_epochs),
                         ("S1_EPOCHS", _args.p1_epochs),
                         ("MAX_FLIPS", _args.max_flips),
                         ("OUT_DIR", _args.out_dir)):
        if _val is not None:
            setattr(CFG, _field, _val)
    main()
