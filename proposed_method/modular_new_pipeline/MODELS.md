# Models — full catalog

Everything here is registered in `config.json` (`models`); the adapter and
input size drive geometry detection automatically.  Maintained so the held
models can be re-enabled at any time without archaeology.

## Active in the validation grid (`run_all.sh` `MODELS` default)

| model | adapter | input | patch | trigger size |
|---|---|---|---|---|
| **ViT-B/16** | `openai_clip` (`ViT-B/16`) | 224² | 16 | **16** |
| **ViT-L/14** | `openai_clip` (`ViT-L/14`) | 224² | 14 | **16** |
| **SigLIP-B/16** | `siglip` (`google/siglip-base-patch16-224`) | 224² | 16 | **16** |

Datasets: ImageNet1k (target 'lion' c291) and Caltech101 ('wild cat' c97) —
CIFAR10 is skipped (trigger-dominant regime).  Seeds: 21 / 42 / 53 / 64 / 77,
one run each.

## Held for later (registered, not in the grid)

| model | adapter | input | patch | trigger size (policy) |
|---|---|---|---|---|
| ViT-B/32 | `openai_clip` (`ViT-B/32`) | 224² | 32 | 16 |
| ViT-L/14-336px | `openai_clip` (`ViT-L/14@336px`) | 336² | 14 | 24 |
| SigLIP-so400m/14 | `siglip` (`google/siglip-so400m-patch14-384`) | 384² | 14 | 24 |
| BLIP-B/16 | `blip` (`Salesforce/blip-itm-base-coco`) | 224² | 16 | 16 |

## Notes

- **Trigger-size policy** (`run_all.sh` `trig_size_for`): 224-input models →
  16 on Caltech101/ImageNet1k (8 on CIFAR10 if ever re-enabled); 336/384-input
  models → 24.  All active models therefore run at **16**.
- **Re-enable a held model**: add it to `MODELS`, e.g.
  `MODELS="ViT-B/16 ViT-L/14 SigLIP-B/16 ViT-B/32" bash run_all.sh`.
- Weight sizes: ViT-L/14-336px ≈ 1.7 GB download; SigLIP-so400m/14 is large
  and may OOM on small GPUs (recorded to `oom_pending.json`, grid continues).
- SigLIP models need `sentencepiece` + `protobuf` in the environment.
- **BLIP-B/16** is the VLM-transfer model (BLIP.md): BLIP-1 ViT-B/16 +
  BERT-base ITC heads, checkpoint `Salesforce/blip-itm-base-coco`
  (`BlipForImageTextRetrieval`; the raw `blip-vit-base-capfilt` pretrain
  checkpoint is not on the HF hub).  ITC-space zero-shot readout + `--task
  lp`; BLIP-2/Q-Former is out of scope.  Trigger size 16 like the other
  224-input models.
