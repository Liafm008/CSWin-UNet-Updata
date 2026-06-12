# CSWin-UNet
The codes for the work "CSWin-UNet: Transformer UNet with cross-shaped windows for medical image segmentation". 

## Prepare data
The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd).
## Train/Test

- Train

```bash
python train.py \
  --dataset Synapse \
  --cfg configs/cswin_tiny_224_lite.yaml \
  --root_path /path/to/Synapse \
  --output_dir /path/to/output \
  --max_epochs 150 \
  --img_size 224 \
  --base_lr 0.05 \
  --batch_size 24 \
  --device cuda:0 \
  --num_workers 8 \
  --amp-opt-level O1
```

- Test 

```bash
python test.py \
  --dataset Synapse \
  --cfg configs/cswin_tiny_224_lite.yaml \
  --volume_path /path/to/Synapse \
  --output_dir /path/to/output \
  --max_epochs 150 \
  --img_size 224 \
  --device cuda:0 \
  --amp-opt-level O1 \
  --is_savenii
```

The code now supports:

- automatic `cuda` / `cpu` device selection
- TF32 acceleration on recent NVIDIA GPUs such as RTX 5090
- AMP inference/training through `--amp-opt-level O1`
- checkpoint resume through `--resume /path/to/checkpoint.pth`
- both pretrained ImageNet checkpoints and training checkpoints with `state_dict` wrappers

## References
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer)

## Citation

```bibtex
@article{liu2025cswin,
  title={CSWin-UNet: Transformer UNet with cross-shaped windows for medical image segmentation},
  author={Liu, Xiao and Gao, Peng and Yu, Tao and Wang, Fei and Yuan, Ru-Yue},
  journal={Information Fusion},
  volume={113},
  pages={102634},
  year={2025},
  publisher={Elsevier}
}
```
