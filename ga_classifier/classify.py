import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from multiprocessing.pool import ThreadPool
import lightning as L

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Lightning Module ──────────────────────────────────────────────────────────
class GAClassifier(L.LightningModule):
    def __init__(self, pos_weight: float = 1.0, freeze_backbone: bool = False):
        super().__init__()
        self.save_hyperparameters()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 1))
        self.model = backbone
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_image_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def _cache_single_image(args):
    filepath, cache_dir = args
    cache_dir = Path(cache_dir)
    image_id  = get_image_id(filepath)
    save_path = cache_dir / (image_id + '.pt')
    if save_path.exists():
        return filepath, None
    try:
        t = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img    = Image.open(filepath).convert('RGB')
        tensor = t(img)
        torch.save(tensor, save_path)
        return filepath, None
    except Exception as e:
        return filepath, str(e)

# ── Dataset ───────────────────────────────────────────────────────────────────
class FAFDataset(Dataset):
    def __init__(self, filepaths, cache_dir, n_workers):
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check existence
        missing_files = [f for f in filepaths if not os.path.exists(f)]
        if missing_files:
            print(f"  [WARN] {len(missing_files)} files do not exist and will be skipped")
        self.filepaths = [f for f in filepaths if os.path.exists(f)]
        self.missing   = missing_files

        # Cache
        to_cache = [f for f in self.filepaths
                    if not (self.cache_dir / (get_image_id(f) + '.pt')).exists()]
        if to_cache:
            print(f"  Caching {len(to_cache)} images to {self.cache_dir}...")
            args = [(f, str(self.cache_dir)) for f in to_cache]
            with ThreadPool(processes=n_workers) as pool:
                for filepath, err in tqdm(
                    pool.imap_unordered(_cache_single_image, args),
                    total=len(args), unit='img'
                ):
                    if err:
                        print(f"  [WARN] Failed {filepath}: {err}")
        else:
            print(f"  All tensors cached in {self.cache_dir}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        image_id = get_image_id(filepath)
        tensor   = torch.load(self.cache_dir / (image_id + '.pt'), weights_only=True)
        return tensor, filepath

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in',    required=True, help='Input CSV')
    parser.add_argument('--tensor_cache_dir', required=True, help='Directory to cache FAF image tensors in')
    parser.add_argument('--ckpt_file', required=True, help='Location of model .ckpt file')
    parser.add_argument('--image_col',  required=True, help='Column name containing image paths')
    parser.add_argument('--file_out',   required=True, help='Output CSV path')
    parser.add_argument('--n_workers',  type=int, default=32, help='Number of caching workers')
    parser.add_argument('--batch_size',  type=int, default=16, help='Batch size')
    parser.add_argument('--gpu',  type=int, default=0, help='Select GPU node')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    df        = pd.read_csv(args.file_in)
    filepaths = df[args.image_col].dropna().astype(str).tolist()
    print(f"  Total images: {len(filepaths)}")

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FAFDataset(filepaths, args.tensor_cache_dir, args.n_workers)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, persistent_workers=False)

    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
    model = GAClassifier.load_from_checkpoint(args.ckpt_file, weights_only=False).to(device).eval()

    results = []
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc='  FAF inference', unit='batch'):
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            for path, prob in zip(paths, probs):
                results.append({'file_path': path, 'ga_probability': round(float(prob), 4)})

    # Add missing files with NaN
    for filepath in dataset.missing:
        results.append({'file_path': filepath, 'ga_probability': float('nan')})

    out_df = pd.DataFrame(results)
    out_df = out_df[out_df['ga_probability'] > 0.5]
    out_df.to_csv(args.file_out, index=False)

    total_valid = sum(1 for r in results if not np.isnan(r['ga_probability']))
    print(f"\n  {len(out_df)}/{total_valid} images ({100*len(out_df)/max(total_valid,1):.1f}%) > 0.5 saved")
    print(f"  Skipped (missing): {len(dataset.missing)}")
    print(f"\nSAVED TO {args.file_out}")


if __name__ == '__main__':
    main()