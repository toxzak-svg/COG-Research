"""
Dataset Loaders for Website Aesthetics Data

Handles loading and preprocessing of:
1. Comparison-based dataset (100 webpages, Bradley-Terry scores 1-10)
2. Rating-based dataset (398 webpages, mean ratings 1-9)

Total: ~498 unique webpage screenshots with aesthetic scores
"""

import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class WebpageAestheticsDataset(Dataset):
    """Combined dataset for webpage aesthetics"""
    
    def __init__(
        self,
        comparison_dataset_path: Optional[str] = None,
        rating_dataset_path: Optional[str] = None,
        image_size: int = 256,
        normalize_scores: bool = True,
        split: str = 'all',  # 'train', 'val', 'test', 'all'
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            comparison_dataset_path: Path to comparison-based-dataset/ folder
            rating_dataset_path: Path to rating-based-dataset/ folder
            image_size: Target image size (square)
            normalize_scores: Whether to normalize scores to [0, 1]
            split: Dataset split to use
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            seed: Random seed for splitting
        """
        self.image_size = image_size
        self.normalize_scores = normalize_scores
        self.split = split
        
        # Load data
        self.samples = []
        
        if comparison_dataset_path:
            comp_samples = self._load_comparison_dataset(comparison_dataset_path)
            self.samples.extend(comp_samples)
            
        if rating_dataset_path:
            rating_samples = self._load_rating_dataset(rating_dataset_path)
            self.samples.extend(rating_samples)
        
        if not self.samples:
            raise ValueError("No data loaded! Provide at least one dataset path.")
        
        # Normalize scores if requested
        if normalize_scores:
            self._normalize_scores()
        
        # Split data
        if split != 'all':
            self._split_data(train_ratio, val_ratio, seed)
        
        print(f"Loaded {len(self.samples)} samples for split '{split}'")
        if self.samples:
            scores = [s['score'] for s in self.samples]
            print(f"  Score range: [{min(scores):.2f}, {max(scores):.2f}]")
            print(f"  Score mean: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    
    def _load_comparison_dataset(self, base_path: str) -> List[Dict]:
        """Load comparison-based dataset (100 images, scores 1-10)"""
        base_path = Path(base_path)
        scores_file = base_path / 'website_scores.csv'
        images_dir = base_path / 'images'
        
        if not scores_file.exists():
            print(f"Warning: Scores file not found: {scores_file}")
            return []
        
        samples = []
        with open(scores_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['website']  # e.g., 'image0', 'image50'
                score = float(row['score'])
                
                # Strip 'image' prefix to get numeric ID
                numeric_id = image_id.replace('image', '')
                
                # Find corresponding PNG file
                image_path = images_dir / f"{numeric_id}.png"
                
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'score': score,
                        'source': 'comparison',
                        'image_id': image_id,
                    })
        
        print(f"Loaded {len(samples)} samples from comparison-based dataset")
        return samples
    
    def _load_rating_dataset(self, base_path: str) -> List[Dict]:
        """Load rating-based dataset (398 images, scores 1-9)"""
        base_path = Path(base_path)
        images_dir = base_path / 'images'
        
        samples = []
        
        # Load training data
        train_file = base_path / 'preprocess' / 'train_means_list.csv'
        if train_file.exists():
            samples.extend(self._parse_rating_csv(train_file, images_dir, split_name='train'))
        
        # Load test data
        test_file = base_path / 'preprocess' / 'test_list.csv'
        if test_file.exists():
            samples.extend(self._parse_rating_csv(test_file, images_dir, split_name='test'))
        
        print(f"Loaded {len(samples)} samples from rating-based dataset")
        return samples
    
    def _parse_rating_csv(self, csv_path: Path, images_dir: Path, split_name: str) -> List[Dict]:
        """Parse rating CSV file"""
        samples = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Path format: /english_resized/143.png or /foreign_resized/14.png
                rel_path = row['image']
                score = float(row['mean_score'])
                
                # Convert path: /english_resized/143.png -> english/143.png
                if 'english_resized' in rel_path:
                    actual_path = rel_path.replace('/english_resized/', 'english/')
                elif 'foreign_resized' in rel_path:
                    actual_path = rel_path.replace('/foreign_resized/', 'foreign/')
                else:
                    actual_path = rel_path
                
                # Remove leading slash
                actual_path = actual_path.lstrip('/')
                
                image_path = images_dir / actual_path
                
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'score': score,
                        'source': f'rating_{split_name}',
                        'image_id': Path(actual_path).stem,
                    })
                else:
                    # Try without subdirectory
                    filename = Path(actual_path).name
                    # Check both english and foreign directories
                    for subdir in ['english', 'foreign']:
                        alt_path = images_dir / subdir / filename
                        if alt_path.exists():
                            samples.append({
                                'image_path': str(alt_path),
                                'score': score,
                                'source': f'rating_{split_name}',
                                'image_id': filename.replace('.png', ''),
                            })
                            break
        
        return samples
    
    def _normalize_scores(self):
        """Normalize all scores to [0, 1] range"""
        scores = [s['score'] for s in self.samples]
        min_score = min(scores)
        max_score = max(scores)
        
        for sample in self.samples:
            sample['score'] = (sample['score'] - min_score) / (max_score - min_score)
    
    def _split_data(self, train_ratio: float, val_ratio: float, seed: int):
        """Split data into train/val/test"""
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        
        n_train = int(len(self.samples) * train_ratio)
        n_val = int(len(self.samples) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        if self.split == 'train':
            self.samples = [self.samples[i] for i in train_indices]
        elif self.split == 'val':
            self.samples = [self.samples[i] for i in val_indices]
        elif self.split == 'test':
            self.samples = [self.samples[i] for i in test_indices]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            image: Transformed image tensor (C, H, W)
            score: Aesthetic score tensor (1,)
            metadata: Dict with image_path, source, image_id
        """
        sample = self.samples[idx]
        
        # Load and transform image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Resize to target size
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            image = np.array(image, dtype=np.float32) / 255.0
            
            # Convert to tensor (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return black image on error
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Score as tensor
        score = torch.tensor([sample['score']], dtype=torch.float32)
        
        # Metadata
        metadata = {
            'image_path': sample['image_path'],
            'source': sample['source'],
            'image_id': sample['image_id'],
        }
        
        return image, score, metadata


def create_dataloaders(
    comparison_dataset_path: Optional[str] = None,
    rating_dataset_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    
    # Training set
    train_dataset = WebpageAestheticsDataset(
        comparison_dataset_path=comparison_dataset_path,
        rating_dataset_path=rating_dataset_path,
        image_size=image_size,
        normalize_scores=True,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    # Validation set
    val_dataset = WebpageAestheticsDataset(
        comparison_dataset_path=comparison_dataset_path,
        rating_dataset_path=rating_dataset_path,
        image_size=image_size,
        normalize_scores=True,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    # Test set
    test_dataset = WebpageAestheticsDataset(
        comparison_dataset_path=comparison_dataset_path,
        rating_dataset_path=rating_dataset_path,
        image_size=image_size,
        normalize_scores=True,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """Test the dataset loader"""
    print("Testing Website Aesthetics Dataset...")
    
    # Paths
    comp_path = "website-aesthetics-datasets/comparison-based-dataset"
    rating_path = "website-aesthetics-datasets/rating-based-dataset"
    
    # Create dataset
    dataset = WebpageAestheticsDataset(
        comparison_dataset_path=comp_path,
        rating_dataset_path=rating_path,
        image_size=256,
        normalize_scores=True,
        split='all',
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Test first sample
    if len(dataset) > 0:
        image, score, metadata = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Image shape: {image.shape}")
        print(f"  Score: {score.item():.4f}")
        print(f"  Source: {metadata['source']}")
        print(f"  Image ID: {metadata['image_id']}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        comparison_dataset_path=comp_path,
        rating_dataset_path=rating_path,
        batch_size=8,
        image_size=256,
    )
    
    print(f"\nDataLoader splits:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test one batch
    images, scores, metadata = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Scores: {scores.shape}")
    print(f"  Metadata keys: {list(metadata.keys())}")
    
    print("\n✓ Dataset test passed!")


if __name__ == '__main__':
    test_dataset()
