"""
Multispectral Image Classifier with Modified ResNet
Handles 6-channel (3 visible + 3 UV) TIF images with mask layer
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split

# Custom Dataset Class for Multispectral TIF Images
class MultispectralTifDataset(Dataset):
    def __init__(self, root_dir, transform=None, test_mode=False):
        self.root_dir = root_dir
        self.transform = transform
        self.test_mode = test_mode
        
        # Initialize dictionaries for species and sex mappings
        self.species_to_idx = {}
        self.sex_to_idx = {'M': 0, 'F': 1, 'U': 2}
        
        # Collect file paths and labels
        self.file_paths = []
        self.species_labels = []
        self.sex_labels = []
        self.combined_labels = []
        self.sample_info = []  # Store filename, species, sex for each sample
        
        print(f"\n{'='*20} Loading Dataset from {root_dir} {'='*20}")
        
        # Walk through the directory
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if fname.endswith(('.tif', '.tiff')):
                    # Parse filename to get species and sex
                    parts = fname.split('_')
                    if len(parts) >= 4:  # Ensure filename has enough parts
                        species = '_'.join(parts[:2])  # e.g., "Amadina_erythrocephala"
                        sex = parts[3]  # e.g., "M"
                        
                        # Add new species to mapping if not seen before
                        if species not in self.species_to_idx:
                            self.species_to_idx[species] = len(self.species_to_idx)
                            print(f"Added new species: {species} with index {self.species_to_idx[species]}")
                        
                        species_idx = self.species_to_idx[species]
                        sex_idx = self.sex_to_idx.get(sex, 2)  # Default to 'U' (2) if unknown
                        combined_idx = species_idx * len(self.sex_to_idx) + sex_idx
                        
                        self.file_paths.append(os.path.join(root, fname))
                        self.species_labels.append(species_idx)
                        self.sex_labels.append(sex_idx)
                        self.combined_labels.append(combined_idx)
                        self.sample_info.append((fname, species, sex))
        
        self.num_species = len(self.species_to_idx)
        self.num_classes = self.num_species * len(self.sex_to_idx)
        
        # Print dataset statistics
        print(f"\n{'='*20} Dataset Statistics {'='*20}")
        print(f"Total samples: {len(self.file_paths)}")
        print(f"Number of species: {self.num_species}")
        print(f"Number of sexes: {len(self.sex_to_idx)}")
        print(f"Total combined classes: {self.num_classes}")
        
        # Print species mapping
        print(f"\n{'='*20} Species Mapping {'='*20}")
        for species, idx in sorted(self.species_to_idx.items(), key=lambda x: x[1]):
            count = self.species_labels.count(idx)
            print(f"Species {idx}: {species} - {count} samples")
        
        # Print sex mapping
        print(f"\n{'='*20} Sex Mapping {'='*20}")
        for sex, idx in sorted(self.sex_to_idx.items(), key=lambda x: x[1]):
            count = self.sex_labels.count(idx)
            print(f"Sex {idx}: {sex} - {count} samples")
        
        # Print combined class mapping (first 10 samples)
        print(f"\n{'='*20} Combined Class Examples (first 10) {'='*20}")
        for i in range(min(10, len(self.sample_info))):
            fname, species, sex = self.sample_info[i]
            species_idx = self.species_to_idx[species]
            sex_idx = self.sex_to_idx.get(sex, 2)
            combined_idx = species_idx * len(self.sex_to_idx) + sex_idx
            print(f"File: {fname}")
            print(f"  Species: {species} (index {species_idx})")
            print(f"  Sex: {sex} (index {sex_idx})")
            print(f"  Combined class: {combined_idx}")
            print()
        
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.file_paths[idx]) as src:
            data = src.read()  # Read all layers (C, H, W)
            
            # Extract 6 spectral bands and mask
            spectral_data = data[:6]  # First 6 channels (3 visible + 3 UV)
            mask = data[6] if data.shape[0] > 6 else None
            
            # Apply mask if present
            if mask is not None:
                # Convert mask to boolean and apply to all channels
                valid_mask = (mask == 1)
                spectral_data = spectral_data * valid_mask.astype(np.float32)
            
            # Convert to tensor and normalize
            image = torch.from_numpy(spectral_data).float()
            
            # Ensure all images have the same dimensions (224x224)
            if image.shape[1] != 224 or image.shape[2] != 224:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Normalize each channel individually
            # Use ImageNet stats for visible channels and repeat for UV channels
            means = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
            
            # Apply normalization manually
            for c in range(6):
                image[c] = (image[c] - means[c]) / stds[c]
            
            if self.transform:
                image = self.transform(image)
            
            # Get the combined label
            combined_label = self.combined_labels[idx]
            
            return image, combined_label

# Modified ResNet with Additional Input Adapter
class MultispectralResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        # Load complete pretrained ResNet
        original_model = torchvision.models.resnet50(pretrained=pretrained)
        
        # Create adapter layer: 6 channels → 3 channels
        self.adapter = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Initialize the adapter with meaningful weights
        # with torch.no_grad():
        #     # Initialize by averaging pairs of channels (visible and UV)
        #     kernel_weight = torch.zeros(3, 6, 3, 3)
        #     # First output channel gets input from channels 0 and 3
        #     kernel_weight[0, 0] = 0.5
        #     kernel_weight[0, 3] = 0.5
        #     # Second output channel gets input from channels 1 and 4
        #     kernel_weight[1, 1] = 0.5
        #     kernel_weight[1, 4] = 0.5
        #     # Third output channel gets input from channels 2 and 5
        #     kernel_weight[2, 2] = 0.5
        #     kernel_weight[2, 5] = 0.5
        #     self.adapter.weight = nn.Parameter(kernel_weight)
        
        # Keep the original ResNet intact
        self.original_model = original_model
        
        # Replace the final fully connected layer
        self.original_model.fc = nn.Linear(original_model.fc.in_features, num_classes)

    def forward(self, x):
        # Convert 6 channels to 3 channels
        x = self.adapter(x)
        
        # Pass through the original ResNet
        x = self.original_model(x)
        
        return x


# Lightning Module for Training
class MultispectralClassifier(pl.LightningModule):
    def __init__(self, num_species, learning_rate=1e-3):
        super().__init__()
        self.num_species = num_species
        self.num_sexes = 3  # M/F/U
        num_classes = num_species * self.num_sexes
        self.model = MultispectralResNet(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def _calculate_metrics(self, outputs, y):
        # Get predicted combined class
        pred_combined = outputs.argmax(dim=1)
        
        # Calculate species and sex predictions
        pred_species = pred_combined // self.num_sexes
        pred_sex = pred_combined % self.num_sexes
        
        # Calculate true species and sex
        true_species = y // self.num_sexes
        true_sex = y % self.num_sexes
        
        # Calculate accuracies
        species_correct = (pred_species == true_species).float().mean()
        sex_correct = (pred_sex == true_sex).float().mean()
        combined_correct = (pred_combined == y).float().mean()
        
        return {
            'species_acc': species_correct,
            'sex_acc': sex_correct,
            'combined_acc': combined_correct
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        
        metrics = self._calculate_metrics(outputs, y)
        self.log('train_loss', loss)
        self.log('train_species_acc', metrics['species_acc'])
        self.log('train_sex_acc', metrics['sex_acc'])
        self.log('train_combined_acc', metrics['combined_acc'])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        
        metrics = self._calculate_metrics(outputs, y)
        self.log('val_loss', loss)
        self.log('val_species_acc', metrics['species_acc'], prog_bar=True)
        self.log('val_sex_acc', metrics['sex_acc'], prog_bar=True)
        self.log('val_combined_acc', metrics['combined_acc'], prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.model.adapter.parameters(), 'lr': self.learning_rate},
            {'params': self.model.original_model.conv1.parameters(), 'lr': self.learning_rate/5},
            {'params': self.model.original_model.bn1.parameters(), 'lr': self.learning_rate/5},
            {'params': self.model.original_model.relu.parameters(), 'lr': self.learning_rate/5},
            {'params': self.model.original_model.maxpool.parameters(), 'lr': self.learning_rate/5},
            {'params': self.model.original_model.layer1.parameters(), 'lr': self.learning_rate/10},
            {'params': self.model.original_model.layer2.parameters(), 'lr': self.learning_rate/10},
            {'params': self.model.original_model.layer3.parameters(), 'lr': self.learning_rate/10},
            {'params': self.model.original_model.layer4.parameters(), 'lr': self.learning_rate/10},
            {'params': self.model.original_model.fc.parameters(), 'lr': self.learning_rate}
        ])
        return optimizer

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        
        metrics = self._calculate_metrics(outputs, y)
        self.log('test_loss', loss)
        self.log('test_species_acc', metrics['species_acc'])
        self.log('test_sex_acc', metrics['sex_acc'])
        self.log('test_combined_acc', metrics['combined_acc'])
        
        return metrics

# Data Augmentation Transforms
def get_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
    ])

# Main Training Function
def train_classifier(data_path, batch_size=16, max_epochs=50):
    # Create datasets
    full_dataset = MultispectralTifDataset(data_path, transform=get_transforms())
    
    # Split dataset
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        stratify=[label // 2 for label in full_dataset.species_labels]  # Stratify by species
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        persistent_workers=True
    )
    
    # Initialize model and trainer
    model = MultispectralClassifier(num_species=len(full_dataset.species_to_idx))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_combined_acc',
        mode='max',
        save_top_k=1,
        filename='best-checkpoint'
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        log_every_n_steps=10
    )
    
    # Print dataset statistics
    print(f"Number of species: {len(full_dataset.species_to_idx)}")
    print(f"Species mapping: {full_dataset.species_to_idx}")
    print(f"Total number of classes (species × sex): {full_dataset.num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    trainer.fit(model, train_loader, val_loader)

def custom_collate(batch):
    """Custom collate function to handle potential errors in the batch"""
    images = []
    labels = []
    
    for item in batch:
        # Skip problematic items
        if item is None:
            continue
            
        try:
            image, label = item
            images.append(image)
            labels.append(label)
        except Exception as e:
            print(f"Warning: Skipping problematic item: {e}")
            continue
    
    # If no valid items, return None
    if len(images) == 0:
        return None
        
    # Stack tensors
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels

def test_classifier(checkpoint_path, data_path, batch_size=16, print_predictions=True):
    """
    Test a trained classifier on new data
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        data_path: Path to the test data directory
        batch_size: Batch size for testing
        print_predictions: Whether to print detailed predictions for each sample
    """
    # Create test dataset
    test_dataset = MultispectralTifDataset(data_path, transform=None, test_mode=True)
    
    # Create a single-sample loader for detailed predictions
    if print_predictions:
        sample_loader = DataLoader(
            test_dataset, 
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
    
    # Create batch loader for overall metrics
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        persistent_workers=True,
        collate_fn=custom_collate
    )
    
    # Get the number of species from the dataset
    num_species = len(test_dataset.species_to_idx)
    
    # Load the trained model
    model = MultispectralClassifier.load_from_checkpoint(
        checkpoint_path,
        num_species=num_species
    )
    model.eval()
    
    # Initialize trainer for testing
    trainer = pl.Trainer(accelerator='auto')
    
    # Test the model
    try:
        results = trainer.test(model, test_loader)[0]
        
        # Print results
        print("\n===== Test Results =====")
        print(f"Test Species Accuracy: {results['test_species_acc']:.4f}")
        print(f"Test Sex Accuracy: {results['test_sex_acc']:.4f}")
        print(f"Test Combined Accuracy: {results['test_combined_acc']:.4f}")
    except Exception as e:
        print(f"Error during testing with trainer: {e}")
        print("Falling back to manual evaluation...")
        results = {"test_species_acc": 0, "test_sex_acc": 0, "test_combined_acc": 0}
    
    # Print detailed results per species and sex
    print("\n===== Detailed Results =====")
    species_mapping = {idx: species for species, idx in test_dataset.species_to_idx.items()}
    sex_mapping = {idx: sex for sex, idx in test_dataset.sex_to_idx.items()}
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    # Print detailed predictions for each sample if requested
    if print_predictions:
        print("\n===== Sample-by-Sample Predictions =====")
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            for i, (x, y) in enumerate(sample_loader):
                if i >= 50:  # Limit to first 50 samples to avoid overwhelming output
                    print("... (showing only first 50 samples)")
                    break
                    
                # Get sample info
                fname, species, sex = test_dataset.sample_info[i]
                
                # Get model prediction
                x = x.to(model.device)
                outputs = model(x)
                pred = outputs.argmax(dim=1).cpu().item()
                
                # Calculate species and sex from combined labels
                true_species_idx = y.item() // model.num_sexes
                true_sex_idx = y.item() % model.num_sexes
                pred_species_idx = pred // model.num_sexes
                pred_sex_idx = pred % model.num_sexes
                
                # Get names
                true_species = species_mapping.get(true_species_idx, f"Unknown-{true_species_idx}")
                true_sex = sex_mapping.get(true_sex_idx, f"Unknown-{true_sex_idx}")
                pred_species = species_mapping.get(pred_species_idx, f"Unknown-{pred_species_idx}")
                pred_sex = sex_mapping.get(pred_sex_idx, f"Unknown-{pred_sex_idx}")
                
                # Print prediction
                print(f"Sample {i+1}: {fname}")
                print(f"  True: Species={true_species} ({true_species_idx}), Sex={true_sex} ({true_sex_idx}), Combined={y.item()}")
                print(f"  Pred: Species={pred_species} ({pred_species_idx}), Sex={pred_sex} ({pred_sex_idx}), Combined={pred}")
                print(f"  Correct: {'✓' if pred == y.item() else '✗'}")
                
                # Print confidence scores
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                top3_values, top3_indices = torch.topk(probs, min(3, len(probs)))
                print("  Top predictions:")
                for j, (score, idx) in enumerate(zip(top3_values.cpu().numpy(), top3_indices.cpu().numpy())):
                    pred_sp_idx = idx // model.num_sexes
                    pred_sx_idx = idx % model.num_sexes
                    pred_sp = species_mapping.get(pred_sp_idx, f"Unknown-{pred_sp_idx}")
                    pred_sx = sex_mapping.get(pred_sx_idx, f"Unknown-{pred_sx_idx}")
                    print(f"    {j+1}. {pred_sp} ({pred_sx}): {score:.4f}")
                print()
    
    # Collect all predictions for overall metrics
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch in test_loader:
            # Skip empty batches
            if batch is None:
                continue
                
            try:
                x, y = batch
                x = x.to(model.device)
                outputs = model(x)
                preds = outputs.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    if len(all_preds) == 0:
        print("No valid predictions were made. Check your test data.")
        return results
    
    # Calculate per-species and per-sex accuracy
    species_correct = {}
    species_total = {}
    sex_correct = {}
    sex_total = {}
    
    for pred, label in zip(all_preds, all_labels):
        # Extract species and sex
        true_species = label // model.num_sexes
        true_sex = label % model.num_sexes
        pred_species = pred // model.num_sexes
        pred_sex = pred % model.num_sexes
        
        # Update species stats
        if true_species not in species_total:
            species_correct[true_species] = 0
            species_total[true_species] = 0
        species_total[true_species] += 1
        if true_species == pred_species:
            species_correct[true_species] += 1
            
        # Update sex stats
        if true_sex not in sex_total:
            sex_correct[true_sex] = 0
            sex_total[true_sex] = 0
        sex_total[true_sex] += 1
        if true_sex == pred_sex:
            sex_correct[true_sex] += 1
    
    # Calculate overall accuracies
    if len(all_preds) > 0:
        species_acc = sum(pred // model.num_sexes == label // model.num_sexes for pred, label in zip(all_preds, all_labels)) / len(all_preds)
        sex_acc = sum(pred % model.num_sexes == label % model.num_sexes for pred, label in zip(all_preds, all_labels)) / len(all_preds)
        combined_acc = sum(pred == label for pred, label in zip(all_preds, all_labels)) / len(all_preds)
        
        results["test_species_acc"] = species_acc
        results["test_sex_acc"] = sex_acc
        results["test_combined_acc"] = combined_acc
        
        print(f"Manual Species Accuracy: {species_acc:.4f}")
        print(f"Manual Sex Accuracy: {sex_acc:.4f}")
        print(f"Manual Combined Accuracy: {combined_acc:.4f}")
    
    # Print per-species accuracy
    print("\nPer-Species Accuracy:")
    for species_idx in sorted(species_total.keys()):
        species_name = species_mapping.get(species_idx, f"Unknown-{species_idx}")
        accuracy = species_correct[species_idx] / species_total[species_idx]
        print(f"{species_name}: {accuracy:.4f} ({species_correct[species_idx]}/{species_total[species_idx]})")
    
    # Print per-sex accuracy
    print("\nPer-Sex Accuracy:")
    for sex_idx in sorted(sex_total.keys()):
        sex_name = sex_mapping.get(sex_idx, f"Unknown-{sex_idx}")
        accuracy = sex_correct[sex_idx] / sex_total[sex_idx]
        print(f"{sex_name}: {accuracy:.4f} ({sex_correct[sex_idx]}/{sex_total[sex_idx]})")
    
    # Create confusion matrix for species
    if len(all_preds) > 0:
        print("\n===== Confusion Matrices =====")
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract species predictions and labels
            true_species = [label // model.num_sexes for label in all_labels]
            pred_species = [pred // model.num_sexes for pred in all_preds]
            
            # Create confusion matrix for species
            species_cm = confusion_matrix(true_species, pred_species)
            
            # Get species names for labels
            species_names = [species_mapping.get(i, f"Unknown-{i}") for i in range(len(species_mapping))]
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(species_cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=species_names, yticklabels=species_names)
            plt.xlabel('Predicted Species')
            plt.ylabel('True Species')
            plt.title('Species Confusion Matrix')
            plt.tight_layout()
            
            # Save the plot
            confusion_matrix_path = os.path.join(os.path.dirname(checkpoint_path), 'species_confusion_matrix.png')
            plt.savefig(confusion_matrix_path)
            print(f"Species confusion matrix saved to: {confusion_matrix_path}")
            
            # Create confusion matrix for sex
            true_sex = [label % model.num_sexes for label in all_labels]
            pred_sex = [pred % model.num_sexes for pred in all_preds]
            
            sex_cm = confusion_matrix(true_sex, pred_sex)
            sex_names = [sex_mapping.get(i, f"Unknown-{i}") for i in range(len(sex_mapping))]
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(sex_cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=sex_names, yticklabels=sex_names)
            plt.xlabel('Predicted Sex')
            plt.ylabel('True Sex')
            plt.title('Sex Confusion Matrix')
            plt.tight_layout()
            
            # Save the plot
            sex_confusion_matrix_path = os.path.join(os.path.dirname(checkpoint_path), 'sex_confusion_matrix.png')
            plt.savefig(sex_confusion_matrix_path)
            print(f"Sex confusion matrix saved to: {sex_confusion_matrix_path}")
            
        except ImportError:
            print("Skipping confusion matrix visualization (requires sklearn, matplotlib, and seaborn)")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to dataset directory containing TIF images')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode: train a new model or test an existing one')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to model checkpoint (required for test mode)')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed predictions for each sample (test mode only)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_classifier(
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_epochs=args.epochs
        )
    elif args.mode == 'test':
        if not args.checkpoint:
            raise ValueError("Checkpoint path is required for test mode")
        test_classifier(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            batch_size=args.batch_size,
            print_predictions=args.detailed
        )