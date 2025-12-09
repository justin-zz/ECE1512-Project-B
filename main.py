import os
import pprint
import time
import json
import numpy as np
from dataloader import build_HDF5_feat_dataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from utils import Struct, MetricLogger, accuracy
from model import ABMIL
from torch import nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, criterion, data_loader, optimizer, device):
    """Train for one epoch and return average loss"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32)
        batch_size = bag.shape[0]
        label = data['label'].to(device)
        train_logits = model(bag)
        train_loss = criterion(train_logits.view(batch_size, -1), label)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        total_loss += train_loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf):
    """Evaluate model and return metrics"""
    model.eval()
    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter="  ")
    
    for data in data_loader:
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        slide_preds = model(image_patches)
        loss = criterion(slide_preds, labels)
        pred = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(pred, labels, topk=(1,))[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])
        y_pred.append(pred)
        y_true.append(labels)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    
    AUROC_metric = torchmetrics.AUROC(num_classes=conf.n_class, task='multiclass').to(device)
    AUROC_metric(y_pred, y_true)
    auroc = AUROC_metric.compute().item()
    
    F1_metric = torchmetrics.F1Score(num_classes=conf.n_class, average='macro').to(device)
    F1_metric(y_pred, y_true)
    f1_score = F1_metric.compute().item()
    
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg


def analyze_model_structure(model, conf):
    """Analyze and print model structure information"""
    print("\n" + "="*60)
    print("MODEL STRUCTURE ANALYSIS")
    print("="*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    print("\nModel Components:")
    print("-" * 40)
    
    component_info = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        component_info[name] = {
            'parameters': params,
            'parameter_percentage': params / total_params if total_params > 0 else 0,
            'module_type': str(type(module)).split('.')[-1].strip("'>")
        }
        print(f"{name:20s}: {params:>10,} parameters ({params/total_params:.1%})")
    
    print("\nModel Architecture Summary:")
    print("-" * 40)
    print(f"Input feature dimension: {conf.D_feat}")
    print(f"Number of classes: {conf.n_class}")
    print(f"Backbone: {conf.backbone}")
    print(f"Pretraining: {conf.pretrain}")
    
    print("="*60)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_ratio': trainable_params / total_params if total_params > 0 else 0,
        'components': component_info,
        'architecture': {
            'input_dim': conf.D_feat,
            'num_classes': conf.n_class,
            'backbone': conf.backbone,
            'pretrain': conf.pretrain
        }
    }


def save_training_metrics(results_dir, train_losses, val_metrics_history, best_state, model_analysis, conf):
    """Save training metrics and results to files"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare metrics data
    metrics_data = {
        'config_name': os.path.basename(args.config),
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'configuration': dict(vars(conf)) if hasattr(conf, '__dict__') else conf,
        'model_analysis': model_analysis,
        'training_procedure': {
            'train_epochs': conf.train_epoch,
            'batch_size': conf.B,
            'learning_rate': conf.lr,
            'weight_decay': conf.wd,
            'warmup_epochs': conf.warmup_epoch,
            'min_learning_rate': conf.min_lr,
            'optimizer': 'AdamW',
            'scheduler': 'Linear Warmup + Cosine Annealing',
            'loss_function': 'MultiMarginLoss',
            'dataset': conf.dataset,
            'num_classes': conf.n_class,
            'feature_dimension': conf.D_feat
        },
        'best_epoch_results': best_state,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': [m[3] for m in val_metrics_history],
            'val_accuracies': [m[1] for m in val_metrics_history],
            'val_aurocs': [m[0] for m in val_metrics_history],
            'val_f1_scores': [m[2] for m in val_metrics_history]
        }
    }
    
    # Save as JSON
    with open(os.path.join(results_dir, 'baseline_results.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    # Save as text report
    with open(os.path.join(results_dir, 'baseline_report.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("BASELINE EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL STRUCTURE AND COMPONENTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total parameters: {model_analysis['total_parameters']:,}\n")
        f.write(f"Trainable parameters: {model_analysis['trainable_parameters']:,}\n")
        f.write(f"Trainable ratio: {model_analysis['parameter_ratio']:.2%}\n\n")
        
        f.write("Model Components:\n")
        for comp_name, comp_info in model_analysis['components'].items():
            f.write(f"  {comp_name}: {comp_info['parameters']:,} params ({comp_info['parameter_percentage']:.1%})\n")
        
        f.write("\nARCHITECTURE DETAILS\n")
        f.write("-"*40 + "\n")
        arch = model_analysis['architecture']
        f.write(f"Input feature dimension: {arch['input_dim']}\n")
        f.write(f"Number of classes: {arch['num_classes']}\n")
        f.write(f"Backbone: {arch['backbone']}\n")
        f.write(f"Pretraining: {arch['pretrain']}\n\n")
        
        f.write("TRAINING PROCEDURE\n")
        f.write("-"*40 + "\n")
        for key, value in metrics_data['training_procedure'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nPERFORMANCE RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Best epoch: {best_state['epoch']}\n")
        f.write(f"Validation Accuracy: {best_state['val_acc']:.4f}\n")
        f.write(f"Validation AUC: {best_state['val_auc']:.4f}\n")
        f.write(f"Validation F1 Score: {best_state['val_f1']:.4f}\n")
        f.write(f"Test Accuracy: {best_state['test_acc']:.4f}\n")
        f.write(f"Test AUC: {best_state['test_auc']:.4f}\n")
        f.write(f"Test F1 Score: {best_state['test_f1']:.4f}\n\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\nResults saved to: {results_dir}")
    return metrics_data


def main(args):
    # Create results directory
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = f"baseline_results/{config_name}_{timestamp}"
    
    # Print GPU info
    print("="*60)
    print(f"Baseline Evaluation: {config_name}")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Using CPU")
    
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        print("\nConfiguration:")
        pprint.pprint(c)
        conf = Struct(**c)
    
    # Load data
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf=conf)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    
    # Create and analyze model
    model = ABMIL(conf=conf, D=conf.D_feat)
    model.to(device)
    
    # Analyze model structure
    model_analysis = analyze_model_structure(model, conf)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=conf.lr, 
                                  weight_decay=conf.wd)
    
    sched_warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=conf.warmup_epoch
    )

    sched_cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, conf.train_epoch - conf.warmup_epoch),
        eta_min=1e-10
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[sched_warmup, sched_cosine],
        milestones=[conf.warmup_epoch]
    )

    criterion = nn.MultiMarginLoss(p=1, margin=4.0, weight=None, reduction='mean')
    
    # Training tracking
    best_state = {'epoch': -1, 'val_acc': 0, 'val_auc': 0, 'val_f1': 0, 
                  'test_acc': 0, 'test_auc': 0, 'test_f1': 0}
    train_losses = []
    val_metrics_history = []
    
    print(f"\nStarting training for {conf.train_epoch} epochs...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(conf.train_epoch):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_auc, val_acc, val_f1, val_loss = evaluate(model, criterion, val_loader, device, conf)
        test_auc, test_acc, test_f1, test_loss = evaluate(model, criterion, test_loader, device, conf)
        
        val_metrics_history.append((val_auc, val_acc, val_f1, val_loss))
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{conf.train_epoch} - Time: {epoch_time:.1f}s, Train Loss: {train_loss:.4f}")
        
        # Check if best model
        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            print(f"  New best model at epoch {epoch+1}")
    
    # Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best epoch: {best_state['epoch']}")
    print(f"Validation - Accuracy: {best_state['val_acc']:.4f}, AUC: {best_state['val_auc']:.4f}, F1: {best_state['val_f1']:.4f}")
    print(f"Test - Accuracy: {best_state['test_acc']:.4f}, AUC: {best_state['test_auc']:.4f}, F1: {best_state['test_f1']:.4f}")
    print("="*60)
    
    # Save results
    metrics_data = save_training_metrics(results_dir, train_losses, val_metrics_history, 
                                         best_state, model_analysis, conf)

def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml',
                        help='settings of Tip-Adapter in yaml format')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)