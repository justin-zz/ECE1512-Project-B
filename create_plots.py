import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from textwrap import wrap
import seaborn as sns

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_results(json_path):
    """Load baseline results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_training_curves(data, output_dir):
    """Plot training and validation curves"""
    history = data['training_history']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Training Progress: {data['config_name']}", fontsize=14, fontweight='bold')
    
    # Plot 1: Training Loss
    axes[0, 0].plot(history['train_losses'], linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    axes[0, 1].plot(history['val_losses'], linewidth=2, color='#A23B72')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy
    axes[0, 2].plot(history['val_accuracies'], linewidth=2, color='#18A999')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Validation Accuracy')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1])
    
    # Plot 4: Validation AUC
    axes[1, 0].plot(history['val_aurocs'], linewidth=2, color='#F18F01')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('Validation AUC')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 5: Validation F1 Score
    axes[1, 1].plot(history['val_f1_scores'], linewidth=2, color='#C73E1D')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    # Plot 6: All metrics together
    axes[1, 2].plot(history['val_accuracies'], label='Accuracy', linewidth=2, color='#18A999')
    axes[1, 2].plot(history['val_aurocs'], label='AUC', linewidth=2, color='#F18F01')
    axes[1, 2].plot(history['val_f1_scores'], label='F1 Score', linewidth=2, color='#C73E1D')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Validation Metrics Comparison')
    axes[1, 2].legend(loc='lower right')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), bbox_inches='tight')
    plt.show()

def plot_model_analysis(data, output_dir):
    """Plot model structure and parameter analysis"""
    model_info = data['model_analysis']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Model Analysis: {data['config_name']}", fontsize=14, fontweight='bold')
    
    # Plot 1: Parameter distribution
    components = model_info['components']
    comp_names = list(components.keys())
    comp_params = [comp['parameters'] for comp in components.values()]
    comp_percentages = [comp['parameter_percentage'] * 100 for comp in components.values()]
    
    bars = axes[0].barh(comp_names, comp_params, color=plt.cm.Set3(np.arange(len(comp_names))))
    axes[0].set_xlabel('Number of Parameters')
    axes[0].set_title('Parameter Distribution by Component')
    axes[0].invert_yaxis()
    
    # Add value labels
    for i, (bar, percentage) in enumerate(zip(bars, comp_percentages)):
        width = bar.get_width()
        axes[0].text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:,}\n({percentage:.1f}%)', 
                    va='center', ha='left', fontsize=8)
    
    # Plot 2: Trainable vs Non-trainable parameters
    trainable = model_info['trainable_parameters']
    total = model_info['total_parameters']
    non_trainable = total - trainable
    
    labels = ['Trainable', 'Non-trainable']
    sizes = [trainable, non_trainable]
    colors = ['#4CAF50', '#F44336']
    
    wedges, texts, autotexts = axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90)
    axes[1].set_title(f'Parameter Trainability\nTotal: {total:,} parameters')
    
    # Make autotexts white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'model_analysis.pdf'), bbox_inches='tight')
    plt.show()

def plot_performance_comparison(all_data, output_dir):
    """Compare performance across different configurations"""
    if len(all_data) < 2:
        print("Need at least 2 configurations for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison Across Configurations', fontsize=16, fontweight='bold')
    
    config_names = [d['config_name'] for d in all_data]
    
    # Extract metrics
    val_accs = [d['best_epoch_results']['val_acc'] for d in all_data]
    val_aucs = [d['best_epoch_results']['val_auc'] for d in all_data]
    val_f1s = [d['best_epoch_results']['val_f1'] for d in all_data]
    test_accs = [d['best_epoch_results']['test_acc'] for d in all_data]
    test_aucs = [d['best_epoch_results']['test_auc'] for d in all_data]
    test_f1s = [d['best_epoch_results']['test_f1'] for d in all_data]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    # Plot 1: Validation Metrics
    axes[0, 0].bar(x - width/2, val_accs, width, label='Accuracy', color='#2E86AB')
    axes[0, 0].bar(x + width/2, val_aucs, width, label='AUC', color='#A23B72')
    axes[0, 0].set_xlabel('Configuration')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Best Validation Metrics')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(config_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1])
    
    # Add value labels
    for i, (acc, auc) in enumerate(zip(val_accs, val_aucs)):
        axes[0, 0].text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        axes[0, 0].text(i + width/2, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Test Metrics
    axes[0, 1].bar(x - width/2, test_accs, width, label='Accuracy', color='#18A999')
    axes[0, 1].bar(x + width/2, test_aucs, width, label='AUC', color='#F18F01')
    axes[0, 1].set_xlabel('Configuration')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Corresponding Test Metrics')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(config_names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1])
    
    # Add value labels
    for i, (acc, auc) in enumerate(zip(test_accs, test_aucs)):
        axes[0, 1].text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        axes[0, 1].text(i + width/2, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: F1 Scores comparison
    axes[1, 0].bar(x - width/2, val_f1s, width, label='Validation', color='#C73E1D')
    axes[1, 0].bar(x + width/2, test_f1s, width, label='Test', color='#6A4C93')
    axes[1, 0].set_xlabel('Configuration')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(config_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1])
    
    # Add value labels
    for i, (val_f1, test_f1) in enumerate(zip(val_f1s, test_f1s)):
        axes[1, 0].text(i - width/2, val_f1 + 0.01, f'{val_f1:.3f}', ha='center', va='bottom', fontsize=8)
        axes[1, 0].text(i + width/2, test_f1 + 0.01, f'{test_f1:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Performance summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    # Create summary table
    table_data = []
    for i, config in enumerate(config_names):
        table_data.append([
            config,
            f"{val_accs[i]:.3f}",
            f"{val_aucs[i]:.3f}",
            f"{val_f1s[i]:.3f}",
            f"{test_accs[i]:.3f}",
            f"{test_aucs[i]:.3f}",
            f"{test_f1s[i]:.3f}"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Config', 'Val Acc', 'Val AUC', 'Val F1', 'Test Acc', 'Test AUC', 'Test F1'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        table[(i, 0)].set_facecolor('#f2f2f2')
    
    axes[1, 1].set_title('Performance Summary Table', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'performance_comparison.pdf'), bbox_inches='tight')
    plt.show()

def create_summary_tables(all_data, output_dir):
    """Create comprehensive summary tables"""
    
    # Table 1: Performance metrics
    perf_data = []
    for data in all_data:
        perf_data.append({
            'Configuration': data['config_name'],
            'Best Epoch': data['best_epoch_results']['epoch'],
            'Val Accuracy': f"{data['best_epoch_results']['val_acc']:.4f}",
            'Val AUC': f"{data['best_epoch_results']['val_auc']:.4f}",
            'Val F1': f"{data['best_epoch_results']['val_f1']:.4f}",
            'Test Accuracy': f"{data['best_epoch_results']['test_acc']:.4f}",
            'Test AUC': f"{data['best_epoch_results']['test_auc']:.4f}",
            'Test F1': f"{data['best_epoch_results']['test_f1']:.4f}",
            'Final Train Loss': f"{data['training_history']['train_losses'][-1]:.4f}",
            'Final Val Loss': f"{data['training_history']['val_losses'][-1]:.4f}"
        })
    
    perf_df = pd.DataFrame(perf_data)
    
    # Table 2: Model architecture
    arch_data = []
    for data in all_data:
        arch = data['model_analysis']['architecture']
        arch_data.append({
            'Configuration': data['config_name'],
            'Input Dimension': arch['input_dim'],
            'Num Classes': arch['num_classes'],
            'Backbone': arch['backbone'],
            'Pretraining': arch['pretrain'],
            'Total Params': f"{data['model_analysis']['total_parameters']:,}",
            'Trainable Params': f"{data['model_analysis']['trainable_parameters']:,}",
            'Trainable Ratio': f"{data['model_analysis']['parameter_ratio']:.2%}"
        })
    
    arch_df = pd.DataFrame(arch_data)
    
    # Table 3: Training hyperparameters
    train_data = []
    for data in all_data:
        train_info = data['training_procedure']
        arch = data['model_analysis']['architecture']
        train_data.append({
            'Configuration': data['config_name'],
            'Dataset': train_info['dataset'],
            'Epochs': train_info['train_epochs'],
            'Batch Size': train_info['batch_size'],
            'Learning Rate': train_info['learning_rate'],
            'Weight Decay': train_info['weight_decay'],
            'Warmup Epochs': train_info['warmup_epochs'],
            'Optimizer': train_info['optimizer'],
            'Scheduler': train_info['scheduler']
        })
    
    train_df = pd.DataFrame(train_data)
    
    # Save tables as CSV
    perf_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    arch_df.to_csv(os.path.join(output_dir, 'architecture_summary.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)
    
    # Create formatted HTML table
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th { background-color: #4CAF50; color: white; padding: 12px; text-align: left; }
            td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .timestamp { color: #666; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>Baseline Evaluation Report</h1>
        <p class="timestamp">Generated on: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        
        <h2>Performance Summary</h2>
        """ + perf_df.to_html(index=False) + """
        
        <h2>Model Architecture</h2>
        """ + arch_df.to_html(index=False) + """
        
        <h2>Training Configuration</h2>
        """ + train_df.to_html(index=False) + """
        
        <h2>Component Explanations</h2>
        <ul>
            <li><strong>Encoder (Feature Extractor):</strong> Extracts features from individual image patches using pre-trained Vision Transformer (ViT-S/16). Converts images to 384-dimensional feature vectors capturing local texture and morphological patterns.</li>
            <li><strong>Aggregator (Attention Mechanism):</strong> Weighs importance of different patches using attention mechanism. Identifies diagnostically relevant regions and produces weighted sum of patch features. Handles variable number of patches per slide.</li>
            <li><strong>Classifier (Prediction Layer):</strong> Takes aggregated features as input and maps to final class predictions (normal vs. tumor). Typically consists of fully connected layers with softmax activation, outputting probability distribution over classes.</li>
        </ul>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'summary_report.html'), 'w') as f:
        f.write(html_content)
    
    # Print tables to console
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(perf_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(arch_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(train_df.to_string(index=False))
    
    return perf_df, arch_df, train_df

def plot_metric_correlations(all_data, output_dir):
    """Plot correlations between different metrics"""
    if len(all_data) < 3:
        print("Need at least 3 configurations for correlation analysis")
        return
    
    # Extract metrics
    metrics = {
        'Val Accuracy': [d['best_epoch_results']['val_acc'] for d in all_data],
        'Val AUC': [d['best_epoch_results']['val_auc'] for d in all_data],
        'Val F1': [d['best_epoch_results']['val_f1'] for d in all_data],
        'Test Accuracy': [d['best_epoch_results']['test_acc'] for d in all_data],
        'Test AUC': [d['best_epoch_results']['test_auc'] for d in all_data],
        'Test F1': [d['best_epoch_results']['test_f1'] for d in all_data],
        'Final Train Loss': [d['training_history']['train_losses'][-1] for d in all_data],
        'Final Val Loss': [d['training_history']['val_losses'][-1] for d in all_data]
    }
    
    df = pd.DataFrame(metrics)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                          fontsize=9, fontweight='bold')
    
    # Set labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
    
    ax.set_title('Metric Correlations Across Configurations', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlations.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metric_correlations.pdf'), bbox_inches='tight')
    plt.show()
    
    return corr_matrix

def main():
    """Main function to analyze baseline results"""
    
    # Find all baseline result JSON files
    result_files = glob.glob('baseline_results/**/baseline_results.json', recursive=True)
    
    if not result_files:
        print("No baseline results found. Run the training script first.")
        return
    
    print(f"Found {len(result_files)} result file(s):")
    for f in result_files:
        print(f"  - {f}")
    
    # Load all data
    all_data = []
    for file_path in result_files:
        data = load_results(file_path)
        all_data.append(data)
    
    # Create analysis directory
    analysis_dir = 'baseline_analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate individual plots for each configuration
    for data in all_data:
        config_name = data['config_name']
        config_dir = os.path.join(analysis_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        print(f"\nAnalyzing {config_name}...")
        plot_training_curves(data, config_dir)
        plot_model_analysis(data, config_dir)
    
    # Generate comparison plots if multiple configurations
    if len(all_data) > 1:
        print("\nGenerating comparison plots...")
        plot_performance_comparison(all_data, analysis_dir)
        plot_metric_correlations(all_data, analysis_dir)
    
    # Create summary tables
    print("\nCreating summary tables...")
    perf_df, arch_df, train_df = create_summary_tables(all_data, analysis_dir)
    
    print(f"\nAnalysis complete! Results saved to: {analysis_dir}")
    print("\nFiles generated:")
    print("  For each configuration:")
    print("    - training_curves.png/pdf: Training progress plots")
    print("    - model_analysis.png/pdf: Model structure analysis")
    print("  For comparison:")
    print("    - performance_comparison.png/pdf: Cross-configuration comparison")
    print("    - metric_correlations.png/pdf: Metric correlation heatmap")
    print("    - performance_summary.csv: Performance metrics table")
    print("    - architecture_summary.csv: Model architecture table")
    print("    - training_summary.csv: Training hyperparameters table")
    print("    - summary_report.html: Comprehensive HTML report")

if __name__ == '__main__':
    main()