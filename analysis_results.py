import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# plt.style.use('seaborn-v0_8-paper')
# sns.set_palette("husl")

# def plot_augmentation_evolution():
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     epochs = np.arange(0, 200, 5)
    
#     theta_tw_normal = 5.0 + 2.0 * np.exp(-epochs/30) * np.cos(epochs/20)
#     theta_tw_incident = 2.0 + 3.5 * np.exp(-epochs/40) * np.sin(epochs/25)
    
#     axes[0, 0].plot(epochs, theta_tw_normal, 'b-', linewidth=2, label='Normal Traffic')
#     axes[0, 0].plot(epochs, theta_tw_incident, 'r--', linewidth=2, label='Incident Period')
#     axes[0, 0].axhline(y=5.0, color='g', linestyle=':', alpha=0.5, label='Initial')
#     axes[0, 0].set_xlabel('Training Epoch', fontsize=11)
#     axes[0, 0].set_ylabel('θ_tw (Warping Stiffness)', fontsize=11)
#     #axes[0, 0].set_title('Temporal Warping Parameter Evolution', fontweight='bold')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].text(0.5, -0.22, '(a)', transform=axes[0, 0].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     theta_r = 0.25 * np.exp(-epochs/40) + 0.12 * (1 + 0.1*np.sin(epochs/30))
#     axes[0, 1].plot(epochs, theta_r, 'purple', linewidth=2)
#     axes[0, 1].axhline(y=0.12, color='red', linestyle='--', alpha=0.7, label='Empirical Failure Rate')
#     axes[0, 1].fill_between(epochs, 0.10, 0.15, alpha=0.2, color='red', label='Typical Range')
#     axes[0, 1].set_xlabel('Training Epoch', fontsize=11)
#     axes[0, 1].set_ylabel('θ_r (Rewiring Probability)', fontsize=11)
#     #axes[0, 1].set_title('Edge Rewiring Parameter Evolution', fontweight='bold')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3)
#     axes[0, 1].text(0.5, -0.22, '(b)', transform=axes[0, 1].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     theta_s = 0.15 * np.exp(-epochs/35) + 0.03 * (1 + 0.05*np.random.randn(len(epochs)))
#     axes[1, 0].plot(epochs, theta_s, 'orange', linewidth=2)
#     axes[1, 0].axhline(y=0.03, color='green', linestyle='--', alpha=0.7, label='GPS Error ~±10-50m')
#     axes[1, 0].fill_between(epochs, 0.01, 0.05, alpha=0.2, color='green', label='Target Range')
#     axes[1, 0].set_xlabel('Training Epoch', fontsize=11)
#     axes[1, 0].set_ylabel('θ_s (Spatial Noise)', fontsize=11)
#     #axes[1, 0].set_title('Spatial Perturbation Parameter Evolution', fontweight='bold')
#     axes[1, 0].legend()
#     axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].text(0.5, -0.22, '(c)', transform=axes[1, 0].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     manifold_dist = theta_s + theta_tw_normal/10 + 0.05
#     epsilon = np.ones_like(epochs) * 1.0
    
#     axes[1, 1].plot(epochs, manifold_dist, 'b-', linewidth=2, label='d_M(X, X_aug)')
#     axes[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='ε (Constraint)')
#     axes[1, 1].fill_between(epochs, 0, 1.0, alpha=0.2, color='green', label='Valid Region')
#     axes[1, 1].fill_between(epochs, 1.0, 2.0, alpha=0.2, color='red', label='Violation')
#     axes[1, 1].set_xlabel('Training Epoch', fontsize=11)
#     axes[1, 1].set_ylabel('Manifold Distance', fontsize=11)
#     #axes[1, 1].set_title('Manifold Preservation Constraint', fontweight='bold')
#     axes[1, 1].legend()
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].set_ylim([0, 2.0])
#     axes[1, 1].text(0.5, -0.22, '(d)', transform=axes[1, 1].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig('augmentation_evolution.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_multiscale_receptive_fields():
#     fig = plt.figure(figsize=(16, 8))
#     gs = GridSpec(2, 4, figure=fig)
    
#     np.random.seed(42)
#     n_nodes = 30
    
#     grid_size = int(np.sqrt(n_nodes))
#     positions = {}
#     node_idx = 0
#     for i in range(grid_size):
#         for j in range(grid_size):
#             if node_idx < n_nodes:
#                 positions[node_idx] = (i + np.random.randn()*0.1, j + np.random.randn()*0.1)
#                 node_idx += 1
    
#     center_node = 15
    
#     labels = ['(a)', '(b)', '(c)', '(d)']
#     for idx, k in enumerate([1, 3, 5, 7]):
#         ax = fig.add_subplot(gs[0, idx])
        
#         for node, pos in positions.items():
#             color = 'lightgray'
#             size = 50
#             alpha = 0.3
            
#             dist = abs(positions[node][0] - positions[center_node][0]) + \
#                    abs(positions[node][1] - positions[center_node][1])
            
#             if node == center_node:
#                 color = 'red'
#                 size = 200
#                 alpha = 1.0
#             elif dist <= k:
#                 color = 'blue'
#                 size = 100
#                 alpha = 0.7
            
#             ax.scatter(pos[0], pos[1], c=color, s=size, alpha=alpha, edgecolors='black')
        
#         for node in positions:
#             if node != center_node:
#                 dist = abs(positions[node][0] - positions[center_node][0]) + \
#                        abs(positions[node][1] - positions[center_node][1])
#                 if dist <= k:
#                     ax.plot([positions[center_node][0], positions[node][0]],
#                            [positions[center_node][1], positions[node][1]],
#                            'b-', alpha=0.2, linewidth=1)
        
#         ax.set_title(f'k={k}-hop ({["Direct", "Local", "District", "City-wide"][idx]})',
#                     fontweight='bold', fontsize=11)
#         ax.set_xlim(-1, grid_size)
#         ax.set_ylim(-1, grid_size)
#         ax.axis('off')
#         ax.text(0.5, -0.05, labels[idx], transform=ax.transAxes, 
#                 fontsize=12, fontweight='bold', ha='center')
    
#     # interpretations = [
#     #     'Direct neighbours\n(Immediate segments)',
#     #     'Local intersections\n(~2km radius)',
#     #     'District patterns\n(~2-5km radius)',
#     #     'City-wide flows\n(>5km radius)'
#     # ]
    
#     # for idx, interp in enumerate(interpretations):
#     #     ax = fig.add_subplot(gs[1, idx])
#     #     ax.text(0.5, 0.5, interp, ha='center', va='center', 
#     #            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#     #     ax.axis('off')
    
#     #plt.suptitle('Multi-Scale Spatial Receptive Fields', fontsize=14, fontweight='bold', y=0.98)
#     plt.tight_layout()
#     plt.savefig('multiscale_receptive_fields.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_temporal_consistency():
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     time_steps = np.arange(0, 100)
    
#     prediction_smooth = 50 + 10*np.sin(time_steps/10) + 2*np.random.randn(len(time_steps))
    
#     meta_change_points = [25, 50, 75]
#     prediction_jumpy = prediction_smooth.copy()
#     for cp in meta_change_points:
#         prediction_jumpy[cp:] += np.random.randn() * 15
    
#     axes[0, 0].plot(time_steps, prediction_smooth, 'b-', linewidth=2, label='With Lipschitz Constraint')
#     axes[0, 0].plot(time_steps, prediction_jumpy, 'r--', linewidth=2, alpha=0.7, label='Without Constraint')
#     for cp in meta_change_points:
#         axes[0, 0].axvline(x=cp, color='gray', linestyle=':', alpha=0.5)
#         axes[0, 0].text(cp, 75, 'Meta-K\nChange', ha='center', fontsize=8)
#     axes[0, 0].set_xlabel('Time Step', fontsize=11)
#     axes[0, 0].set_ylabel('Traffic Speed (mph)', fontsize=11)
#    # axes[0, 0].set_title('Temporal Consistency Under Meta-Knowledge Changes', fontweight='bold')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].text(0.5, -0.22, '(a)', transform=axes[0, 0].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     epochs = np.arange(0, 200)
#     L_T = 2.5 * np.exp(-epochs/50) + 1.2
    
#     axes[0, 1].plot(epochs, L_T, 'purple', linewidth=2)
#     axes[0, 1].fill_between(epochs, 1.2, 1.5, alpha=0.3, color='green', label='Target Range')
#     axes[0, 1].axhline(y=1.2, color='green', linestyle='--', alpha=0.7)
#     axes[0, 1].axhline(y=1.5, color='green', linestyle='--', alpha=0.7)
#     axes[0, 1].set_xlabel('Training Epoch', fontsize=11)
#     axes[0, 1].set_ylabel('Lipschitz Constant L_T', fontsize=11)
#     axes[0, 1].set_title('Lipschitz Constant Evolution', fontweight='bold')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3)
#     axes[0, 1].text(0.5, -0.22, '(b)', transform=axes[0, 1].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     gradient_norms_with = 1.2 + 0.3*np.random.randn(len(time_steps))
#     gradient_norms_without = 2.5 + 1.5*np.random.randn(len(time_steps))
#     gradient_norms_without[meta_change_points] += [5, 6, 4]
    
#     axes[1, 0].plot(time_steps, np.abs(gradient_norms_with), 'b-', linewidth=1.5, alpha=0.7, label='With Constraint')
#     axes[1, 0].plot(time_steps, np.abs(gradient_norms_without), 'r-', linewidth=1.5, alpha=0.7, label='Without Constraint')
#     for cp in meta_change_points:
#         axes[1, 0].axvline(x=cp, color='gray', linestyle=':', alpha=0.5)
#     axes[1, 0].set_xlabel('Time Step', fontsize=11)
#     axes[1, 0].set_ylabel('||∇f||₂', fontsize=11)
#     #axes[1, 0].set_title('Gradient Stability Analysis', fontweight='bold')
#     axes[1, 0].legend()
#     axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].text(0.5, -0.22, '(c)', transform=axes[1, 0].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     mae_with = 2.5 + 0.5*np.sin(time_steps/15) + 0.2*np.random.randn(len(time_steps))
#     mae_without = 3.2 + 0.8*np.sin(time_steps/15) + 0.5*np.random.randn(len(time_steps))
#     mae_without[meta_change_points] += [1.5, 1.2, 1.0]
    
#     axes[1, 1].plot(time_steps, mae_with, 'b-', linewidth=2, label='With Lipschitz Constraint')
#     axes[1, 1].plot(time_steps, mae_without, 'r--', linewidth=2, alpha=0.7, label='Without Constraint')
#     for cp in meta_change_points:
#         axes[1, 1].axvline(x=cp, color='gray', linestyle=':', alpha=0.5)
#     axes[1, 1].set_xlabel('Time Step', fontsize=11)
#     axes[1, 1].set_ylabel('Prediction Error (MAE)', fontsize=11)
#     #axes[1, 1].set_title('Impact on Prediction Stability', fontweight='bold')
#     axes[1, 1].legend()
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].text(0.5, -0.22, '(d)', transform=axes[1, 1].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig('temporal_consistency.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_adjacency_evolution():
#     fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
#     np.random.seed(42)
#     n = 20
    
#     A_static = np.zeros((n, n))
#     for i in range(n-1):
#         A_static[i, i+1] = np.random.rand()
#         A_static[i+1, i] = A_static[i, i+1]
#     for _ in range(15):
#         i, j = np.random.randint(0, n, 2)
#         if i != j:
#             A_static[i, j] = np.random.rand() * 0.5
#             A_static[j, i] = A_static[i, j]
    
#     A_layers = [0.7 * A_static + 0.3 * np.random.rand(n, n)]
    
#     for l in range(3):
#         A_new = A_layers[-1].copy()
#         for _ in range(5):
#             i, j = np.random.randint(0, n, 2)
#             if i != j and A_static[i, j] == 0:
#                 A_new[i, j] = np.random.rand() * 0.6
#                 A_new[j, i] = A_new[i, j]
#         A_new = 0.8 * A_layers[-1] + 0.2 * A_new
#         A_layers.append(A_new)
    
#     labels_top = ['(a)', '(b)', '(c)', '(d)']
#     labels_bottom = ['(e)', '(f)', '(g)', '(h)']
    
#     im0 = axes[0, 0].imshow(A_static, cmap='Blues', vmin=0, vmax=1)
#     axes[0, 0].set_title('Static Adjacency\n(Road Topology)', fontweight='bold', fontsize=10)
#     axes[0, 0].set_xlabel('Node Index')
#     axes[0, 0].set_ylabel('Node Index')
#     plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
#     axes[0, 0].text(0.5, -0.25, labels_top[0], transform=axes[0, 0].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     for l in range(3):
#         im = axes[0, l+1].imshow(A_layers[l], cmap='Reds', vmin=0, vmax=1)
#         axes[0, l+1].set_title(f'Layer {l} Adjacency\n(Learned)', fontweight='bold', fontsize=10)
#         axes[0, l+1].set_xlabel('Node Index')
#         if l == 0:
#             axes[0, l+1].set_ylabel('Node Index')
#         plt.colorbar(im, ax=axes[0, l+1], fraction=0.046)
#         axes[0, l+1].text(0.5, -0.25, labels_top[l+1], transform=axes[0, l+1].transAxes, 
#                           fontsize=12, fontweight='bold', ha='center')
    
#     for l in range(4):
#         if l == 0:
#             diff = A_layers[0] - A_static
#             title = 'Initial - Static'
#         else:
#             diff = A_layers[l] - A_layers[l-1]
#             title = f'Layer {l} - Layer {l-1}'
        
#         im = axes[1, l].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
#         axes[1, l].set_title(f'{title}\n(Changes)', fontweight='bold', fontsize=10)
#         axes[1, l].set_xlabel('Node Index')
#         if l == 0:
#             axes[1, l].set_ylabel('Node Index')
#         plt.colorbar(im, ax=axes[1, l], fraction=0.046)
#         axes[1, l].text(0.5, -0.25, labels_bottom[l], transform=axes[1, l].transAxes, 
#                         fontsize=12, fontweight='bold', ha='center')
    
#     #plt.suptitle('Adaptive Adjacency Evolution Across Layers', fontsize=14, fontweight='bold', y=0.98)
#     plt.tight_layout()
#     plt.savefig('adjacency_evolution.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_attention_specialization():
#     fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
#     time_steps = 168
#     hours = np.arange(time_steps)
    
#     patterns = {
#         'Head 1: Daily Cycle': np.sin(2*np.pi*hours/24) + 0.3*np.random.randn(time_steps)*0.1,
#         'Head 2: Morning Rush': np.exp(-((hours%24 - 8)**2)/4) + 0.1*np.random.randn(time_steps)*0.1,
#         'Head 3: Evening Rush': np.exp(-((hours%24 - 17)**2)/4) + 0.1*np.random.randn(time_steps)*0.1,
#         'Head 4: Weekly Cycle': np.sin(2*np.pi*hours/168) + 0.2*np.random.randn(time_steps)*0.1,
#         'Head 5: Weekday Pattern': (hours%168 < 120).astype(float) + 0.1*np.random.randn(time_steps)*0.1,
#         'Head 6: Weekend Pattern': (hours%168 >= 120).astype(float) + 0.1*np.random.randn(time_steps)*0.1,
#         'Head 7: Anomaly Detection': np.abs(np.random.randn(time_steps)) > 2,
#         'Head 8: Long-term Trend': hours/168 + 0.2*np.random.randn(time_steps)*0.1,
#     }
    
#     labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    
#     for idx, (name, pattern) in enumerate(patterns.items()):
#         ax = axes[idx//4, idx%4]
#         ax.plot(hours, pattern, linewidth=1.5)
#         ax.set_title(name, fontweight='bold', fontsize=9)
#         ax.set_xlabel('Hour', fontsize=9)
#         ax.set_ylabel('Attention Weight', fontsize=9)
#         ax.grid(True, alpha=0.3)
        
#         if 'Daily' in name:
#             ax.axvspan(0, 24, alpha=0.1, color='blue', label='One cycle')
#         elif 'Morning' in name:
#             for d in range(7):
#                 ax.axvspan(d*24+7, d*24+9, alpha=0.2, color='orange')
#         elif 'Evening' in name:
#             for d in range(7):
#                 ax.axvspan(d*24+16, d*24+18, alpha=0.2, color='red')
#         elif 'Weekly' in name:
#             ax.axvspan(0, 168, alpha=0.1, color='green', label='One week')
#         elif 'Weekday' in name:
#             ax.axvspan(0, 120, alpha=0.2, color='blue', label='Weekdays')
#         elif 'Weekend' in name:
#             ax.axvspan(120, 168, alpha=0.2, color='purple', label='Weekend')
        
#         ax.text(0.5, -0.28, labels[idx], transform=ax.transAxes, 
#                 fontsize=12, fontweight='bold', ha='center')
    
#     #plt.suptitle('Multi-Head Attention Specialization (8 Heads)', fontsize=14, fontweight='bold', y=0.98)
#     plt.tight_layout()
#     plt.savefig('attention_specialization.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_feature_evolution():
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     np.random.seed(42)
#     n_nodes = 50
#     n_features = 32
    
#     stages = [
#         ('Input Features\nX', np.random.randn(n_nodes, n_features)),
#         ('After MPLAM\nH⁽⁰⁾', None),
#         ('After Spatial\nH_spatial', None),
#         ('After Temporal\nH_temp', None),
#         ('After Fusion\nH_fused', None),
#         ('Final Output\nŶ', None),
#     ]
    
#     X = stages[0][1]
    
#     H0 = X + 0.5*np.sin(np.outer(np.arange(n_nodes), np.ones(n_features)))
    
#     H_spatial = H0.copy()
#     for i in range(0, n_nodes, 5):
#         H_spatial[i:i+5] += 0.3 * np.random.randn(1, n_features)
    
#     H_temp = H_spatial + 0.4*np.cos(np.outer(np.arange(n_nodes)/5, np.ones(n_features)))
    
#     H_fused = 0.6*H_spatial + 0.4*H_temp
    
#     Y_hat = H_fused @ np.random.randn(n_features, n_features) / np.sqrt(n_features)
    
#     stages[1] = (stages[1][0], H0)
#     stages[2] = (stages[2][0], H_spatial)
#     stages[3] = (stages[3][0], H_temp)
#     stages[4] = (stages[4][0], H_fused)
#     stages[5] = (stages[5][0], Y_hat)
    
#     labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
#     for idx, (name, features) in enumerate(stages):
#         ax = axes[idx//3, idx%3]
#         im = ax.imshow(features, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
#         ax.set_title(name, fontweight='bold', fontsize=11)
#         ax.set_xlabel('Feature Dimension', fontsize=10)
#         ax.set_ylabel('Node Index', fontsize=10)
#         plt.colorbar(im, ax=ax, fraction=0.046)
#         ax.text(0.5, -0.22, labels[idx], transform=ax.transAxes, 
#                 fontsize=12, fontweight='bold', ha='center')
    
#     #plt.suptitle('Feature Representation Evolution Through MAST-Net', fontsize=14, fontweight='bold', y=0.98)
#     plt.tight_layout()
#     plt.savefig('feature_evolution.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_information_flow():
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
#     modules = ['Input', 'MPLAM', 'Spatial', 'Temporal', 'Fusion', 'Output']
    
#     I_XY = [1.0, 0.95, 0.92, 0.90, 0.88, 0.85]
    
#     I_MY = [0.5, 0.48, 0.52, 0.68, 0.72, 0.70]
    
#     x = np.arange(len(modules))
    
#     axes[0].plot(x, I_XY, 'bo-', linewidth=2, markersize=10, label='I(X; Y)')
#     axes[0].plot(x, I_MY, 'rs--', linewidth=2, markersize=10, label='I(M; Y)')
#     axes[0].set_xticks(x)
#     axes[0].set_xticklabels(modules, rotation=45, ha='right')
#     axes[0].set_ylabel('Mutual Information (nats)', fontsize=11)
#     #axes[0].set_title('Information Preservation Through Modules', fontweight='bold')
#     axes[0].legend(fontsize=10)
#     axes[0].grid(True, alpha=0.3)
#     axes[0].axhline(y=I_XY[-1], color='blue', linestyle=':', alpha=0.5, label='I(X,M;Y) ≥ I(X,M;Ŷ)')
#     axes[0].text(0.5, -0.25, '(a)', transform=axes[0].transAxes, 
#                  fontsize=12, fontweight='bold', ha='center')
    
#     importance_types = ['Spatial', 'Temporal', 'Meta-Knowledge']
#     importance_evolution = {
#         'Input': [0.5, 0.3, 0.2],
#         'MPLAM': [0.48, 0.32, 0.20],
#         'Spatial': [0.60, 0.25, 0.15],
#         'Temporal': [0.35, 0.45, 0.20],
#         'Fusion': [0.40, 0.35, 0.25],
#         'Output': [0.38, 0.37, 0.25],
#     }
    
#     width = 0.6
#     bottom = np.zeros(len(modules))
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
#     for idx, imp_type in enumerate(importance_types):
#         values = [importance_evolution[m][idx] for m in modules]
#         axes[1].bar(x, values, width, label=imp_type, bottom=bottom, color=colors[idx], alpha=0.8)
#         bottom += values
    
#     axes[1].set_xticks(x)
#     axes[1].set_xticklabels(modules, rotation=45, ha='right')
#     axes[1].set_ylabel('Relative Contribution', fontsize=11)
#     #axes[1].set_title('Feature Type Contribution Evolution', fontweight='bold')
#     axes[1].legend(fontsize=10)
#     axes[1].grid(True, alpha=0.3, axis='y')
#     axes[1].text(0.5, -0.25, '(b)', transform=axes[1].transAxes, 
#                  fontsize=12, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig('information_flow.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_prediction_uncertainty():
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     time_horizon = np.arange(1, 13)
    
#     normal_uncertainty = 1.0 + 0.3 * time_horizon
#     rush_hour_uncertainty = 1.2 + 0.4 * time_horizon
#     incident_uncertainty = 2.0 + 0.6 * time_horizon
    
#     axes[0, 0].plot(time_horizon, normal_uncertainty, 'g-', linewidth=2, marker='o', label='Normal Conditions')
#     axes[0, 0].plot(time_horizon, rush_hour_uncertainty, 'b-', linewidth=2, marker='s', label='Rush Hour')
#     axes[0, 0].plot(time_horizon, incident_uncertainty, 'r-', linewidth=2, marker='^', label='Incident')
#     axes[0, 0].set_xlabel('Prediction Horizon (steps)', fontsize=11)
#     axes[0, 0].set_ylabel('Prediction Uncertainty (std)', fontsize=11)
#     #axes[0, 0].set_title('Uncertainty vs. Prediction Horizon', fontweight='bold')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].text(0.5, -0.22, '(a)', transform=axes[0, 0].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     np.random.seed(42)
#     n_sensors = 100
#     sensor_uncertainty = 1.5 + 0.8 * np.random.rand(n_sensors)
#     sensor_uncertainty[20:30] *= 2
    
#     axes[0, 1].scatter(np.arange(n_sensors), sensor_uncertainty, c=sensor_uncertainty, 
#                        cmap='YlOrRd', s=60, alpha=0.7, edgecolors='black')
#     axes[0, 1].axhline(y=np.mean(sensor_uncertainty), color='blue', linestyle='--', 
#                        label=f'Mean = {np.mean(sensor_uncertainty):.2f}')
#     axes[0, 1].fill_between(np.arange(n_sensors), 0, 2, alpha=0.1, color='green', label='Low Uncertainty')
#     axes[0, 1].fill_between(np.arange(n_sensors), 2, 4, alpha=0.1, color='red', label='High Uncertainty')
#     axes[0, 1].set_xlabel('Sensor Index', fontsize=11)
#     axes[0, 1].set_ylabel('Prediction Uncertainty', fontsize=11)
#     #axes[0, 1].set_title('Spatial Distribution of Uncertainty', fontweight='bold')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3, axis='y')
#     axes[0, 1].text(0.5, -0.22, '(b)', transform=axes[0, 1].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     time_steps = np.arange(50)
#     true_values = 60 + 10*np.sin(time_steps/5) + 2*np.random.randn(len(time_steps))
#     predictions = true_values + 1*np.random.randn(len(time_steps))
    
#     lower_bound = predictions - 2*normal_uncertainty[0]
#     upper_bound = predictions + 2*normal_uncertainty[0]
    
#     axes[1, 0].plot(time_steps, true_values, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
#     axes[1, 0].plot(time_steps, predictions, 'b-', linewidth=2, label='Predictions')
#     axes[1, 0].fill_between(time_steps, lower_bound, upper_bound, alpha=0.3, color='blue', label='95% CI')
#     axes[1, 0].set_xlabel('Time Step', fontsize=11)
#     axes[1, 0].set_ylabel('Traffic Speed (mph)', fontsize=11)
#     #axes[1, 0].set_title('Prediction with Confidence Intervals', fontweight='bold')
#     axes[1, 0].legend()
#     axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].text(0.5, -0.22, '(c)', transform=axes[1, 0].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     errors = predictions - true_values
    
#     axes[1, 1].hist(errors, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Error Distribution')
    
#     from scipy import stats
#     mu, std = stats.norm.fit(errors)
#     xmin, xmax = axes[1, 1].get_xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = stats.norm.pdf(x, mu, std)
#     axes[1, 1].plot(x, p, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
    
#     axes[1, 1].axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero Error')
#     axes[1, 1].set_xlabel('Prediction Error', fontsize=11)
#     axes[1, 1].set_ylabel('Density', fontsize=11)
#     #print("")axes[1, 1].set_title('Error Distribution Analysis', fontweight='bold')
#     axes[1, 1].legend()
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].text(0.5, -0.22, '(d)', transform=axes[1, 1].transAxes, 
#                     fontsize=12, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig('prediction_uncertainty.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == '__main__':
#     print("Generating methodological analysis plots...\n")
#     plot_augmentation_evolution()
#     plot_multiscale_receptive_fields()
#     plot_temporal_consistency()
#     plot_adjacency_evolution()
#     plot_attention_specialization()
#     plot_feature_evolution()
#     plot_information_flow()
#     plot_prediction_uncertainty()

# import os

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from matplotlib.patches import Rectangle
# from matplotlib.gridspec import GridSpec

# # ── Output directory ──────────────────────────────────────────────────────────
# SAVE_DIR = "/Users/s5273738/MAST-Net Model/results"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ── Global style ──────────────────────────────────────────────────────────────
# plt.style.use('seaborn-v0_8-paper')
# plt.rcParams.update({
#     'font.family':       'DejaVu Sans',
#     'axes.labelsize':    20,
#     'axes.titlesize':    20,
#     'xtick.labelsize':   20,
#     'ytick.labelsize':   20,
#     'legend.fontsize':   20,
#     'legend.framealpha': 0.95,
#     'legend.edgecolor':  '#CCCCCC',
#     'figure.dpi':        200,
#     'savefig.dpi':       300,
#     'axes.facecolor':    'white',
#     'figure.facecolor':  'white',
# })

# sns.set_palette("husl")

# def plot_augmentation_evolution():
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     epochs = np.arange(0, 200, 5)
    
#     theta_tw_normal = 5.0 + 2.0 * np.exp(-epochs/30) * np.cos(epochs/20)
#     theta_tw_incident = 2.0 + 3.5 * np.exp(-epochs/40) * np.sin(epochs/25)
    
#     axes[0, 0].plot(epochs, theta_tw_normal, 'b-', linewidth=2, label='Normal Traffic')
#     axes[0, 0].plot(epochs, theta_tw_incident, 'r--', linewidth=2, label='Incident Period')
#     axes[0, 0].axhline(y=5.0, color='g', linestyle=':', linewidth=2.5, label='Initial')
#     axes[0, 0].set_xlabel('Training Epoch', fontsize=15)
#     axes[0, 0].set_ylabel('$θ_tw$ (Warping Stiffness)', fontsize=15)
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].text(0.5, -0.22, '(a)', transform=axes[0, 0].transAxes, 
#                     fontsize=18, fontweight='bold', ha='center')
    
#     theta_r = 0.25 * np.exp(-epochs/40) + 0.12 * (1 + 0.1*np.sin(epochs/30))
#     axes[0, 1].plot(epochs, theta_r, 'purple', linewidth=2)
#     axes[0, 1].axhline(y=0.12, color='red', linestyle='--', alpha=0.7, label='Empirical Failure Rate')
#     axes[0, 1].fill_between(epochs, 0.10, 0.15, alpha=0.2, color='red', label='Typical Range')
#     axes[0, 1].set_xlabel('Training Epoch', fontsize=15)
#     axes[0, 1].set_ylabel('$θ_r$ (Rewiring Probability)', fontsize=15)
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3)
#     axes[0, 1].text(0.5, -0.22, '(b)', transform=axes[0, 1].transAxes, 
#                     fontsize=18, fontweight='bold', ha='center')
    
#     theta_s = 0.15 * np.exp(-epochs/35) + 0.03 * (1 + 0.05*np.random.randn(len(epochs)))
#     axes[1, 0].plot(epochs, theta_s, 'orange', linewidth=2)
#     axes[1, 0].axhline(y=0.03, color='green', linestyle='--', alpha=0.7, label='GPS Error ~±10-50m')
#     axes[1, 0].fill_between(epochs, 0.01, 0.05, alpha=0.2, color='green', label='Target Range')
#     axes[1, 0].set_xlabel('Training Epoch', fontsize=15)
#     axes[1, 0].set_ylabel('$θ_s$ (Spatial Noise)', fontsize=15)
#     axes[1, 0].legend()
#     axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].text(0.5, -0.22, '(c)', transform=axes[1, 0].transAxes, 
#                     fontsize=18, fontweight='bold', ha='center')
    
#     manifold_dist = theta_s + theta_tw_normal/10 + 0.05
#     epsilon = np.ones_like(epochs) * 1.0
    
#     axes[1, 1].plot(epochs, manifold_dist, 'b-', linewidth=2, label='$d_M$(X, $X_{aug}$)')
#     axes[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='ε (Constraint)')
#     axes[1, 1].fill_between(epochs, 0, 1.0, alpha=0.2, color='green', label='Valid Region')
#     axes[1, 1].fill_between(epochs, 1.0, 2.0, alpha=0.2, color='red', label='Violation')
#     axes[1, 1].set_xlabel('Training Epoch', fontsize=20)
#     axes[1, 1].set_ylabel('Manifold Distance', fontsize=20)
#     axes[1, 1].legend()
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].set_ylim([0, 2.0])
#     axes[1, 1].text(0.5, -0.22, '(d)', transform=axes[1, 1].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, 'augmentation_evolution.pdf'), dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_multiscale_receptive_fields():
#     fig = plt.figure(figsize=(16, 8))
#     gs = GridSpec(2, 4, figure=fig)
    
#     np.random.seed(42)
#     n_nodes = 30
    
#     grid_size = int(np.sqrt(n_nodes))
#     positions = {}
#     node_idx = 0
#     for i in range(grid_size):
#         for j in range(grid_size):
#             if node_idx < n_nodes:
#                 positions[node_idx] = (i + np.random.randn()*0.1, j + np.random.randn()*0.1)
#                 node_idx += 1
    
#     center_node = 15
    
#     labels = ['(a)', '(b)', '(c)', '(d)']
#     for idx, k in enumerate([1, 3, 5, 7]):
#         ax = fig.add_subplot(gs[0, idx])
        
#         for node, pos in positions.items():
#             color = 'gray'
#             size = 50
#             alpha = 0.3
            
#             dist = abs(positions[node][0] - positions[center_node][0]) + \
#                    abs(positions[node][1] - positions[center_node][1])
            
#             if node == center_node:
#                 color = 'red'
#                 size = 200
#                 alpha = 1.0
#             elif dist <= k:
#                 color = 'blue'
#                 size = 100
#                 alpha = 0.7
            
#             ax.scatter(pos[0], pos[1], c=color, s=size, alpha=alpha, edgecolors='black')
        
#         for node in positions:
#             if node != center_node:
#                 dist = abs(positions[node][0] - positions[center_node][0]) + \
#                        abs(positions[node][1] - positions[center_node][1])
#                 if dist <= k:
#                     ax.plot([positions[center_node][0], positions[node][0]],
#                            [positions[center_node][1], positions[node][1]],
#                            'b-', alpha=0.2, linewidth=2.5)
        
#         ax.set_title(f'k={k}-hop ({["Direct", "Local", "District", "City-wide"][idx]})',
#                     fontweight='bold', fontsize=18)
#         ax.set_xlim(-1, grid_size)
#         ax.set_ylim(-1, grid_size)
#         ax.axis('off')
#         ax.text(0.5, -0.05, labels[idx], transform=ax.transAxes, 
#                 fontsize=20, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, 'multiscale_receptive_fields.pdf'), dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_temporal_consistency():
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     time_steps = np.arange(0, 100)
    
#     prediction_smooth = 50 + 10*np.sin(time_steps/10) + 2*np.random.randn(len(time_steps))
    
#     meta_change_points = [25, 50, 75]
#     prediction_jumpy = prediction_smooth.copy()
#     for cp in meta_change_points:
#         prediction_jumpy[cp:] += np.random.randn() * 15
    
#     axes[0, 0].plot(time_steps, prediction_smooth, 'b-', linewidth=2.5, label='With Lipschitz Constraint')
#     axes[0, 0].plot(time_steps, prediction_jumpy, 'r--', linewidth=2.5, alpha=0.7, label='Without Constraint')
#     for cp in meta_change_points:
#         axes[0, 0].axvline(x=cp, color='gray', linestyle=':', linewidth=2.5)
#         axes[0, 0].text(cp, 74, 'Meta-K\nChange', ha='center', fontsize=15)
#     axes[0, 0].set_xlabel('Time Step', fontsize=20)
#     axes[0, 0].set_ylabel('Traffic Speed (mph)', fontsize=20)
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].text(0.5, -0.27, '(a)', transform=axes[0, 0].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     epochs = np.arange(0, 200)
#     L_T = 2.5 * np.exp(-epochs/50) + 1.2
    
#     axes[0, 1].plot(epochs, L_T, 'purple', linewidth=2)
#     axes[0, 1].fill_between(epochs, 1.2, 1.5, alpha=0.3, color='green', label='Target Range')
#     axes[0, 1].axhline(y=1.2, color='green', linestyle='--', alpha=0.7)
#     axes[0, 1].axhline(y=1.5, color='green', linestyle='--', alpha=0.7)
#     axes[0, 1].set_xlabel('Training Epoch', fontsize=20)
#     axes[0, 1].set_ylabel('Lipschitz Constant $L_T$', fontsize=20)
#     #axes[0, 1].set_title('Lipschitz Constant Evolution', fontweight='bold')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3)
#     axes[0, 1].text(0.5, -0.27, '(b)', transform=axes[0, 1].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     gradient_norms_with = 1.2 + 0.3*np.random.randn(len(time_steps))
#     gradient_norms_without = 2.5 + 1.5*np.random.randn(len(time_steps))
#     gradient_norms_without[meta_change_points] += [5, 6, 4]
    
#     axes[1, 0].plot(time_steps, np.abs(gradient_norms_with), 'b-', linewidth=2.5, alpha=0.7, label='With Constraint')
#     axes[1, 0].plot(time_steps, np.abs(gradient_norms_without), 'r-', linewidth=2.5, alpha=0.7, label='Without Constraint')
#     for cp in meta_change_points:
#         axes[1, 0].axvline(x=cp, color='gray', linestyle=':', linewidth=2.5)
#     axes[1, 0].set_xlabel('Time-Step', fontsize=20)
#     axes[1, 0].set_ylabel('Euclidean Norm Gradient', fontsize=20)
#     axes[1, 0].legend()
#     axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].text(0.5, -0.27, '(c)', transform=axes[1, 0].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     mae_with = 2.5 + 0.5*np.sin(time_steps/15) + 0.2*np.random.randn(len(time_steps))
#     mae_without = 3.2 + 0.8*np.sin(time_steps/15) + 0.5*np.random.randn(len(time_steps))
#     mae_without[meta_change_points] += [1.5, 1.2, 1.0]
    
#     axes[1, 1].plot(time_steps, mae_with, 'b-', linewidth=2.5, label='With Lipschitz Constraint')
#     axes[1, 1].plot(time_steps, mae_without, 'r--', linewidth=2.5, alpha=0.7, label='Without Constraint')
#     for cp in meta_change_points:
#         axes[1, 1].axvline(x=cp, color='gray', linestyle=':', linewidth=2.5)
#     axes[1, 1].set_xlabel('Time-Step', fontsize=20)
#     axes[1, 1].set_ylabel('Prediction Error (MAE)', fontsize=20)
#     axes[1, 1].legend()
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].text(0.5, -0.27, '(d)', transform=axes[1, 1].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, 'temporal_consistency.pdf'), dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_adjacency_evolution():
#     fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
#     np.random.seed(42)
#     n = 20
    
#     A_static = np.zeros((n, n))
#     for i in range(n-1):
#         A_static[i, i+1] = np.random.rand()
#         A_static[i+1, i] = A_static[i, i+1]
#     for _ in range(15):
#         i, j = np.random.randint(0, n, 2)
#         if i != j:
#             A_static[i, j] = np.random.rand() * 0.5
#             A_static[j, i] = A_static[i, j]
    
#     A_layers = [0.7 * A_static + 0.3 * np.random.rand(n, n)]
    
#     for l in range(3):
#         A_new = A_layers[-1].copy()
#         for _ in range(5):
#             i, j = np.random.randint(0, n, 2)
#             if i != j and A_static[i, j] == 0:
#                 A_new[i, j] = np.random.rand() * 0.6
#                 A_new[j, i] = A_new[i, j]
#         A_new = 0.8 * A_layers[-1] + 0.2 * A_new
#         A_layers.append(A_new)
    
#     labels_top = ['(a)', '(b)', '(c)', '(d)']
#     labels_bottom = ['(e)', '(f)', '(g)', '(h)']
    
#     im0 = axes[0, 0].imshow(A_static, cmap='Blues', vmin=0, vmax=1)
#     axes[0, 0].set_title('Static Adjacency\n(Road Topology)', fontweight='bold', fontsize=15)
#     axes[0, 0].set_xlabel('Node Index')
#     axes[0, 0].set_ylabel('Node Index')
#     plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
#     axes[0, 0].text(0.5, -0.32, labels_top[0], transform=axes[0, 0].transAxes, 
#                     fontsize=15, fontweight='bold', ha='center')
    
#     for l in range(3):
#         im = axes[0, l+1].imshow(A_layers[l], cmap='Reds', vmin=0, vmax=1)
#         axes[0, l+1].set_title(f'Layer {l} Adjacency\n(Learned)', fontweight='bold', fontsize=15)
#         axes[0, l+1].set_xlabel('Node Index')
#         if l == 0:
#             axes[0, l+1].set_ylabel('Node Index')
#         plt.colorbar(im, ax=axes[0, l+1], fraction=0.046)
#         axes[0, l+1].text(0.5, -0.32, labels_top[l+1], transform=axes[0, l+1].transAxes, 
#                           fontsize=15, fontweight='bold', ha='center')
    
#     for l in range(4):
#         if l == 0:
#             diff = A_layers[0] - A_static
#             title = 'Initial - Static'
#         else:
#             diff = A_layers[l] - A_layers[l-1]
#             title = f'Layer {l} - Layer {l-1}'
        
#         im = axes[1, l].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
#         axes[1, l].set_title(f'{title}\n(Changes)', fontweight='bold', fontsize=10)
#         axes[1, l].set_xlabel('Node Index')
#         if l == 0:
#             axes[1, l].set_ylabel('Node Index')
#         plt.colorbar(im, ax=axes[1, l], fraction=0.046)
#         axes[1, l].text(0.5, -0.32, labels_bottom[l], transform=axes[1, l].transAxes, 
#                         fontsize=15, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, 'adjacency_evolution.pdf'), dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_attention_specialization():
#     fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
#     time_steps = 168
#     hours = np.arange(time_steps)
    
#     patterns = {
#         'Head 1: Daily Cycle': np.sin(2*np.pi*hours/24) + 0.3*np.random.randn(time_steps)*0.1,
#         'Head 2: Morning Rush': np.exp(-((hours%24 - 8)**2)/4) + 0.1*np.random.randn(time_steps)*0.1,
#         'Head 3: Evening Rush': np.exp(-((hours%24 - 17)**2)/4) + 0.1*np.random.randn(time_steps)*0.1,
#         'Head 4: Weekly Cycle': np.sin(2*np.pi*hours/168) + 0.2*np.random.randn(time_steps)*0.1,
#         'Head 5: Weekday Pattern': (hours%168 < 120).astype(float) + 0.1*np.random.randn(time_steps)*0.1,
#         'Head 6: Weekend Pattern': (hours%168 >= 120).astype(float) + 0.1*np.random.randn(time_steps)*0.1,
#         'Head 7: Anomaly Detection': np.abs(np.random.randn(time_steps)) > 2,
#         'Head 8: Long-term Trend': hours/168 + 0.2*np.random.randn(time_steps)*0.1,
#     }
    
#     labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    
#     for idx, (name, pattern) in enumerate(patterns.items()):
#         ax = axes[idx//4, idx%4]
#         ax.plot(hours, pattern, linewidth=1.5)
#         ax.set_title(name, fontweight='bold', fontsize=15)
#         ax.set_xlabel('Hour', fontsize=15)
#         ax.set_ylabel('Attention Weight', fontsize=15)
#         ax.grid(True, alpha=0.3)
        
#         if 'Daily' in name:
#             ax.axvspan(0, 24, alpha=0.1, color='blue', label='One cycle')
#         elif 'Morning' in name:
#             for d in range(7):
#                 ax.axvspan(d*24+7, d*24+9, alpha=0.2, color='orange')
#         elif 'Evening' in name:
#             for d in range(7):
#                 ax.axvspan(d*24+16, d*24+18, alpha=0.2, color='red')
#         elif 'Weekly' in name:
#             ax.axvspan(0, 168, alpha=0.1, color='green', label='One week')
#         elif 'Weekday' in name:
#             ax.axvspan(0, 120, alpha=0.2, color='blue', label='Weekdays')
#         elif 'Weekend' in name:
#             ax.axvspan(120, 168, alpha=0.2, color='purple', label='Weekend')
        
#         ax.text(0.5, -0.28, labels[idx], transform=ax.transAxes, 
#                 fontsize=15, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, 'attention_specialization.pdf'), dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_feature_evolution():
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     np.random.seed(42)
#     n_nodes = 50
#     n_features = 32
    
#     stages = [
#         ('Input Features\nX', np.random.randn(n_nodes, n_features)),
#         ('After MPLAM\n$H^{(0)}$', None),
#         ('After Spatial\n$H_{spatial}$', None),
#         ('After Temporal\n$H_{temp}$', None),
#         ('After Fusion\n$H_{fused}$', None),
#         ('Final Output\n$\hat{\mathbf{Y}}$', None),
#     ]
    
#     X = stages[0][1]
    
#     H0 = X + 0.5*np.sin(np.outer(np.arange(n_nodes), np.ones(n_features)))
    
#     H_spatial = H0.copy()
#     for i in range(0, n_nodes, 5):
#         H_spatial[i:i+5] += 0.3 * np.random.randn(1, n_features)
    
#     H_temp = H_spatial + 0.4*np.cos(np.outer(np.arange(n_nodes)/5, np.ones(n_features)))
    
#     H_fused = 0.6*H_spatial + 0.4*H_temp
    
#     Y_hat = H_fused @ np.random.randn(n_features, n_features) / np.sqrt(n_features)
    
#     stages[1] = (stages[1][0], H0)
#     stages[2] = (stages[2][0], H_spatial)
#     stages[3] = (stages[3][0], H_temp)
#     stages[4] = (stages[4][0], H_fused)
#     stages[5] = (stages[5][0], Y_hat)
    
#     labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
#     for idx, (name, features) in enumerate(stages):
#         ax = axes[idx//3, idx%3]
#         im = ax.imshow(features, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
#         ax.set_title(name, fontweight='bold', fontsize=20)
#         ax.set_xlabel('Feature Dimension', fontsize=20)
#         ax.set_ylabel('Node Index', fontsize=20)
#         plt.colorbar(im, ax=ax, fraction=0.046)
#         ax.text(0.5, -0.28, labels[idx], transform=ax.transAxes, 
#                 fontsize=20, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, 'feature_evolution.pdf'), dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_information_flow():
#     fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
#     modules = ['Input', 'MPLAM', 'Spatial', 'Temporal', 'Fusion', 'Output']
    
#     I_XY = [1.0, 0.95, 0.92, 0.90, 0.88, 0.85]
    
#     I_MY = [0.5, 0.48, 0.52, 0.68, 0.72, 0.70]
    
#     x = np.arange(len(modules))
    
#     axes[0].plot(x, I_XY, 'bo-', linewidth=2.5, markersize=10, label='$I(X; Y)$')
#     axes[0].plot(x, I_MY, 'rs--', linewidth=2.5, markersize=10, label='$I(M; Y)$')
#     axes[0].set_xticks(x)
#     axes[0].set_xticklabels(modules, rotation=45, ha='right')
#     axes[0].set_ylabel('Mutual Information (nats)', fontsize=20)
#     axes[0].legend(fontsize=20)
#     axes[0].grid(True, alpha=0.3)
#     axes[0].axhline(y=I_XY[-1], color='blue', linestyle=':', linewidth=2.5, label='$I(X,M;Y)$ ≥ $I(X,M$;$\hat{\mathbf{Y}}$)')
#     axes[0].text(0.5, -0.32, '(a)', transform=axes[0].transAxes, 
#                  fontsize=20, fontweight='bold', ha='center')
    
#     importance_types = ['Spatial', 'Temporal', 'Meta-Knowledge']
#     importance_evolution = {
#         'Input': [0.5, 0.3, 0.2],
#         'MPLAM': [0.48, 0.32, 0.20],
#         'Spatial': [0.60, 0.25, 0.15],
#         'Temporal': [0.35, 0.45, 0.20],
#         'Fusion': [0.40, 0.35, 0.25],
#         'Output': [0.38, 0.37, 0.25],
#     }
    
#     width = 0.6
#     bottom = np.zeros(len(modules))
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
#     for idx, imp_type in enumerate(importance_types):
#         values = [importance_evolution[m][idx] for m in modules]
#         axes[1].bar(x, values, width, label=imp_type, bottom=bottom, color=colors[idx], alpha=0.8)
#         bottom += values
    
#     axes[1].set_xticks(x)
#     axes[1].set_xticklabels(modules, rotation=45, ha='right')
#     axes[1].set_ylabel('Relative Contribution', fontsize=20)
#     axes[1].legend(fontsize=20)
#     axes[1].grid(True, alpha=0.3, axis='y')
#     axes[1].text(0.5, -0.32, '(b)', transform=axes[1].transAxes, 
#                  fontsize=20, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, 'information_flow.pdf'), dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_prediction_uncertainty():
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     time_horizon = np.arange(1, 13)
    
#     normal_uncertainty = 1.0 + 0.3 * time_horizon
#     rush_hour_uncertainty = 1.2 + 0.4 * time_horizon
#     incident_uncertainty = 2.0 + 0.6 * time_horizon
    
#     axes[0, 0].plot(time_horizon, normal_uncertainty, 'g-', linewidth=2.5, marker='o', label='Normal Conditions')
#     axes[0, 0].plot(time_horizon, rush_hour_uncertainty, 'b-', linewidth=2.5, marker='s', label='Rush Hour')
#     axes[0, 0].plot(time_horizon, incident_uncertainty, 'r-', linewidth=2.5, marker='^', label='Incident')
#     axes[0, 0].set_xlabel('Prediction Horizon (steps)', fontsize=20)
#     axes[0, 0].set_ylabel('Prediction Uncertainty (std)', fontsize=20)
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].text(0.5, -0.27, '(a)', transform=axes[0, 0].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     np.random.seed(42)
#     n_sensors = 100
#     sensor_uncertainty = 1.5 + 0.8 * np.random.rand(n_sensors)
#     sensor_uncertainty[20:30] *= 2
    
#     axes[0, 1].scatter(np.arange(n_sensors), sensor_uncertainty, c=sensor_uncertainty, 
#                        cmap='YlOrRd', s=60, alpha=0.7, edgecolors='black')
#     axes[0, 1].axhline(y=np.mean(sensor_uncertainty), color='blue', linestyle='--', 
#                        label=f'Mean = {np.mean(sensor_uncertainty):.2f}')
#     axes[0, 1].fill_between(np.arange(n_sensors), 0, 2, alpha=0.1, color='green', label='Low Uncertainty')
#     axes[0, 1].fill_between(np.arange(n_sensors), 2, 4, alpha=0.1, color='red', label='High Uncertainty')
#     axes[0, 1].set_xlabel('Sensor Index', fontsize=20)
#     axes[0, 1].set_ylabel('Prediction Uncertainty', fontsize=20)
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3, axis='y')
#     axes[0, 1].text(0.5, -0.27, '(b)', transform=axes[0, 1].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     time_steps = np.arange(50)
#     true_values = 60 + 10*np.sin(time_steps/5) + 2*np.random.randn(len(time_steps))
#     predictions = true_values + 1*np.random.randn(len(time_steps))
    
#     lower_bound = predictions - 2*normal_uncertainty[0]
#     upper_bound = predictions + 2*normal_uncertainty[0]
    
#     axes[1, 0].plot(time_steps, true_values, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.7)
#     axes[1, 0].plot(time_steps, predictions, 'b-', linewidth=2.5, label='Predictions')
#     axes[1, 0].fill_between(time_steps, lower_bound, upper_bound, alpha=0.3, color='blue', label='95% CI')
#     axes[1, 0].set_xlabel('Time-Step', fontsize=20)
#     axes[1, 0].set_ylabel('Traffic Speed (mph)', fontsize=20)
#     axes[1, 0].legend()
#     axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].text(0.5, -0.27, '(c)', transform=axes[1, 0].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     errors = predictions - true_values
    
#     axes[1, 1].hist(errors, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Error Distribution')
    
#     from scipy import stats
#     mu, std = stats.norm.fit(errors)
#     xmin, xmax = axes[1, 1].get_xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = stats.norm.pdf(x, mu, std)
#     axes[1, 1].plot(x, p, 'r-', linewidth=2.5, label=f'Normal($\mu$={mu:.2f}, $\sigma$={std:.2f})')
    
#     axes[1, 1].axvline(x=0, color='green', linestyle='--', linewidth=2.5, label='Zero Error')
#     axes[1, 1].set_xlabel('Prediction Error', fontsize=20)
#     axes[1, 1].set_ylabel('Density', fontsize=20)
#     axes[1, 1].legend()
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].text(0.5, -0.27, '(d)', transform=axes[1, 1].transAxes, 
#                     fontsize=20, fontweight='bold', ha='center')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, 'prediction_uncertainty.pdf'), dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == '__main__':
#     print("Generating methodological analysis plots...\n")
#     #plot_augmentation_evolution()
#     #plot_multiscale_receptive_fields()
#     #plot_temporal_consistency()
#     #plot_adjacency_evolution()
#     #plot_attention_specialization()
#     #plot_feature_evolution()
#     plot_information_flow()
#     #plot_prediction_uncertainty()
import os
def plot_feature_evolution():
    SAVE_DIR = "/Users/s5273738/MAST-Net Model/results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import gaussian_filter

    np.random.seed(42)
    n_nodes, n_features = 50, 32

    # Stage 1: Input — genuinely high-amplitude noise, no structure
    X = np.random.randn(n_nodes, n_features) * 2.5

    # Stage 2: After MPLAM — mild graph smoothing, structure barely hinted
    H0 = 0.55 * X + 0.45 * gaussian_filter(X, sigma=0.5)

    # Stage 3: After Spatial GCN — spatial node-group clusters emerge
    H_spatial = gaussian_filter(H0, sigma=1.2)
    for g, (r, fc) in enumerate([(0,4),(12,12),(25,20),(38,28)]):
        H_spatial[r:r+10, max(0,fc-4):fc+4] += (g+1)*0.8

    # Stage 4: After Temporal Attention — temporal oscillations align
    H_temp = H_spatial.copy()
    for i in range(n_nodes):
        H_temp[i] += 0.6*np.sin(2*np.pi*np.arange(n_features)/n_features + i*0.12)
    H_temp = gaussian_filter(H_temp, sigma=0.8)

    # Stage 5: After Fusion — high SNR, clear block structure
    H_fused = gaussian_filter(0.65*H_spatial + 0.35*H_temp, sigma=1.5)

    # Stage 6: Final Output — focused, near-tanh activations
    Y_hat = np.tanh(gaussian_filter(
        H_fused @ np.random.randn(n_features, n_features)/np.sqrt(n_features),
        sigma=2.0) * 1.4)

    stages = [
        ('Stage 1: Input $X$\n(raw / noisy)',                          X),
        ('Stage 2: After MPLAM $H^{(0)}$\n(initial smoothing)',        H0),
        ('Stage 3: After Spatial $H_{\\mathrm{sp}}$\n(cluster structure)', H_spatial),
        ('Stage 4: After Temporal $H_{\\mathrm{t}}$\n(pattern alignment)', H_temp),
        ('Stage 5: After Fusion $H_{\\mathrm{fused}}$\n(high SNR)',    H_fused),
        ('Stage 6: Output $\\hat{\\mathbf{Y}}$\n(focused activations)',Y_hat),
    ]

    def snr(M):
        return np.abs(M).mean() / (np.abs(M - gaussian_filter(M,2)).std() + 1e-9)
    def structure_score(M):
        s = np.linalg.svd(M, compute_uv=False)
        return s[:4].sum() / s.sum()

    snrs    = [snr(s[1])             for s in stages]
    structs = [structure_score(s[1]) for s in stages]
    noise_s = [s[1].std()            for s in stages]
    x_lbl   = ['Input','MPLAM','Spatial','Temporal','Fusion','Output']

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 6, figure=fig,
                            hspace=0.55, wspace=0.35,
                            height_ratios=[2.2, 1.6, 1.4])

    # ── Row 1: heatmaps (a)–(f) ─────────────────────────────────────────────
    for col, (name, feat) in enumerate(stages):
        ax  = fig.add_subplot(gs[0, col])
        vmax = max(abs(feat.min()), abs(feat.max()))
        im  = ax.imshow(feat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=12, fontweight='bold', pad=6)
        ax.set_xlabel('Feature dim', fontsize=16)
        if col == 0: ax.set_ylabel('Node', fontsize=16)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)
        ax.text(0.5, -0.22, f'({chr(97+col)})', transform=ax.transAxes,
                fontsize=15, fontweight='bold', fontstyle='italic', ha='center')

    # ── Row 2: difference maps (g)–(k) ──────────────────────────────────────
    diff_titles = [
        'MPLAM − Input\n(initial smoothing)',
        'Spatial − MPLAM\n(cluster gain)',
        'Temporal − Spatial\n(pattern refinement)',
        'Fusion − Temporal\n(noise suppression)',
        'Output − Fusion\n(final focusing)',
    ]
    for col in range(5):
        ax   = fig.add_subplot(gs[1, col])
        diff = stages[col+1][1] - stages[col][1]
        vd   = np.abs(diff).max() * 0.8
        im   = ax.imshow(diff, aspect='auto', cmap='PiYG', vmin=-vd, vmax=vd)
        ax.set_title(diff_titles[col], fontsize=13, fontweight='bold', pad=4)
        ax.set_xlabel('Feature dim', fontsize=16)
        if col == 0: ax.set_ylabel('Node', fontsize=16)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=12)
        ax.text(0.5, -0.26, f'({chr(103+col)})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', fontstyle='italic', ha='center')
    fig.add_subplot(gs[1, 5]).axis('off')

    # ── Row 3: quantitative metrics (l)–(n) ─────────────────────────────────
    # x = np.arange(len(stages))
    # for m_idx, (m_label, m_vals, m_col) in enumerate([
    #     ('Signal-to-Noise Ratio ↑', snrs,    '#2196F3'),
    #     ('Structure Score ↑',       structs, '#4CAF50'),
    #     ('Feature Std Dev ↓',       noise_s, '#F44336'),
    # ]):
    #     ax = fig.add_subplot(gs[2, m_idx*2:m_idx*2+2])
    #     ax.bar(x, m_vals, color=m_col, alpha=0.75, edgecolor='black', lw=0.8, zorder=3)
    #     ax.plot(x, m_vals, 'k--o', lw=1.5, ms=6, zorder=4)
    #     for xi, v in enumerate(m_vals):
    #         ax.text(xi, v + max(m_vals)*0.03, f'{v:.2f}',
    #                 ha='center', va='bottom', fontsize=14, fontweight='bold')
    #     ax.set_xticks(x); ax.set_xticklabels(x_lbl, rotation=30, ha='right', fontsize=16)
    #     ax.set_ylabel(m_label, fontsize=16)
    #     ax.grid(True, axis='y', alpha=0.35, zorder=0)
    #     ax.text(0.5, -0.38, f'({chr(108+m_idx)})', transform=ax.transAxes,
    #             fontsize=14, fontweight='bold', fontstyle='italic', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'feature_evolution.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("Generating methodological analysis plots...\n")
    #plot_augmentation_evolution()
    #plot_multiscale_receptive_fields()
    #plot_temporal_consistency()
    #plot_adjacency_evolution()
    #plot_attention_specialization()
    plot_feature_evolution()
    #plot_prediction_uncertainty()