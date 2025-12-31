import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Any, Callable
import json


def create_interactive_loss_landscape(
    loss_fn: Callable,
    theta_opt: np.ndarray,
    output_path: Path,
    n_points: int = 50,
    direction_indices: tuple = (0, 1)
):
    """
    Create interactive 3D loss landscape visualization.
    
    Args:
        loss_fn: Loss function
        theta_opt: Optimal parameters
        output_path: Path to save HTML file
        n_points: Grid resolution
        direction_indices: Which dimensions to visualize
    """
    i, j = direction_indices
    
    # Create grid
    delta = np.linalg.norm(theta_opt) / 5.0
    if delta < 1e-6:
        delta = 0.1
    
    x = np.linspace(-delta, delta, n_points)
    y = np.linspace(-delta, delta, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate loss
    Z = np.zeros_like(X)
    for idx_x in range(n_points):
        for idx_y in range(n_points):
            theta_probe = theta_opt.copy()
            theta_probe[i] += X[idx_x, idx_y]
            theta_probe[j] += Y[idx_x, idx_y]
            Z[idx_x, idx_y] = loss_fn(theta_probe)
    
    # Create figure
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            name='Loss Surface'
        )
    ])
    
    # Add optimum marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[loss_fn(theta_opt)],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Optimum'
    ))
    
    fig.update_layout(
        title='Interactive Loss Landscape',
        scene=dict(
            xaxis_title=f'Δθ[{i}]',
            yaxis_title=f'Δθ[{j}]',
            zaxis_title='Loss',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=900,
        height=700
    )
    
    fig.write_html(output_path)


def create_interactive_parameter_trajectory(
    theta_history: List[np.ndarray],
    theta_true: np.ndarray,
    loss_history: List[float],
    output_path: Path
):
    """
    Create interactive parameter trajectory visualization.
    
    Args:
        theta_history: Parameter history
        theta_true: Ground truth parameters
        loss_history: Loss history
        output_path: Path to save HTML
    """
    theta_history = np.array(theta_history)
    iterations = np.arange(len(theta_history))
    d = theta_history.shape[1]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Parameter Evolution', 
            'Distance to Ground Truth',
            'Loss Evolution',
            'Trajectory in Parameter Space'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Plot 1: Parameter components
    for i in range(min(d, 8)):
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=theta_history[:, i],
                name=f'θ[{i}]',
                mode='lines',
                line=dict(width=2)
            ),
            row=1, col=1
        )
    
    # Plot 2: Distance to optimum
    distances = [np.linalg.norm(theta - theta_true) for theta in theta_history]
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=distances,
            name='||θ - θ*||₂',
            mode='lines',
            line=dict(width=3, color='blue')
        ),
        row=1, col=2
    )
    
    # Plot 3: Loss evolution
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=loss_history,
            name='Loss',
            mode='lines',
            line=dict(width=3, color='red')
        ),
        row=2, col=1
    )
    
    # Plot 4: 2D trajectory
    if d >= 2:
        fig.add_trace(
            go.Scatter(
                x=theta_history[:, 0],
                y=theta_history[:, 1],
                name='Trajectory',
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        # Add start, end, and true markers
        fig.add_trace(
            go.Scatter(
                x=[theta_history[0, 0]],
                y=[theta_history[0, 1]],
                name='Start',
                mode='markers',
                marker=dict(size=12, color='green', symbol='circle')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[theta_history[-1, 0]],
                y=[theta_history[-1, 1]],
                name='End',
                mode='markers',
                marker=dict(size=12, color='red', symbol='circle')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[theta_true[0]],
                y=[theta_true[1]],
                name='Ground Truth',
                mode='markers',
                marker=dict(size=15, color='black', symbol='star')
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_xaxes(title_text="θ[0]", row=2, col=2)
    
    fig.update_yaxes(title_text="Parameter Value", row=1, col=1)
    fig.update_yaxes(title_text="Distance", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Loss", type="log", row=2, col=1)
    fig.update_yaxes(title_text="θ[1]", row=2, col=2)
    
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        title_text="Parameter Optimization Trajectory"
    )
    
    fig.write_html(output_path)


def create_interactive_noise_heatmap(
    results: List[Dict[str, Any]],
    output_path: Path
):
    """
    Create interactive heatmap with multiple metrics.
    
    Args:
        results: Experiment results
        output_path: Path to save HTML
    """
    # Extract unique noise values
    p_deps = sorted(set(r['p_dep'] for r in results))
    sigmas = sorted(set(r['sigma_phase'] for r in results))
    
    # Create grids for each metric
    metrics = ['acc', 'param_l2', 'ident_proxy']
    metric_labels = {
        'acc': 'Accuracy',
        'param_l2': 'Parameter L2 Error',
        'ident_proxy': 'Identifiability Proxy'
    }
    
    grids = {}
    for metric in metrics:
        grid = np.full((len(sigmas), len(p_deps)), np.nan)
        for r in results:
            i = sigmas.index(r['sigma_phase'])
            j = p_deps.index(r['p_dep'])
            grid[i, j] = r[metric]
        grids[metric] = grid
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[metric_labels[m] for m in metrics],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    for idx, metric in enumerate(metrics, 1):
        fig.add_trace(
            go.Heatmap(
                z=grids[metric],
                x=[f'{p:.2f}' for p in p_deps],
                y=[f'{s:.2f}' for s in sigmas],
                colorscale='Viridis',
                name=metric_labels[metric],
                text=grids[metric],
                texttemplate='%{text:.3f}',
                textfont={"size": 10},
                colorbar=dict(title=metric_labels[metric])
            ),
            row=1, col=idx
        )
    
    fig.update_xaxes(title_text="Depolarizing Probability (p)", row=1, col=1)
    fig.update_xaxes(title_text="Depolarizing Probability (p)", row=1, col=2)
    fig.update_xaxes(title_text="Depolarizing Probability (p)", row=1, col=3)
    
    fig.update_yaxes(title_text="Phase Noise Sigma (σ)", row=1, col=1)
    
    fig.update_layout(
        height=500,
        width=1400,
        title_text="Noise Parameter Sweep Heatmaps"
    )
    
    fig.write_html(output_path)


def create_interactive_metrics_dashboard(
    results: List[Dict[str, Any]],
    output_path: Path
):
    """
    Create comprehensive interactive dashboard.
    
    Args:
        results: Experiment results
        output_path: Path to save HTML
    """
    # Extract data
    p_deps = np.array([r['p_dep'] for r in results])
    sigmas = np.array([r['sigma_phase'] for r in results])
    noise_index = p_deps + sigmas
    accs = np.array([r['acc'] for r in results])
    param_errs = np.array([r['param_l2'] for r in results])
    idents = np.array([r['ident_proxy'] for r in results])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Accuracy vs Noise',
            'Parameter Error vs Noise',
            'Identifiability Collapse',
            'Accuracy vs Identifiability (Key Result)'
        )
    )
    
    # Plot 1: Accuracy vs Noise
    fig.add_trace(
        go.Scatter(
            x=noise_index,
            y=accs,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=idents,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Identifiability', x=0.46, len=0.45)
            ),
            name='Accuracy',
            hovertemplate='<b>Noise:</b> %{x:.3f}<br>' +
                         '<b>Accuracy:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: Parameter Error vs Noise
    fig.add_trace(
        go.Scatter(
            x=noise_index,
            y=param_errs,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=idents,
                colorscale='Viridis',
                showscale=False
            ),
            name='Parameter Error',
            hovertemplate='<b>Noise:</b> %{x:.3f}<br>' +
                         '<b>Param Error:</b> %{y:.3f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Plot 3: Identifiability vs Noise
    fig.add_trace(
        go.Scatter(
            x=noise_index,
            y=idents,
            mode='markers+lines',
            marker=dict(size=10, color='red'),
            name='Identifiability',
            hovertemplate='<b>Noise:</b> %{x:.3f}<br>' +
                         '<b>Identifiability:</b> %{y:.3e}<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot 4: Accuracy vs Identifiability (main result)
    hover_text = [f"p={r['p_dep']:.2f}, σ={r['sigma_phase']:.2f}" 
                  for r in results]
    
    fig.add_trace(
        go.Scatter(
            x=accs,
            y=idents,
            mode='markers+text',
            marker=dict(
                size=12,
                color=noise_index,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title='Noise Level', x=1.0, len=0.45)
            ),
            text=hover_text,
            textposition='top center',
            textfont=dict(size=8),
            name='Results',
            hovertemplate='<b>Accuracy:</b> %{x:.3f}<br>' +
                         '<b>Identifiability:</b> %{y:.3e}<br>' +
                         '<b>%{text}</b><br>' +
                         '<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Noise Index (p + σ)", row=1, col=1)
    fig.update_xaxes(title_text="Noise Index (p + σ)", row=1, col=2)
    fig.update_xaxes(title_text="Noise Index (p + σ)", row=2, col=1)
    fig.update_xaxes(title_text="Accuracy", row=2, col=2)
    
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Parameter L2 Error", row=1, col=2)
    fig.update_yaxes(title_text="Identifiability Proxy", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Identifiability Proxy", type="log", row=2, col=2)
    
    fig.update_layout(
        height=900,
        width=1400,
        showlegend=False,
        title_text="<b>Interactive Results Dashboard</b><br>" +
                   "<sub>High accuracy can coexist with identifiability collapse</sub>"
    )
    
    fig.write_html(output_path)


def create_curvature_interactive_plot(
    hessian_diag: np.ndarray,
    output_path: Path
):
    """
    Create interactive curvature spectrum visualization.
    
    Args:
        hessian_diag: Hessian diagonal elements
        output_path: Path to save HTML
    """
    d = len(hessian_diag)
    indices = np.arange(d)
    sorted_h = np.sort(np.abs(hessian_diag))[::-1]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Hessian Diagonal by Index', 'Sorted Curvature Spectrum')
    )
    
    # Plot 1: Bar chart by index
    colors = ['red' if h < 0 else 'blue' for h in hessian_diag]
    
    fig.add_trace(
        go.Bar(
            x=indices,
            y=np.abs(hessian_diag),
            marker_color=colors,
            name='|H_ii|',
            hovertemplate='<b>Index:</b> %{x}<br>' +
                         '<b>|Hessian|:</b> %{y:.3e}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: Sorted spectrum
    fig.add_trace(
        go.Scatter(
            x=list(range(d)),
            y=sorted_h,
            mode='markers+lines',
            marker=dict(size=8, color='blue'),
            line=dict(width=2),
            name='Sorted |H_ii|',
            hovertemplate='<b>Rank:</b> %{x}<br>' +
                         '<b>|Hessian|:</b> %{y:.3e}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Parameter Index", row=1, col=1)
    fig.update_xaxes(title_text="Sorted Index", row=1, col=2)
    fig.update_yaxes(title_text="|Hessian Diagonal|", type="log", row=1, col=1)
    fig.update_yaxes(title_text="|Hessian Diagonal|", type="log", row=1, col=2)
    
    # Add condition number annotation
    if sorted_h[-1] > 1e-12:
        condition = sorted_h[0] / sorted_h[-1]
        annotation_text = f"Condition Number: {condition:.2e}"
    else:
        annotation_text = "Condition Number: ∞ (near-singular)"
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.75, y=0.95,
        showarrow=False,
        bgcolor="wheat",
        bordercolor="black",
        borderwidth=2
    )
    
    fig.update_layout(
        height=500,
        width=1200,
        title_text="Interactive Curvature Spectrum Analysis",
        showlegend=False
    )
    
    fig.write_html(output_path)
