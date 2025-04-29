import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, auc

def plot_pareto_front(study):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[t.values[0] for t in study.best_trials],
            y=[t.values[1] for t in study.best_trials],
            z=[t.values[2] for t in study.best_trials],
            mode="markers",
            marker=dict(size=8, color=[t.number for t in study.best_trials], colorscale='Viridis'),
            text=[f"Trial {t.number}" for t in study.best_trials],
            hoverinfo="text+x+y+z"
        )
    )

    fig.update_layout(
        title="Pareto Front (Recall vs F1 vs PR-AUC)",
        scene=dict(
            xaxis_title='Recall',
            yaxis_title='F1 Score',
            zaxis_title='PR-AUC'
        )
    )

    return fig

def log_pr_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR curve (AUC = {pr_auc:.2f})'
        )
    )

    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision'
    )

    return fig