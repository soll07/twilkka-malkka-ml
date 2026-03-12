import plotly.graph_objects as go

_CHART_BG = "rgba(0,0,0,0)"
_GRID_COLOR = "#e5e7eb"
_TICK_COLOR = "#6b7280"


def make_trend_chart(monthly_trend: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_trend["months"],
            y=monthly_trend["이탈률"],
            mode="lines+markers",
            name="예측",
            line=dict(color="#8B5CF6", width=3),
            marker=dict(size=7, color="#8B5CF6"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_trend["months"],
            y=monthly_trend["시청감소율"],
            mode="lines+markers",
            name="실제",
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=7, color="#3B82F6"),
            opacity=0.9,
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(color=_TICK_COLOR, family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(color="#374151", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color=_TICK_COLOR, size=12),
            showline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRID_COLOR,
            zeroline=False,
            tickfont=dict(color=_TICK_COLOR, size=12),
            title=dict(text="비율(%)", font=dict(color=_TICK_COLOR, size=11)),
        ),
    )
    return fig


def make_risk_donut(risk_segments: dict) -> go.Figure:
    colors = ["#E50914", "#F97316", "#FACC15", "#22C55E"]
    total = sum(risk_segments["values"])

    fig = go.Figure(
        data=[
            go.Pie(
                labels=risk_segments["labels"],
                values=risk_segments["values"],
                hole=0.68,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=3)),
                textinfo="none",
                sort=False,
            )
        ]
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br><span style='font-size:13px;color:#6b7280'>총 사용자</span>",
                x=0.5,
                y=0.5,
                font=dict(size=22, color="#111827", family="Inter, sans-serif"),
                showarrow=False,
            )
        ],
    )
    return fig


def make_genre_donut(genres: list[dict]) -> go.Figure:
    labels = [g["label"] for g in genres]
    values = [g["value"] for g in genres]
    colors = ["#E50914", "#8B5CF6", "#3B82F6", "#F59E0B", "#10B981"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=3)),
                textinfo="none",
                sort=False,
            )
        ]
    )

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        showlegend=False,
    )
    return fig
