# import plotly.graph_objects as go

# def plot_function_vs_param(
#     f,
#     x_param,
#     x_values,
#     base_params,
#     y_title
# ):
#     y = []

#     for x in x_values:
#         params = base_params.copy()
#         params[x_param] = x
#         y.append(f(**params))

#     fig = go.Figure(
#         go.Scatter(x=x_values, y=y, mode="lines")
#     )

#     fig.update_layout(
#         xaxis_title=x_param,
#         yaxis_title=y_title,
#         template="plotly_white"
#     )

#     return fig

import plotly.graph_objects as go

def plot_function_vs_param(
    f,
    x_param,
    x_values,
    base_params,
    y_title,
    varying_param=None,
    varying_values=None,
    title=None
):
    fig = go.Figure()

    # Single line if no second parameter
    if varying_param is None or varying_values is None:
        y = []
        for x in x_values:
            params = base_params.copy()
            params[x_param] = x
            y.append(f(**params))
        fig.add_trace(go.Scatter(x=x_values, y=y, mode="lines", name=""))
    else:
        # Multiple lines for each value of the second parameter
        for val in varying_values:
            y = []
            for x in x_values:
                params = base_params.copy()
                params[x_param] = x
                params[varying_param] = val
                y.append(f(**params))
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y,
                    mode="lines",
                    name=f"{varying_param}={val}"
                )
            )

    fig.update_layout(
        title=title or f"{y_title} vs {x_param}",
        xaxis_title=x_param,
        yaxis_title=y_title,
        template="plotly_white"
    )

    return fig
