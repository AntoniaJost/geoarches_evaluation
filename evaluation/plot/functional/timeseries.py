import numpy as np
import matplotlib.pyplot as plt


fontdict = {"font.size": 14}


get_xticks_from_time = lambda time: [f"{t.year}-{t.month:02d}" for t in time]

def get_xlabel_multiplier(n_xticks):
    if n_xticks in [365, 366]:
        return 30

    if n_xticks <= 12:
        return 1
    elif n_xticks <= 24:
        return 2
    elif n_xticks <= 36:
        return 3
    elif n_xticks <= 48:
        return 4
    elif n_xticks <= 120:
        return 6
    elif n_xticks <= 240:
        return 12
    elif n_xticks <= 1200:
        return 24


def _fill_positive_negative(ax, data):
    # Highlight positive and negative values
    time_ids = np.arange(len(data))

    positive = np.where(data > 0, data, 0)
    negative = np.where(data < 0, data, 0)

    pos_ids = data > 0
    neg_ids = data < 0
    
    pos_ids = time_ids[data > 0] #[time_ids[i] for i, pidx in enumerate(pos_ids) if pidx == 1]
    neg_ids = time_ids[data < 0] #[time_ids[i] for i, nidx in enumerate(neg_ids) if nidx == 1]

    ax.fill_between(time_ids, 0, positive, color="orange")
    ax.fill_between(time_ids, 0, negative, color="paleturquoise")
    ax.hlines(0, time_ids[0], time_ids[-1], color="k", linewidth=0.5, linestyle="-")

def fill_timeseries(
    ax,
    x,
    std=None,
    fill_type="full",
    alpha=0.2
):
    if fill_type == "full":
        assert std is not None, "Standard deviation must be provided for 'full' fill type."
        ax.fill_between(range(len(x)), x - std, x + std, color="gray", alpha=alpha)
    elif fill_type == "sign":
        _fill_positive_negative(ax, x)
    else:
        pass



def timerseries_to_ax(
    ax,
    x,
    y=None,
    color="black",
    linear_trend=None,
    std=None,
    linewidth=2.0,
    fill: None | str = None,
    label="",
    marker=None,
    fill_alpha=0.2,
):
    """

    Args:
        ax (_type_): The matplotlib axis to plot on.
        x (_type_): The x data. If only x is provided, it is assumed to be the y data.
        y (_type_, optional): The y data. Defaults to None.
        color (str, optional): The color of the plot. Defaults to "black".
        xticks (_type_, optional): The x-axis ticks. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "".
        linear_trend (_type_, optional): The linear trend data. Defaults to None.
        std (_type_, optional): The standard deviation data. Defaults to None.
        linewidth (float, optional): The width of the plot line. Defaults to 2.0.
        fill (None | str, optional): The fill type for the plot. Defaults to None.
        Options are "full", "sign". If "full", the area between (y - std) and (y + std) is filled.
        If "sign", positive and negative areas are filled with different colors. Defaults to None.
        label (str, optional): The label for the plot. Defaults to "".
        marker (_type_, optional): The marker style for the plot. Defaults to None.
    """


    if linear_trend is not None: 
        alpha = 0.7
    else:
        alpha = 1.0

    if y is not None:
        ax.plot(x, y, color=color, linewidth=linewidth, label=label, marker=marker, alpha=alpha)
        fill_timeseries(ax, y, std, fill_type=fill, alpha=fill_alpha)

    else:
        ax.plot(x, color="black", linewidth=linewidth, label=label, marker=marker, alpha=alpha)
        fill_timeseries(ax, x, std, fill_type=fill, alpha=fill_alpha)

    if linear_trend is not None:
        if y is not None:
            ax.plot(
                x,
                linear_trend,
                linestyle="--",
                color=color,
                linewidth=linewidth,
            )
        else:
            ax.plot(
                x,
                linear_trend,
                linestyle="--",
                color=color,
                linewidth=linewidth,
            )   



def plot_annual_oscillation(
    time, data, output_path, variable_name, add_linear_trend=True, ref=None, std=None
):
    """
    Plot the annual oscillation of a variable.

    :param data: The data to plot.
    :param variable_name: The name of the variable to plot.
    :return: None
    """
    plt.figure(figsize=(15, 5), dpi=150)
    plt.rcParams.update(fontdict)

    plt.plot(
        data,
        label="Simulated" if std is None else "Ensemble Mean",
        color="k",
        linewidth=2.0,
    )

    if std is not None:
        plt.fill_between(
            range(len(time)), (data - std), (data + std), color="blue", alpha=0.2
        )

    if ref is not None:
        plt.plot(
            ref,
            label=f"ERA5 {variable_name}",
            linestyle="-",
            color="orange",
            linewidth=2.0,
        )

    plt.xticks(rotation=45)
    plt.grid()
    plt.title(f"Annual Oscillation of {variable_name}")
    plt.xlabel("Time")

    # time is a list of tuples, each tuple containing (year, month)
    # Create xticklabels of form 'YYYY-MM'
    xticklabels = [f"{y}-{m:02d}" for (y, m) in time]
    print(xticklabels)
    indices = list(range(len(xticklabels)))

    mult = get_xlabel_multiplier(len(indices))
    plt.xticks(ticks=indices[::mult], labels=xticklabels[::mult], rotation=45)
    plt.ylabel(variable_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/annual_oscillation_{variable_name}.png")
