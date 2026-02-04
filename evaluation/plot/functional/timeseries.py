import numpy as np
import matplotlib.pyplot as plt


fontdict = {"font.size": 14}


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
    time_ids = list(range(len(data)))

    positive = data.where(data > 0, drop=False)
    negative = data.where(data < 0, drop=False)

    pos_ids = (data > 0).to_numpy().astype(np.int32).tolist()
    neg_ids = (data < 0).to_numpy().astype(np.int32).tolist()

    pos_ids = [time_ids[i] for i, pidx in enumerate(pos_ids) if pidx == 1]
    neg_ids = [time_ids[i] for i, nidx in enumerate(neg_ids) if nidx == 1]

    ax.fill_between(time_ids, 0, positive, color="orange")
    ax.fill_between(time_ids, 0, negative, color="paleturquoise")

def fill_timeseries(
    ax,
    x,
    std=None,
    fill_type="std",
):
    if fill_type == "std":
        assert std is not None, "Standard deviation must be provided for 'std' fill type."
        ax.fill_between(range(len(x)), x - std, x + std, color="gray", alpha=0.2)
    elif fill_type == "positive_negative":
        _fill_positive_negative(ax, x)
    else:
        pass



def timerseries_to_ax(
    ax,
    x,
    y,
    color="black",
    xticks=None,
    title="",
    linear_trend=None,
    std=None,
    linewidth=2.0,
    fill=None,
    label="",
    marker=None,
):


    ax.plot(x, color="black", linewidth=linewidth, label=label, marker=marker)
    
    fill_timeseries(ax, x, std, fill_type=fill)

    if linear_trend is not None:
        ax.plot(
            linear_trend,
            label=label + " Trend",
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

    # if add_linear_trend:
    #    # Add a linear trend line
    #    z = np.polyfit(np.arange(len(data)), data, 1)
    #    p = np.poly1d(z)
    #    plt.plot(np.arange(len(data)), p(np.arange(len(data))), label='Trend', linestyle='--', color='black')

    mult = get_xlabel_multiplier(len(indices))
    plt.xticks(ticks=indices[::mult], labels=xticklabels[::mult], rotation=45)
    plt.ylabel(variable_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/annual_oscillation_{variable_name}.png")
