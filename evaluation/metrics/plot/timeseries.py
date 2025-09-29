import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


fontdict = {'font.size': 14}

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

    ax.fill_between(time_ids, 0, positive, color='orange')
    ax.fill_between(time_ids, 0, negative, color='paleturquoise')


def plot_timeseries(x, xticks=None, title='', xlabel='', ylabel='', output_path='', 
                    add_data_trend=False, ref=None, std=None, 
                    linewidth=2., fill=None, label=None, ref_label=None
                    ):

    fig = plt.figure(figsize=(15, 5), dpi=150)
    ax = fig.add_subplot(111)
    plt.rcParams.update(fontdict)

    
    ax.plot(x, color='black', linewidth=linewidth, label=label)

    if fill == 'all':
        ax.fill_between(range(len(x)), x - std, x + std, color='gray', alpha=0.2)
    elif fill == 'positive_negative':
        _fill_positive_negative(ax, x)

    if add_data_trend:
        # Add a linear trend line
        z = np.polyfit(np.arange(len(x)), x, 1)
        p = np.poly1d(z)
        ax.plot(np.arange(len(x)), 
                p(np.arange(len(x))), 
                label='Linear Trend', 
                linestyle='--', 
                color='black'
                )
        

    if ref is not None:
        ax.plot(ref, color='orange', linewidth=linewidth, label=ref_label)

    if std is not None:
        ax.fill_between(range(len(x)), x - std, x + std, color='gray', alpha=0.2)


    if xticks is not None:
        if isinstance(xticks, tuple) or isinstance(xticks, list):
            ax.set_xticks(ticks=xticks[0], labels=xticks[1], rotation=45, ha='right')
        else:
            mult = get_xlabel_multiplier(len(xticks))
            ax.set_xticks(
                ticks=range(0, len(xticks), mult),
                labels=xticks[::mult], rotation=45, ha='right')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if label is not None:
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)


"""def plot_enso_index(enso_data, time, output_path, enso_type='Nino34', ref=''):
    # function that plots the ENSO 3.4 index (ENSO34) or soi data
    # values larger than 0 are red and values smaller than 0 are blue
    
    time = list(set(time.astype('datetime64[M]').to_numpy()))  # Ensure time is in datetime format
    time.sort()
    time_ids = list(range(0, len(time)))

    plt.figure(figsize=(15, 5), dpi=150)
    plt.rcParams.update(fontdict)

    plt.plot(time_ids, enso_data, label=f"ENSO Index: {enso_type}", color='black', linewidth=0.7)

    # Highlight positive and negative values
    positive = enso_data.where(enso_data > 0, drop=False)
    negative = enso_data.where(enso_data < 0, drop=False)
    pos_ids = (enso_data > 0).to_numpy().astype(np.int32).tolist()
    neg_ids = (enso_data < 0).to_numpy().astype(np.int32).tolist()
    pos_ids = [time_ids[i] for i, pidx in enumerate(pos_ids) if pidx == 1]
    neg_ids = [time_ids[i] for i, nidx in enumerate(neg_ids) if nidx == 1]

    plt.fill_between(time_ids, 0, positive, color='orange')
    plt.fill_between(time_ids, 0, negative, color='paleturquoise')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)

    years = list(set([t.astype('datetime64[Y]') for t in time]))
    years.sort()


    mult = get_xlabel_multiplier(len(years))
    plt.xticks(ticks=list(range(0, len(time), mult * 12)), labels=years[::mult], rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('°C' if enso_type == 'Nino34' else 'SOI')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/{ref}{enso_type}_plot.png')"""


def plot_annual_oscillation(time, data, output_path, variable_name, add_linear_trend=True, ref=None, std=None):
    """
    Plot the annual oscillation of a variable.
    
    :param data: The data to plot.
    :param variable_name: The name of the variable to plot.
    :return: None
    """
    plt.figure(figsize=(15, 5), dpi=150)
    plt.rcParams.update(fontdict)

    plt.plot(data, label="Simulated" if std is None else "Ensemble Mean", color='k', linewidth=2.)
    
    if std is not None:
        plt.fill_between(range(len(time)), (data - std), (data + std), color='blue', alpha=0.2)

    if ref is not None:
        plt.plot(ref, label=f'ERA5 {variable_name}', linestyle='-', color='orange', linewidth=2. )

    plt.xticks(rotation=45)
    plt.grid()
    plt.title(f'Annual Oscillation of {variable_name}')
    plt.xlabel('Time')


    # time is a list of tuples, each tuple containing (year, month)
    # Create xticklabels of form 'YYYY-MM'
    xticklabels = [f"{y}-{m:02d}" for (y, m) in time]
    print(xticklabels)
    indices = list(range(len(xticklabels)))

    #if add_linear_trend:
    #    # Add a linear trend line
    #    z = np.polyfit(np.arange(len(data)), data, 1)
    #    p = np.poly1d(z)
    #    plt.plot(np.arange(len(data)), p(np.arange(len(data))), label='Trend', linestyle='--', color='black')

    mult = get_xlabel_multiplier(len(indices))
    plt.xticks(ticks=indices[::mult], labels=xticklabels[::mult], rotation=45)
    plt.ylabel(variable_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/annual_oscillation_{variable_name}.png')