import xarray as xr


class ClimatePlotter:
    def __init__(self, model_data: dict, reference_data: dict = None):
        """
        Initialize the ClimatePlotter with model and reference data.
        args:
            model_data (dict): Dictionary containing model metrics with keys being the model names and values being the paths to the climate data files.
            reference_data (dict): Dictionary containing reference data names as keys and values being the paths to the climate data files.
        """
    
        self.model_data = model_data
        self.reference_data = reference_data


    def iter_data(self, var, lvl, fnc, data_path, fnc_kwargs: dict = {}):
        for model_name, model_path in self.model_data.items():
            label = model_name
            fpath = model_path + f"/{data_path}.nc"
            data = xr.load_dataset(fpath)
            if lvl is not None:
                x = data.sel(level=lvl)[var]
            else:
                x = data[var]
            
            fnc(x, **fnc_kwargs)

        if self.reference_data:
            self.reference_metrics = {}
            for ref_name, ref_path in self.reference_data.items():
                label = model_name
                fpath = model_path + f"/{data_path}.nc"
                fnc(self.model_metrics[model_name], **fnc_kwargs)

    def iter_variables(self, fnc, data_path, fnc_kwargs: dict = {}):
        """
        This class serves as a wrapper around iter_data to iterate over multiple variables and levels.
        args:
            fnc (function): Plotting function to apply to each variable and level.
            data_path (str): Path to the data file (e.g. annual_cycle/data.nc).
            fnc_kwargs (dict): Additional keyword arguments to pass to the plotting function.
        """

        for var, lvl in self.variables:
            for model_name, model_path in self.model_data.items():
                label = model_name
                fpath = model_path + f"/{data_path}.nc"
                data = xr.load_dataset(fpath)
                if lvl is not None:
                    x = data.sel(level=lvl)[var]
                else:
                    x = data[var]
                
                fnc(x, **fnc_kwargs)

        if self.reference_data:
            self.reference_metrics = {}
            for ref_name, ref_path in self.reference_data.items():
                label = model_name
                fpath = model_path + f"/{data_path}.nc"
                fnc(self.model_metrics[model_name], **fnc_kwargs)
            
    def plot_annual_cycle(self):
        # This method plots the annual cycle of the model runs and reference data
        
        file_path = "annual_cycle/data.nc"
        self.iter_data(self.compute_annual_cycle, file_path, fnc_kwargs={'var': 'temperature'})
