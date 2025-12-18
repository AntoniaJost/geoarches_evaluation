def _compute_indian_monsoon_index(self, data, period, kinetic_energy=False):
    u850_1 = data.sel(longitude=slice(220, 260), latitude=slice(15, 5), level=850)[
        "u_component_of_wind"
    ]
    u850_2 = data.sel(longitude=slice(250, 270), latitude=slice(30, 20), level=850)[
        "u_component_of_wind"
    ]

    if period:
        u850_1 = u850_1.sel(time=slice(period[0], period[1]))
        u850_2 = u850_2.sel(time=slice(period[0], period[1]))

    if kinetic_energy:
        v850_1 = data.sel(longitude=slice(220, 260), latitude=slice(15, 5), level=850)[
            "v_component_of_wind"
        ]
        v850_2 = data.sel(longitude=slice(250, 270), latitude=slice(30, 20), level=850)[
            "v_component_of_wind"
        ]

        ke850_1 = 0.5 * (u850_1**2 + v850_1**2).mean(["latitude", "longitude"])
        ke850_2 = 0.5 * (u850_2**2 + v850_2**2).mean(["latitude", "longitude"])

        ke850 = ke850_1 + ke850_2
        ke850 = ke850.groupby("time.dayofyear").mean(dim=["time"])
        return ke850
    else:
        imd_index = u850_1.mean(["latitude", "longitude"]) - u850_2.mean(
            ["latitude", "longitude"]
        )

        imd_index = imd_index.groupby("time.dayofyear").mean(dim=["time"])

        print("imd_index: ", imd_index)

        return imd_index
