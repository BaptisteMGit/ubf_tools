N_FILEIN_RIGHT_ALIGNMENT = 50


def write_kraken_envfile(
    src_properties,
    rcv_properties,
    env_properties,
    bathymetry,
    water_ssp,
    sediment_properties,
    title,
    outfile,
):
    """
    Write env file.env to be used by KRAKEN.exe
    :param src_properties: (dict) Source properties (keys = 'frequency', 'depth')
    :param rcv_properties: (dict) Receiver properties (keys = 'depth')
    :param env_properties: (dict) Environment properties (keys = 'range_max', 'dr', 'range_decimination_factor',
    'depth_max', 'dz', 'depth_decimination_factor', 'depth_max_output', 'ref_sound_speed', 'nb_term_rational_approx',
     'nb_stability_constaints', 'max_range_stability_constraints')
    :param bathymetry: (dict) Bathymetry data (keys = 'range', 'depth')
    :param water_ssp: (dict) Sound celerity profiles (keys = 'range', 'depth', 'sound_celerity')
    :param sediment_properties: (dict) Sediment properties (keys = 'ssp', 'rhob', 'attn')
    :param title: (str) Simulation title
    :param outfile: (str) Output filepath
    :return:
    """

    with open(outfile, "w") as f_out:
        f_out.write(title + "\n")
        f_out.write(
            align_var_description(
                var_line=f"{src_properties['nominal_frequency']}",
                desc="Nominal frequency (Hz)",
            )
        )
        f_out.write(
            align_var_description(
                var_line=f"{env_properties['nmedia']}",
                desc="NMEDIA",
            )
        )
        f_out.write(
            align_var_description(
                var_line=f"{env_properties['options_code']}",
                desc=env_properties["options_description"],
            )
        )
        f_out.write(
            align_var_description(
                var_line=f"{env_properties['top_halfspace']} {env_properties['nb_term_rational_approx']} "
                f"{env_properties['nb_stability_constaints']} {env_properties['max_range_stability_constraints']}",
                desc="c0 np ns rs",
            )
        )
        f_out.writelines(get_bathymetry_input_lines(bathymetry))
        f_out.writelines(
            get_water_sediment_lines(
                water_ssp,
                sediment_properties["ssp"],
                sediment_properties["rhob"],
                sediment_properties["attn"],
            )
        )


def align_var_description(var_line, desc):
    """
    Add variable description at the end of the line
    :param var_line:
    :param desc:
    :return:
    """
    return var_line + f"{desc : >{N_FILEIN_RIGHT_ALIGNMENT - len(var_line)}}\n"
