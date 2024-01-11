from colorama import Fore

""" Usefull constants """


# Physical properties
C0 = 1500  # Sound celerity in water (m/s)
RHO_W = 1000  # Water density (kg/m3)
SAND_PROPERTIES = {
    "rho": 1.9 * RHO_W,
    "c_p": 1650,  # P-wave celerity (m/s)
    "c_s": 0.0,  # S-wave celerity (m/s) TODO check and update
    "a_p": 0.8,  # Compression wave attenuation (dB/wavelength)
    "a_s": 2.5,  # Shear wave attenuation (dB/wavelength)
}  # Sand properties from Jensen et al. (2000) p.39


# Verlinden parameters
MAX_LOC_DISTANCE = 50 * 1e3  # Maximum localisation distance (m)
AVERAGE_LOC_DISTANCE = 10 * 1e3  # Average localisation distance (m)
LIBRARY_COLOR = "blue"
EVENT_COLOR = "black"

# TDQM bar format
BAR_FORMAT = "%s{l_bar}%s{bar}%s{r_bar}%s" % (
    Fore.YELLOW,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.RESET,
)
