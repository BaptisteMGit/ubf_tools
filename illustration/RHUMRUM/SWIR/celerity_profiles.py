import os
import arlpy
import xarray as xr
import matplotlib.pyplot as plt

from publication.PublicationFigure import PubFigure

pubfig = PubFigure(legend_fontsize=20)

path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\cmems_mod_glo_phy-all_my_0.25deg_P1D-m_multi-vars_65.75E_27.50S_0.51-5902.06m_2013-01-01-2013-12-30.nc"
ds = xr.open_dataset(path)

# Plot
# f, axs = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
# for it in range(ds.sizes["time"]):
#     if it % 15 == 0:
#         ds.isel(time=it).so_glor.plot(
#             y="depth", yincrease=False, alpha=0.25, color="b", ax=axs[0]
#         )
#         ds.isel(time=it).thetao_glor.plot(
#             y="depth", yincrease=False, alpha=0.25, color="b", ax=axs[1]
#         )
#         axs[0].set_title("")
#         axs[1].set_title("")
#         axs[1].set_ylabel("")

# plt.tight_layout()
# plt.show()
root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\data\ssp"
ssp = arlpy.uwa.soundspeed(
    temperature=ds.thetao_glor,
    salinity=ds.so_glor,
    depth=ds.depth,
)
ssp.to_netcdf(
    os.path.join(root, "ssp_65.75E_27.50S_0.51-5902.06m_2013-01-01-2013-12-30.nc")
)
plt.figure()
for it in range(ds.sizes["time"]):
    if it % 10 == 0:
        ssp.isel(time=it).plot(y="depth", yincrease=False, alpha=0.25)
# plt.show()


path = r"cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m_multi-vars_65.75E_27.50S_0.51-5902.06m_2013-01-01-2013-12-30.nc"
ds_std = xr.open_dataset(path)

alpha = 3
temp_min = ds_std.thetao_mean - alpha * ds_std.thetao_std
temp_max = ds_std.thetao_mean + alpha * ds_std.thetao_std
salinity_min = ds_std.so_mean - alpha * ds_std.so_std
salinity_max = ds_std.so_mean + alpha * ds_std.so_std

ssp_min = arlpy.uwa.soundspeed(
    temperature=temp_min, salinity=salinity_min, depth=ds_std.depth
)
ssp_max = arlpy.uwa.soundspeed(
    temperature=temp_max, salinity=salinity_max, depth=ds_std.depth
)
ssp_mean = arlpy.uwa.soundspeed(
    temperature=ds_std.thetao_mean, salinity=ds_std.so_mean, depth=ds_std.depth
)

ssp_min.to_netcdf(
    os.path.join(root, "ssp_min_65.75E_27.50S_0.51-5902.06m_2013-01-01-2013-12-30.nc")
)
ssp_max.to_netcdf(
    os.path.join(root, "ssp_max_65.75E_27.50S_0.51-5902.06m_2013-01-01-2013-12-30.nc")
)
ssp_mean.to_netcdf(
    os.path.join(root, "ssp_mean_65.75E_27.50S_0.51-5902.06m_2013-01-01-2013-12-30.nc")
)


ssp_mean = ssp_mean.sel(time="2013-05-31")
ssp_min = ssp_min.sel(time="2013-05-31")
ssp_max = ssp_max.sel(time="2013-05-31")

plt.figure(figsize=(6, 8))
ssp_mean.plot(y="depth", yincrease=False, label="mean", color="k")

# Plot shaded area between min and max
plt.fill_betweenx(
    ssp_mean.depth.values.flatten(),
    ssp_min.values.flatten(),
    ssp_max.values.flatten(),
    color="k",
    alpha=0.2,
    label="3 std",
)
plt.xlabel("Sound celerity " + r"[m.s$^{-1}$]")
plt.ylabel("Depth [m]")
plt.title("")
plt.tight_layout()
# ssp_min.plot(y="depth", yincrease=False)
# ssp_max.plot(y="depth", yincrease=False)

plt.show()
# if __name__ == '__main__':
#     pass
