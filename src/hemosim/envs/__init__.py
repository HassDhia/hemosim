"""Gymnasium environments for hemostasis and anticoagulation management."""

from gymnasium.envs.registration import register

register(
    id="hemosim/WarfarinDosing-v0",
    entry_point="hemosim.envs.warfarin_dosing:WarfarinDosingEnv",
)

register(
    id="hemosim/HeparinInfusion-v0",
    entry_point="hemosim.envs.heparin_infusion:HeparinInfusionEnv",
)

register(
    id="hemosim/DOACManagement-v0",
    entry_point="hemosim.envs.doac_management:DOACManagementEnv",
)

register(
    id="hemosim/DICManagement-v0",
    entry_point="hemosim.envs.dic_management:DICManagementEnv",
)

# POMDP-formulated variants (ISC-5). Lab ordering is an explicit action;
# observations are delayed noisy readings, not ground truth.
register(
    id="hemosim/HeparinInfusion-POMDP-v0",
    entry_point="hemosim.envs.heparin_infusion_pomdp:HeparinInfusionPOMDPEnv",
)
