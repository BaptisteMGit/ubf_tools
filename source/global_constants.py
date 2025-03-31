#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   global_constants.py
@Time    :   2025/03/31 13:59:01
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Constants
# ======================================================================================================================
eps = np.finfo(float).eps
diagonal_loading = (
    1e-8  # amount of diagonal loading when adding identity matrix to covariance matrix
)
