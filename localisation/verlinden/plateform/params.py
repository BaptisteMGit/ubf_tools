#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   params.py
@Time    :   2024/05/17 10:22:59
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os

# Hardware limitations
if os.name == "nt":
    # Windows
    N_WORKERS = 8
    MAX_RAM_GB = 14
else:
    # Linux
    N_WORKERS = 90
    MAX_RAM_GB = 125
