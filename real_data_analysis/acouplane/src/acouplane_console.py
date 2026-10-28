#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   acouplane_console.py
@Time    :   2026/10/27 10:00:00
@Author  :   Menetrier Baptiste
@Version :   2.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Console messages and figure saving helpers of the scripts.
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import os
import matplotlib.pyplot as plt


def make_logger(portion_id, verbose=True):
    """Build the function printing the progress messages of a script.

    Parameters
    ----------
    portion_id : int
        Identifier of the portion, shown in the main messages.
    verbose : bool, optional
        If False, the returned function prints nothing.

    Returns
    -------
    callable
        ``log(message, level=0)``: level 0 is a main step (prefixed by
        ``[Portion <id>]``), level 1 or more a detail of the current step.
    """

    def log(message, level=0):
        # Nothing is printed in silent mode
        if not verbose:
            return

        if level == 0:
            print(f"[Portion {portion_id}] {message}")
        else:
            print("    " * level + f"- {message}")

    return log


def make_figure_saver(directory, log=None):
    """Build the function saving, then closing, a figure in a directory.

    Parameters
    ----------
    directory : str
        Directory where the figures are saved (created if needed).
    log : callable or None, optional
        Progress function (see ``make_logger``) used to print the saved path.

    Returns
    -------
    callable
        ``save_figure(fig, filename)``.
    """
    os.makedirs(directory, exist_ok=True)

    def save_figure(fig, filename):
        # Save then close the figure to free the memory
        fpath = os.path.join(directory, filename)
        fig.savefig(fpath)
        plt.close(fig)

        if log is not None:
            log(f"Saved {fpath}", level=2)

    return save_figure
