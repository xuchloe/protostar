from astropy.io import fits
import numpy as np
import emcee
from find_source import summary
import corner
import math
from astropy.coordinates import Angle
import astropy.units as units
import scipy.special as sp
import warnings
import itertools
from uncertainties import ufloat
import sigfig
import numpy.typing as npt
from pathlib import Path

def p_model(
    p_params: npt.ArrayLike,
    u: float | npt.ArrayLike,
    v: float | npt.ArrayLike,
) -> complex | npt.NDArray:
    """
    Evaluate the complex visibility of a point source model.

    Parameters
    ----------
    p_params : npt.ArrayLike
        Point source model parameters `(flux, ra, dec)`, where `flux` is the
        total source flux, in Jy, and `ra` and `dec` are the source position
        offsets from the field center, in radians.
    u : float | npt.ArrayLike
        `u` coordinate of the visibility, in wavelengths.
    v : float | npt.ArrayLike
        `v` coordinate of the visibility, in wavelengths.

    Returns
    -------
    float | npt.ArrayLike
        Complex visibility value of the point source model.

    References
    ----------
    .. [1] A. R. Thompson, J. M. Moran, and G. W. Swenson, Jr.,
        *Interferometry and Synthesis in Radio Astronomy*, 3rd ed.,
        Springer, 2017. https://doi.org/10.1007/978-3-319-44431-4
    """
    flux, ra, dec = p_params
    return flux * np.exp(-2 * np.pi *1j * (u * ra + v * dec))


def c_model(
    c_params: npt.ArrayLike,
    u: float | npt.ArrayLike,
    v: float | npt.ArrayLike,
) -> complex | npt.NDArray:
    """
    Evaluate the complex visibility of a circular Gaussian source model.

    Parameters
    ----------
    c_params : npt.ArrayLike
        Circular Gaussian source model parameters `(flux, ra, dec, sigma)`,
        where `flux` is the total source flux, in Jy; `ra` and `dec` are the
        source position offsets from the field center, in radian; and `sigma`
        is the Gaussian sigma of the source profile, in radians.
    u : float | npt.ArrayLike
        `u` coordinate of the visibility, in wavelengths.
    v : float | npt.ArrayLike
        `v` coordinate of the visibility, in wavelengths.

    Returns
    -------
    float | npt.NDArray
        Complex visibility value of the circular Gaussian source model.

    References
    ----------
    .. [1] A. R. Thompson, J. M. Moran, and G. W. Swenson, Jr.,
        *Interferometry and Synthesis in Radio Astronomy*, 3rd ed.,
        Springer, 2017. https://doi.org/10.1007/978-3-319-44431-4
    """
    flux, ra, dec, sigma = c_params
    visibility = (
        flux
        * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2))
        * np.exp(-2 * np.pi * 1j * (u * ra + v * dec))
    )
    return visibility


def g_model(
    g_params: npt.ArrayLike,
    u: float | npt.ArrayLike,
    v: float | npt.ArrayLike,
) -> float | npt.NDArray:
    """
    Evaluate the complex visibility of a elliptical Gaussian source model.

    Parameters
    ----------
    g_params : npt.ArrayLike
        Elliptical Gaussian source model parameters `(flux, ra, dec, sigma,
        ratio, vis_theta)`, where `flux` is the total source flux, in Jy;
        `ra` and `dec` are the source position offsets from the field center,
        in radians; `sigma` is the Gaussian sigma of the source profile, in
        radians; `ratio` is the ratio of minor to major axis; and `vis_theta`
        is the source angle in visibility space, in radians.
    u : float | npt.ArrayLike
        `u` coordinate of the visibility, in wavelengths.
    v : float | npt.ArrayLike
        `v` coordinate of the visibility, in wavelengths.

    Returns
    -------
    float | npt.NDArray
        Complex visibility value of the elliptical Gaussian source model.

    References
    ----------
    .. [1] A. R. Thompson, J. M. Moran, and G. W. Swenson, Jr.,
        *Interferometry and Synthesis in Radio Astronomy*, 3rd ed.,
        Springer, 2017. https://doi.org/10.1007/978-3-319-44431-4
    """
    flux, ra, dec, sigma, ratio, vis_theta = g_params
    visibility = (
        flux
        * np.exp(
            -2 * np.pi**2 * sigma**2 *
            (
                (u * np.cos(vis_theta) - v * np.sin(vis_theta))**2
                + (u * np.sin(vis_theta) + v * np.cos(vis_theta))**2
                * ratio**2
            )
        )
        * np.exp(-2 * np.pi *1j * (u * ra + v * dec))
    )
    return visibility


def d_model(
    d_params: npt.ArrayLike,
    u: float | npt.ArrayLike,
    v: float | npt.ArrayLike,
) -> float | npt.NDArray:
    """
    Evaluate the complex visibility of a disk source model.

    Parameters
    ----------
    d_params : npt.ArrayLike
        Disk source model parameters `(flux, ra, dec, r, ratio, vis_theta)`,
        where `flux` is the total source flux, in Jy; `ra` and `dec` are the
        source position offsets from the field center, in radians; `r` is the
        source radius (or major axis if the disk is rotated), in radians;
        `ratio` is the ratio of minor to major axis; and `vis_theta` is the
        source angle in visibility space, in radians.
    u : float | npt.ArrayLike
        `u` coordinate of the visibility, in wavelengths.
    v : float | npt.ArrayLike
        `v` coordinate of the visibility, in wavelengths.

    Returns
    -------
    float | npt.NDArray
        Complex visibility value of the disk source model.

    References
    ----------
    .. [1] A. R. Thompson, J. M. Moran, and G. W. Swenson, Jr.,
        *Interferometry and Synthesis in Radio Astronomy*, 3rd ed.,
        Springer, 2017. https://doi.org/10.1007/978-3-319-44431-4
    """
    flux, ra, dec, r, ratio, vis_theta = d_params
    u_theta = u * np.cos(vis_theta) - v * np.sin(vis_theta)
    v_theta = u * np.sin(vis_theta) + v * np.cos(vis_theta)
    q_theta = r * np.sqrt(u_theta**2 + ratio**2 * v_theta**2)
    visibility = (
        flux
        / (np.pi * q_theta)
        * sp.j1(2 * np.pi * q_theta)
        * np.exp(-2 * np.pi * 1j * (u * ra + v * dec))
    )
    return visibility


def p_p0(
    flux: float,
    rad_coord: tuple,
    rad_bmaj: float,
    width_p0: float,
    ratio_p0: float,
    theta_p0: float,
    n_walkers: float,
) -> npt.NDArray:
    """
    Generate point source model initial guesses for emcee walkers.

    Paramters
    ---------
    flux : float
        The total source flux, in Jy.
    rad_coord : tuple
        The source coordinate (`ra`, `dec`), where `ra` and `dec` are the
        source position offsets from the field center, in radians.
    rad_bmaj : float
        The beam major axis, in radians.
    width_p0 : float
        This parameter is currently unused.
    ratio_p0 : float
        This parameter is currently unused.
    theta_p0 : float
        This parameter is currently unused.
    n_walkers : float
        The number of emcee walkers.

    Returns
    -------
    npt.NDArray
        point source model initial guesses for `n_walkers` number of emcee
        walkers.

    Notes
    -----
    Assumes square pixels in the image domain.

    The `width_p0`, `ratio_p0`, and `theta_p0` parameters are retained for
    consistency with other initial guess generating functions but do not affect
    the point source initial guesses.
    """
    p0 = np.zeros((n_walkers, 3))
    for i in range(n_walkers):
        # Flux.
        p0[i,0] = np.random.uniform(0.95 * flux, 1.05 * flux)

        # RA.
        p0[i,1] = np.random.uniform(
            -rad_bmaj / 2 + rad_coord[0],
            rad_bmaj / 2 + rad_coord[0]
        )

        # Dec.
        p0[i,2] = np.random.uniform(
            -rad_bmaj / 2 + rad_coord[1],
            rad_bmaj / 2 + rad_coord[1]
        )
    return p0


def c_p0(
    flux: float,
    rad_coord: tuple,
    rad_bmaj: float,
    width_p0: float,
    ratio_p0: float,
    theta_p0: float,
    n_walkers: float,
) -> npt.NDArray:
    """
    Generate circular Gaussian source model initial guesses for emcee walkers.

    Paramters
    ---------
    flux : float
        The total source flux, in Jy.
    rad_coord : tuple
        The source coordinate (`ra`, `dec`), where `ra` and `dec` are the
        source position offsets from the field center, in radians.
    rad_bmaj : float
        The beam major axis, in radians.
    width_p0 : float
        A guess for the source width parameter, i.e. the source intensity
        standard deviation, in radians.
    ratio_p0 : float
        This parameter is currently unused.
    theta_p0 : float
        This parameter is currently unused.
    n_walkers : float
        The number of emcee walkers.

    Returns
    -------
    npt.NDArray
        Circular Gaussian source model initial guesses for `n_walkers` number
        of emcee walkers.

    Notes
    -----
    Assumes square pixels in the image domain.

    The `ratio_p0` and `theta_p0` parameters are retained for consistency with
    other initial guess generating functions but do not affect the circular
    Gaussian source initial guesses.
    """
    p0 = np.zeros((n_walkers, 4))
    for i in range(n_walkers):
        # Flux.
        p0[i,0] = np.random.uniform(0.95 * flux, 1.05 * flux)

        # RA.
        p0[i,1] = np.random.uniform(
            -rad_bmaj / 2 + rad_coord[0],
            rad_bmaj / 2 + rad_coord[0],
        )

        # Dec.
        p0[i,2] = np.random.uniform(
            -rad_bmaj / 2 + rad_coord[1],
            rad_bmaj / 2 + rad_coord[1]
        )

        # Width (sigma).
        p0[i,3] = np.random.uniform(
            0.8 * width_p0,
            1.2 * width_p0,
        )
    return p0


def g_p0(
    flux: float,
    rad_coord: tuple,
    rad_bmaj: float,
    width_p0: float,
    ratio_p0: float,
    theta_p0: float,
    n_walkers: float,
) -> npt.NDArray:
    """
    Generate elliptical Gaussian source model initial guesses for emcee
    walkers.

    Paramters
    ---------
    flux : float
        The total source flux, in Jy.
    rad_coord : tuple
        The source coordinate (`ra`, `dec`), where `ra` and `dec` are the
        source position offsets from the field center, in radians.
    rad_bmaj : float
        The beam major axis, in radians.
    width_p0 : float
        A guess for the source width parameter, i.e. the source intensity
        standard deviation, in radians.
    ratio_p0 : float
        A guess for the source minor axis to major axis ratio.
    theta_p0 : float
        A guess for the source angle in visibility space, in radians.
    n_walkers : float
        The number of emcee walkers.

    Returns
    -------
    npt.NDArray
        Circular Gaussian source model initial guesses for `n_walkers` number
        of emcee walkers.

    Notes
    -----
    Assumes square pixels in the image domain.
    """
    p0 = np.zeros((n_walkers, 6))
    for i in range(n_walkers):
        # Flux.
        p0[i,0] = np.random.uniform(0.95 * flux, 1.05 * flux)

        # RA.
        p0[i,1] = np.random.uniform(
            -rad_bmaj / 2 + rad_coord[0],
            rad_bmaj / 2 + rad_coord[0]
        )

        # Dec.
        p0[i,2] = np.random.uniform(
            -rad_bmaj / 2 + rad_coord[1],
            rad_bmaj / 2 + rad_coord[1],
        )

        # Width (sigma).
        p0[i,3] = np.random.uniform(
            0.8 * width_p0,
            1.2 * width_p0
        )

        # Ratio.
        p0[i,4] = np.random.uniform(ratio_p0 - 0.15, ratio_p0 + 0.15)
        if p0[i,4] <= 0:  # Avoid zero or negative ratio.
            p0[i,4] = 1e-3
        elif p0[i,4] > 1:  # Cap ratio at one.
            p0[1,4] = 1

        # Theta.
        p0[i,5] = np.random.uniform(
            theta_p0 - np.pi / 15,
            theta_p0 + np.pi / 15,
        )
        if p0[i,5] < -np.pi / 2:  # Set floor of angle at -pi/2 radians.
            p0[1,4] = -np.pi / 2
        elif p0[i,5] > np.pi / 2:  # Cap angle at pi/2 radians.
            p0[i,5] = np.pi / 2
    return p0


def d_p0(
    flux: float,
    rad_coord: tuple,
    rad_bmaj: float,
    width_p0: float,
    ratio_p0: float,
    theta_p0: float,
    n_walkers: float,
) -> npt.NDArray:
    """
    Generate disk source model initial guesses for emcee walkers.

    Paramters
    ---------
    flux : float
        The total source flux, in Jy.
    rad_coord : tuple
        The source coordinate (`ra`, `dec`), where `ra` and `dec` are the
        source position offsets from the field center, in radians.
    rad_bmaj : float
        The beam major axis, in radians.
    width_p0 : float
        A guess for the source width parameter, i.e. the source radius (or
        major axis if the source is rotated), in radians.
    ratio_p0 : float
        A guess for the source minor axis to major axis ratio.
    theta_p0 : float
        A guess for the source angle in visibility space, in radians.
    n_walkers : float
        The number of emcee walkers.

    Returns
    -------
    npt.NDArray
        Disk model initial guesses for `n_walkers` number of emcee walkers.

    Notes
    -----
    Assumes square pixels in the image domain.
    """
    p0 = np.zeros((n_walkers, 6))
    for i in range(n_walkers):
        # Flux.
        p0[i,0] = np.random.uniform(0.95 * flux, 1.05 * flux)

        # RA.
        p0[i,1] = np.random.uniform(
            -rad_bmaj / 2 + rad_coord[0],
            rad_bmaj / 2 + rad_coord[0],
        )

        # Dec.
        p0[i,2] = np.random.uniform(
            -rad_bmaj / 2 + rad_coord[1],
            rad_bmaj / 2 + rad_coord[1],
        )

        # Width (radius).
        p0[i,3] = np.random.uniform(0.8 * width_p0, 1.2 * width_p0)

        # Ratio.
        p0[i,4] = np.random.uniform(ratio_p0 - 0.15, ratio_p0 + 0.15)
        if p0[i,4] <= 0:  # Avoid zero or negative ratio.
            p0[i,4] = 1e-3
        elif p0[i,4] > 1:  # Cap ratio at one.
            p0[1,4] = 1

        # Angle.
        p0[i,5] = np.random.uniform(
            theta_p0 - np.pi / 15,
            theta_p0 + np.pi / 15,
        )
        if p0[i,5] < -np.pi/2:  # Set floor of angle -pi/2 radians.
            p0[1,4] = -np.pi/2
        elif p0[i,5] > np.pi/2:  # Cap angle at pi/2 radians.
            p0[i,5] = np.pi/2
    return p0


def all_p1(
    med_sd: npt.ArrayLike,
    n_walkers: float,
    chain: npt.ArrayLike,
) -> npt.NDArray:
    """
    Generate refined initial guesses for emcee walkers based on the results of
    a previous fit.

    Paramters
    ---------
    med_sd : npt.ArrayLike
        Median and standard deviation of source parameters from a previous fit.
        Each element should correspond to a source parameters and have the
        median at the 0th index and the standard deviation at the 1st index.
    n_walkers : float
        The number of emcee walkers.
    chain : npt.ArrayLike
        The chain of MCMC samples from the previous fit.

    Returns
    -------
    npt.NDArray
        Refined initial guess for `n_walkers` number of emcee walkers.
    """
    n_params = len(med_sd)
    p1 = np.zeros((n_walkers, n_params))
    for i in range(n_walkers):
        for j in range(n_params):
            # Ensure non-negative width parameter and ratio.
            if j in [3, 4]:
                p1[i,j] = np.random.uniform(
                    max(-2 * med_sd[j][1] + med_sd[j][0], 0),
                    2 * med_sd[j][1] + med_sd[j][0]
                )
                if j == 4:
                    if p1[i,j] == 0:
                        p1[i,j] = 1e-3  # Avoid zero ratio.
                    if p1[i,j] > 1:
                        p1[i,j] = 1  # Cap ratio at one.

            # Perform some more detailed analysis if the angle did not seem to
            # fit well (standard deviation greater than 10 degrees).
            elif j == 5 and med_sd[j][1] > 10 * np.pi / 180:
                vis_theta_samples = [params[j] for params in chain]
                neg_vis_thetas = [
                    theta for theta in vis_theta_samples if theta < 0
                ]
                pos_vis_thetas = [
                    theta for theta in vis_theta_samples if theta >= 0
                ]
                neg_med = np.median(neg_vis_thetas) if neg_vis_thetas else None
                pos_med = np.median(pos_vis_thetas) if pos_vis_thetas else None
                if neg_med is not None and pos_med is not None:
                    if abs(abs(neg_med) - pos_med) < 10 * np.pi / 180:
                        # Suggests a clustering near the bounds of -pi/2 and
                        # pi/2. Since -pi/2 and pi/2 are physically equivalent,
                        # guess -pi/2 to encourage convergence during the next
                        # fit.
                        theta_p0 = -np.pi / 2
                    else:
                        theta_p0 = np.median(vis_theta_samples)
                    # Prevent guesses below -pi/2.
                    p1[i,j] = np.random.uniform(
                        max(theta_p0 - np.pi / 36, -np.pi / 2),
                        theta_p0 + np.pi / 36
                    )
                # If all samples were between -pi/2 and 0 or between 0 and
                # pi/2, then the poor convergence may not have been a result
                # of clustering near the bounds. Then we simply try again.
                else:
                    p1[i,j] = np.random.uniform(
                        -2 * med_sd[j][1] + med_sd[j][0],
                        2 * med_sd[j][1] + med_sd[j][0]
                    )
            else:
                p1[i,j] = np.random.uniform(
                    -2 * med_sd[j][1] + med_sd[j][0],
                    2 * med_sd[j][1] + med_sd[j][0]
                )
    return p1


def p_prior(
    params: npt.ArrayLike,
    vis_priors: npt.ArrayLike,
) -> float:
    """
    Find the contribution of the priors on the log probability of parameter
    values for a point source model.

    Parameters
    ----------
    params : npt.ArrayLike
        Point source model parameters `(flux, ra, dec)`, where `flux` is the
        total source flux, in Jy, and `ra` and `dec` are the source position
        offsets from the field center, in radians.
    vis_priors : npt.ArrayLike
        Priors on the point source model. The 0th index corresponds to priors
        on `flux`, the 1st index corresponds to priors on `ra`, and the
        2nd index corresponds to priors on `dec`. The 0th index of the
        array-like object for each parameter corresponds to the minimum value,
        and the 1st index of the array-like object for each parameter
        corresponds to the maximum value. If the array-like object for the
        parameter is a tuple, then the endpoint(s) are not inclusive. If the
        array-like object is a list, then the endpoint(s) are inclusive.

    Returns
    -------
    float
        The contribution of the priors on the log probability of the given
        parameters.
        0 if the parameters are within the priors and -np.inf if not.
    """
    flux, ra, dec = params
    if vis_priors is not None:
        flux_priors = vis_priors[0]
        ra_priors = vis_priors[1]
        dec_priors = vis_priors[2]
        flux_min, flux_max = flux_priors
        ra_min, ra_max = ra_priors
        dec_min, dec_max = dec_priors

        if flux_min is not None and flux < flux_min:
            return -np.inf
        if flux_max is not None and flux > flux_max:
            return -np.inf
        if isinstance(flux_priors, tuple):
            if flux_min is not None and flux == flux_min:
                return -np.inf
            if flux_max is not None and flux == flux_max:
                return -np.inf
        if ra_min is not None and ra < ra_min:
            return -np.inf
        if ra_max is not None and ra > ra_max:
            return -np.inf
        if isinstance(ra_priors, tuple):
            if ra_min is not None and ra == ra_min:
                return -np.inf
            if ra_max is not None and ra == ra_max:
                return -np.inf
        if dec_min is not None and dec < dec_min:
            return -np.inf
        if dec_max is not None and dec > dec_max:
            return -np.inf
        if isinstance(dec_priors, tuple):
            if dec_min is not None and dec == dec_min:
                return -np.inf
            if dec_max is not None and dec == dec_max:
                return -np.inf
    return 0.0


def c_prior(
    params: npt.ArrayLike,
    vis_priors: npt.ArrayLike,
) -> float:
    """
    Find the contribution of the priors on the log probability of parameter
    values for a circular Gaussian source model.

    Parameters
    ----------
    params : npt.ArrayLike
        Circular Gaussian source model parameters `(flux, ra, dec, sigma)`,
        where `flux` is the total source flux, in Jy; `ra` and `dec` are the
        source position offsets from the field center, in radian; and `sigma`
        is the source intensity standard deviation, in radians.
    vis_priors : npt.ArrayLike
        Priors on the circular Gaussian source model. The 0th index corresponds
        to priors on `flux`, the 1st index corresponds to priors on `ra`, the
        2nd index corresponds to priors on `dec`, and the 3rd index corresopnds
        to priors on `sigma`. The 0th index of the array-like object for each
        parameter corresponds to the minimum value, and the 1st index of the
        array-like object for each parameter corresponds to the maximum value.
        If the array-like object for the parameter is a tuple, then the
        endpoint(s) are not inclusive. If the array-like object is a list, then
        the endpoint(s) are inclusive.

    Returns
    -------
    float
        The contribution of the priors on the log probability of the given
        parameters.
        0 if the parameters are within the priors and -np.inf if not.
    """
    flux, ra, dec, sigma = params
    # Hardcode the requirement that the width parameter of a source should be
    # positive.
    if sigma <= 0:
        return -np.inf

    if vis_priors is not None:
        flux_priors = vis_priors[0]
        ra_priors = vis_priors[1]
        dec_priors = vis_priors[2]
        sigma_priors = vis_priors[3]
        flux_min, flux_max = flux_priors
        ra_min, ra_max = ra_priors
        dec_min, dec_max = dec_priors
        sigma_min, sigma_max = sigma_priors

        if flux_min is not None and flux < flux_min:
            return -np.inf
        if flux_max is not None and flux > flux_max:
            return -np.inf
        if isinstance(flux_priors, tuple):
            if flux_min is not None and flux == flux_min:
                return -np.inf
            if flux_max is not None and flux == flux_max:
                return -np.inf
        if ra_min is not None and ra < ra_min:
            return -np.inf
        if ra_max is not None and ra > ra_max:
            return -np.inf
        if isinstance(ra_priors, tuple):
            if ra_min is not None and ra == ra_min:
                return -np.inf
            if ra_max is not None and ra == ra_max:
                return -np.inf
        if dec_min is not None and dec < dec_min:
            return -np.inf
        if dec_max is not None and dec > dec_max:
            return -np.inf
        if isinstance(dec_priors, tuple):
            if dec_min is not None and dec == dec_min:
                return -np.inf
            if dec_max is not None and dec == dec_max:
                return -np.inf
        if sigma_min is not None and sigma < sigma_min:
            return -np.inf
        if sigma_max is not None and sigma > sigma_max:
            return -np.inf
        if isinstance(sigma_priors, tuple):
            if sigma_min is not None and sigma == sigma_min:
                return -np.inf
            if sigma_max is not None and sigma == sigma_max:
                return -np.inf
    return 0.0


def g_prior(
    params: npt.ArrayLike,
    vis_priors: npt.ArrayLike,
) -> float:
    """
    Find the contribution of the priors on the log probability of parameter
    values for a circular Gaussian source model.

    Parameters
    ----------
    params : npt.ArrayLike
        Circular Gaussian source model parameters `(flux, ra, dec, sigma,
        ratio, vis_theta)`, where `flux` is the total source flux, in Jy; `ra`
        and `dec` are the source position offsets from the field center, in
        radians; `sigma` is the source intensity standard deviation, in
        radians; `ratio` is the ratio of minor to major axis; and `vis_theta`
        is the source angle in visibility space, in radians.
    vis_priors : npt.ArrayLike
        Priors on the circular Gaussian source model. The 0th index
        corresponds to priors on `flux`, the 1st index corresponds to priors
        on `ra`, the 2nd index corresponds to priors on `dec`, the 3rd index
        corresopnds to priors on `sigma`, the 4th index corresponds to priors
        on `ratio`, and the 5th index corresponds to priors on `vis_theta`. The
        0th index of the array-like object for each parameter corresponds to
        the minimum value, and the 1st index of the array-like object for each
        parameter corresponds to the maximum value. If the array-like object
        for the parameter is a tuple, then the endpoint(s) are not inclusive.
        If the array-like object is a list, then the endpoint(s) are inclusive.

    Returns
    -------
    float
        The contribution of the priors on the log probability of the given
        parameters.
        0 if the parameters are within the priors and -np.inf if not.
    """
    flux, ra, dec, sigma, ratio, vis_theta = params
    # Hardcode the requirement that the width parameter of a source should be
    # positive.
    if sigma <= 0:
        return -np.inf
    # Hardcode the requirement that the ratio of the source's minor axis to
    # major axis should be positive but less than or equal to 1.
    if ratio <= 0:
        return -np.inf
    if ratio > 1:
        return -np.inf
    # Hardcode the requirement that the source angle should be between -pi/2
    # and pi/2, inclusive. This prevents the fit from spinning in circles.
    if vis_theta < -np.pi/2:
        return -np.inf
    if vis_theta > np.pi/2:
        return -np.inf

    if vis_priors is not None:
        flux_priors = vis_priors[0]
        ra_priors = vis_priors[1]
        dec_priors = vis_priors[2]
        sigma_priors = vis_priors[3]
        ratio_priors = vis_priors[4]
        vis_theta_priors = vis_priors[5]
        flux_min, flux_max = flux_priors
        ra_min, ra_max = ra_priors
        dec_min, dec_max = dec_priors
        sigma_min, sigma_max = sigma_priors
        ratio_min, ratio_max = ratio_priors
        vis_theta_min, vis_theta_max = vis_theta_priors

        if flux_min is not None and flux < flux_min:
            return -np.inf
        if flux_max is not None and flux > flux_max:
            return -np.inf
        if isinstance(flux_priors, tuple):
            if flux_min is not None and flux == flux_min:
                return -np.inf
            if flux_max is not None and flux == flux_max:
                return -np.inf
        if ra_min is not None and ra < ra_min:
            return -np.inf
        if ra_max is not None and ra > ra_max:
            return -np.inf
        if isinstance(ra_priors, tuple):
            if ra_min is not None and ra == ra_min:
                return -np.inf
            if ra_max is not None and ra == ra_max:
                return -np.inf
        if dec_min is not None and dec < dec_min:
            return -np.inf
        if dec_max is not None and dec > dec_max:
            return -np.inf
        if isinstance(dec_priors, tuple):
            if dec_min is not None and dec == dec_min:
                return -np.inf
            if dec_max is not None and dec == dec_max:
                return -np.inf
        if sigma_min is not None and sigma < sigma_min:
            return -np.inf
        if sigma_max is not None and sigma > sigma_max:
            return -np.inf
        if isinstance(sigma_priors, tuple):
            if sigma_min is not None and sigma == sigma_min:
                return -np.inf
            if sigma_max is not None and sigma == sigma_max:
                return -np.inf
        if ratio_min is not None and ratio < ratio_min:
            return -np.inf
        if ratio_max is not None and ratio > ratio_max:
            return -np.inf
        if isinstance(ratio_priors, tuple):
            if ratio_min is not None and ratio == ratio_min:
                return -np.inf
            if ratio_max is not None and ratio == ratio_max:
                return -np.inf
        if vis_theta_min is not None and vis_theta < vis_theta_min:
            return -np.inf
        if vis_theta_max is not None and vis_theta > vis_theta_max:
            return -np.inf
        if isinstance(vis_theta_priors, tuple):
            if vis_theta_min is not None and vis_theta == vis_theta_min:
                return -np.inf
            if vis_theta_max is not None and vis_theta == vis_theta_max:
                return -np.inf
    return 0.0


def d_prior(
    params: npt.ArrayLike,
    vis_priors: npt.ArrayLike,
) -> float:
    """
    Find the contribution of the priors on the log probability of parameter
    values for a disk source model.

    Parameters
    ----------
    params : npt.ArrayLike
        Disk source model parameters `(flux, ra, dec, r, ratio, vis_theta)`,
        where `flux` is the total source flux, in Jy; `ra` and `dec` are the
        source position offsets from the field center, in radians; `r` is the
        source radius (or major axis if the disk is rotated), in radians;
        `ratio` is the ratio of minor to major axis; and `vis_theta` is the
        source angle in visibility space, in radians.
    vis_priors : npt.ArrayLike
        Priors on the disk source model. The 0th index corresponds to priors on
        `flux`, the 1st index corresponds to priors on `ra`, the 2nd index
        corresponds to priors on `dec`, the 3rd index corresopnds to priors on
        `sigma`, the 4th index corresponds to priors on `ratio`, and the 5th
        index corresponds to priors on `vis_theta`. The 0th index of the
        array-like object for each parameter corresponds to the minimum value,
        and the 1st index of the array-like object for each parameter
        corresponds to the maximum value. If the array-like object for the
        parameter is a tuple, then the endpoint(s) are not inclusive. If the
        array-like object is a list, then the endpoint(s) are inclusive.

    Returns
    -------
    float
        The contribution of the priors on the log probability of the given
        parameters.
        0 if the parameters are within the priors and -np.inf if not.
    """
    flux, ra, dec, r, ratio, vis_theta = params
    # Hardcode the requirement that the width parameter of a source should be
    # positive.
    if r <= 0:
        return -np.inf
    # Hardcode the requirement that the ratio of the source's minor axis to
    # major axis should be positive but less than or equal to 1.
    if ratio <= 0:
        return -np.inf
    if ratio > 1:
        return -np.inf
    # Hardcode the requirement that the source angle should be between -pi/2
    # and pi/2, inclusive. This prevents the fit from spinning in circles.
    if vis_theta < -np.pi/2:
        return -np.inf
    if vis_theta > np.pi/2:
        return -np.inf

    if vis_priors is not None:
        flux_priors = vis_priors[0]
        ra_priors = vis_priors[1]
        dec_priors = vis_priors[2]
        r_priors = vis_priors[3]
        ratio_priors = vis_priors[4]
        vis_theta_priors = vis_priors[5]
        flux_min, flux_max = flux_priors
        ra_min, ra_max = ra_priors
        dec_min, dec_max = dec_priors
        r_min, r_max = r_priors
        ratio_min, ratio_max = ratio_priors
        vis_theta_min, vis_theta_max = vis_theta_priors

        if flux_min is not None and flux < flux_min:
            return -np.inf
        if flux_max is not None and flux > flux_max:
            return -np.inf
        if isinstance(flux_priors, tuple):
            if flux_min is not None and flux == flux_min:
                return -np.inf
            if flux_max is not None and flux == flux_max:
                return -np.inf
        if ra_min is not None and ra < ra_min:
            return -np.inf
        if ra_max is not None and ra > ra_max:
            return -np.inf
        if isinstance(ra_priors, tuple):
            if ra_min is not None and ra == ra_min:
                return -np.inf
            if ra_max is not None and ra == ra_max:
                return -np.inf
        if dec_min is not None and dec < dec_min:
            return -np.inf
        if dec_max is not None and dec > dec_max:
            return -np.inf
        if isinstance(dec_priors, tuple):
            if dec_min is not None and dec == dec_min:
                return -np.inf
            if dec_max is not None and dec == dec_max:
                return -np.inf
        if r_min is not None and r < r_min:
            return -np.inf
        if r_max is not None and r > r_max:
            return -np.inf
        if isinstance(r_priors, tuple):
            if r_min is not None and r == r_min:
                return -np.inf
            if r_max is not None and r == r_max:
                return -np.inf
        if ratio_min is not None and ratio < ratio_min:
            return -np.inf
        if ratio_max is not None and ratio > ratio_max:
            return -np.inf
        if isinstance(ratio_priors, tuple):
            if ratio_min is not None and ratio == ratio_min:
                return -np.inf
            if ratio_max is not None and ratio == ratio_max:
                return -np.inf
        if vis_theta_min is not None and vis_theta < vis_theta_min:
            return -np.inf
        if vis_theta_max is not None and vis_theta > vis_theta_max:
            return -np.inf
        if isinstance(vis_theta_priors, tuple):
            if vis_theta_min is not None and vis_theta == vis_theta_min:
                return -np.inf
            if vis_theta_max is not None and vis_theta == vis_theta_max:
                return -np.inf
    return 0.0


def log_likelihood(
    model: float | npt.ArrayLike,
    re: float | npt.ArrayLike,
    im: float | npt.ArrayLike,
    w: float | npt.ArrayLike,
) -> float:
    """
    Find the log likelihood of model value(s), given visibility data.

    Parameters
    ----------
    model : float | npt.ArrayLike
        The complex visibility data produced by the model function.
    re : float | npt.ArrayLike
        The real component(s) of visibility data, in Jy.
    im : float | npt.ArrayLike
        The imaginary component(s) of visibility data, in Jy.
    w : float | np.ArrayLike
        The data weight(s), equal to 1 over the uncertainty squared, in 1/Jy^2.

    Returns
    -------
    The log likelihood of the model value(s).

    Notes
    -----
    This function does not consider priors, in contrast to log_probability().
    """
    return -0.5 * np.sum(w * ((re - model.real)**2 + (im - model.imag)**2))


def p_flux(
    peak: float,
    width: float,
    ratio: float,
    rad_barea: float,
) -> float:
    """
    Convert the intensity of a point source in Jy/beam to flux in Jy.

    Parameters
    ----------
    peak : float
        The source intensity, in Jy/beam.
    width : float
        This parameter is currently unused.
    ratio : float
        This parameter is currently unused.
    rad_barea : float
        This parameter is currently unused.

    Returns
    -------
    float
        The source flux, in Jy.

    Notes
    -----
    The `width`, `ratio`, and `rad_barea` parameters are retained for
    consistency with other flux functions but do not affect the point source
    flux.
    """
    return peak


def c_flux(
    peak: float,
    width: float,
    ratio: float,
    rad_barea: float,
) -> float:
    """
    Convert the intensity of a circular Gaussian source in Jy/beam to flux in
    Jy.

    Parameters
    ----------
    peak : float
        The source intensity, in Jy/beam.
    width : float
        The Gaussian width parameter, in radians.
    ratio : float
        This parameter is currently unused.
    rad_barea : float
        The synthesized beam area, in radians^2.

    Returns
    -------
    float
        The source flux, in Jy.

    Notes
    -----
    The `ratio` parameter is retained for consistency with other flux functions
    but do not affect the point source flux.
    """
    n_beams = 2 * np.pi * width**2 / rad_barea
    if n_beams > 1:
        return peak * n_beams
    return peak


def g_flux(
    peak: float,
    width: float,
    ratio: float,
    rad_barea: float,
) -> float:
    """
    Convert the intensity of an elliptical Gaussian source in Jy/beam to flux in
    Jy.

    Parameters
    ----------
    peak : float
        The source intensity, in Jy/beam.
    width : float
        The Gaussian width parameter, in radians.
    ratio : float
        The Gaussian minor to major axis ratio.
    rad_barea : float
        The synthesized beam area, in radians^2.

    Returns
    -------
    float
        The source flux, in Jy.
    """
    n_beams = peak * 2 * np.pi * width**2 * ratio / rad_barea
    if n_beams > 1:
        return peak * n_beams
    return peak


def d_flux(
    peak: float,
    width: float,
    ratio: float,
    rad_barea: float,
) -> float:
    """
    Convert the intensity of a disk source in Jy/beam to flux in Jy.

    Parameters
    ----------
    peak : float
        The source intensity, in Jy/beam.
    width : float
        The disk radius (or major axis, if the disk is rotated), in radians.
    ratio : float
        The disk minor to major axis ratio.
    rad_barea : float
        The synthesized beam area, in radians^2.

    Returns
    -------
    float
        The source flux, in Jy.
    """
    n_beams = np.pi * width**2 * ratio / rad_barea
    if n_beams > 1:
        return peak * n_beams
    return peak


P_PARAMS = ['flux', 'ra', 'dec']
C_PARAMS = ['flux', 'ra', 'dec', 'sigma']
G_PARAMS = ['flux', 'ra', 'dec', 'sigma', 'ratio', 'vis_theta']
D_PARAMS = ['flux', 'ra', 'dec', 'r', 'ratio', 'vis_theta']

SOURCE_TYPES = {
    'p': [3, p_p0, p_prior, p_model, P_PARAMS, p_flux],
    'c': [4, c_p0, c_prior, c_model, C_PARAMS, c_flux],
    'g': [6, g_p0, g_prior, g_model, G_PARAMS, g_flux],
    'd': [6, d_p0, d_prior, d_model, D_PARAMS, d_flux]
}


def log_probability(
    params: npt.ArrayLike,
    sources: npt.ArrayLike,
    vis_priors: npt.ArrayLike,
    re: float | npt.ArrayLike,
    im: float | npt.ArrayLike,
    u: float | npt.ArrayLike,
    v: float | npt.ArrayLike,
    w: float | npt.ArrayLike,
    min_sep: float,
) -> float:
    """
    Find the log probability of the model, given the priors and visibility
    data.

    Parameters
    ----------
    params : npt.ArrayLike
        Parameter values whose log probability will be calculated.
    sources : npt.ArrayLike
        Source type(s) to identify the model(s) with which the parameter values
        will be evaluated.
    vis_priors : npt.ArrayLike
        Priors corresponding to the parameter values. For each parameter, the
        prior may either be `None` (no prior), [`min`, `max`] (inclusive), or
        (`min`, `max`) (exclusive).
    re : float | npt.ArrayLike
        The real component(s) of visibility data, in Jy.
    im : float | npt.ArrayLike
        The imaginary component(s) of visibility data, in Jy.
    u : float | npt.ArrayLike
        `u` coordinate of the visibility, in wavelengths.
    v : float | npt.ArrayLike
        `v` coordinate of the visibility, in wavelengths.
    w : float | npt.ArrayLike
        The data weight(s), equal to 1 over the uncertainty squared, in 1/Jy^2.
    rad_bmaj : float
        Beam major axis, in radians.
    rad_bmin : float
        Beam minor axis, in radians.
    min_sep : float
        The minimum separation, in radians, to differentiate two distinct
        sources.

    Returns
    -------
    float
        The log probability of the model.

    Notes
    -----
    If the parameters describe multiple sources separated by a distance of less
    than `min_sep`, then it is assumed that they describe the same source,
    instead of discinct sources as intended. As a result, the log probability
    returned will be negative infinity.

    The value of `min_sep` is assumed to be small, so the separation of sources
    is calculated using a flat space approximation instead of spherical
    offsets.
    """
    log_prior = 0.0
    model = 0.0
    start = 0
    ra_list = []
    dec_list = []
    for i in range(len(sources)):
        source = sources[i]
        n_params = SOURCE_TYPES[source][0]
        prior_func = SOURCE_TYPES[source][2]
        model_func = SOURCE_TYPES[source][3]
        source_params = params[start:start+n_params]
        ra_list.append(source_params[1])
        dec_list.append(source_params[2])
        lp = prior_func(source_params, vis_priors[i])
        if not np.isfinite(lp):
            return -np.inf
        log_prior += lp
        model += model_func(source_params, u, v)
        start += n_params
    # List of indices for ra_list and dec_list.
    indices = list(range(len(ra_list)))
    pairs = itertools.permutations(indices, 2)
    for (idx1, idx2) in pairs:
        dist = np.sqrt(
            (ra_list[idx1] - ra_list[idx2]) ** 2
            + (dec_list[idx1] - dec_list[idx2]) ** 2
        )
        if dist < min_sep:
            return -np.inf
    log_likelihood_value = log_likelihood(model, re, im, w)
    return log_prior + log_likelihood_value

def round_tuple(
    tup : tuple ,
    err_sigfigs: int = 3,
) -> tuple:
    """
    Given a tuple of a value and its uncertainty, round the to a given number
    of significant figures and round the value to the appropriate number of
    digits to match the rounded uncertainty.

    Parameters
    ----------
    tup : tuple
        A tuple of a value and its uncertainty as (`value`, `uncertainty`.
    err_sigfigs : int, optional
        The number of significant figures to which to round the uncertainty.

    Returns
    -------
    tuple
        (`value`, `uncertainty`), where `uncertainty` has been rounded to a
        given number of significant figures and `value` has been rounded to the
        appropriate number of digits to match `uncertainty`.
    """
    rounded_err = sigfig.round(float(tup[1]), sigfigs=err_sigfigs)
    str_err = str(rounded_err)
    places = 0
    if '.' in str_err:
        decimal = str_err.split('.')[1]
        places = len(decimal)
    return (round(float(tup[0]), places), rounded_err)


def sigmas(param_chain: npt.ArrayLike) -> tuple:
    """
    Find the 2.5th, 16th, 50th, 84th, and 97.5th percentile of data.

    Parameters
    ----------
    param_chain : npt.ArrayLike
        The data whose 2.5th, 16th, 50th, 84th, and 97.5th percentile will
        be found.

    Returns
    -------
    tuple of float
        The 2.5th, 16th, 50th, 84th, and 97.5th percentile of data.

    Notes
    -----
    The intention of this function is to give the user a quick way to see
    whether the parameter sampling was roughly Gaussian.
    """
    return (
        np.percentile(param_chain, 2.5),
        np.percentile(param_chain, 16),
        np.percentile(param_chain, 50),
        np.percentile(param_chain, 84),
        np.percentile(param_chain, 97.5)
    )


def auto_detect(
    vis: npt.ArrayLike,
    n_sources: int,
    info: dict,
    clean_output: bool = True,
    corner_plot: bool = True,
    min_sep: float | None = None,
) -> list:
    """
    Fit a given number of point sources to visibility data and summarize the
    results in a list and/or corner plot.

    Parameters
    ----------
    vis : npt.ArrayLike
        An array-like object with 6 indices, corresponding to the data for the
        frequency or frequency bin; the `u` coordinate of the visibility, in
        wavelengths; the `v` coordinate of the visibility, in wavelenghts; the
        real part of the visibility, in Jy; the imaginary part of the
        visibility, in Jy; and the weight (the inverse of the uncertainty
        squared) for each visibility measurement, in 1/Jy^2.
    n_sources : int
        The number of point sources in the model that will be fit to the data.
    info : dict
        Image domain information and some results from image domain fitting.
        Should include the following keys:
        `all_peaks` : npt.ArrayLike
            Array-like object with each element corresponding to a source
            detected in image fitting. Each element is an array-like object
            with the first index corresponding to the flux density, in Jy, of
            the peak detected in the image domain and the second index
            corresponding to the coordinates, each in arcseconds, of the
            detected peak.
        `n_peaks` : int
            The number of peaks detected in the image domain.
        `rad_barea` : float
            Beam area, in radians^2.
        `rad_bmaj` : float
            Beam major axis, in radians.
        `rad_bmin` : float
            Beam minor axis, in radians.
    clean_output : bool, optional
        Whether to return the list of fit results in a more human-parsable
        format. See Returns section for more details.
    corner_plot : bool, optional
        Whether to create a corner plot for each source.
    min_sep : float | None, optional
        The minimum separation, in radians, to differentiate two distinct
        sources.

    Returns
    -------
    list of dict
        A list of a dictionary with the following keys:
        `n_sources` : int
            The number of point sources for which the fit was performed.
        `result` : dict
            A dictionary with the following keys:
            `source_m` : dict
                A dictionary, for each source (m = 1 through m = `n_sources`),
                with the following keys:
                    `type` : str
                        The source type, as a single character code (a key of
                        SOURCE_TYPES).
                        By definition, 'p' for all the sources in
                        auto_detect().
                    `flux` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted flux estimate and its
                        uncertainty, respectively, in Jy/beam. The uncertainty
                        is rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sampled
                        parameters, in Jy, all rounded to 3 signficant figures.
                        Else, a tuple of the fitted flux estimate and its
                        uncertainty, respectively, in Jy.
                    `ra` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Right Ascension
                        estimate and its uncertainty, respectively, in
                        arcseconds relative to the field center. The
                        uncertainty is rounded to 3 significant figures and the
                        estimate is rounded to the appropriate number of digits
                        to match the uncertainty. The second tuple contains
                        the 2.5th, 16th, 50th, 84th, and 97.5th percentile of
                        sample parameters, in relative arcseconds, all rounded
                        to 3 significant figures.
                        Else, a tuple of the fitted Right Ascension estimate
                        and its uncertainty, respectively, in radians relative
                        to the field center.
                    `dec` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Declination estimate
                        and its uncertainty, respectively, in arcseconds
                        relative to the field center. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sample
                        parameters, in relative arcseconds, all rounded to 3
                        significant figures.
                        Else, a tuple of the fitted Declination estimate and
                        its uncertainty, respectively, in radians relative to
                        the field center.
                    `best` : dict
                        This key is only present if `clean_output` is `False`.
                        A dictionary, of the collection of sample parameters
                        that resulted in the lowest chi-squared value, with the
                        following keys:
                            `flux` : float
                                The flux, in Jy.
                            `ra` : float
                                The right ascension, in radians relative to the
                                field center.
                            `dec` : float
                                The declination, in radians relative to the
                                field center.
        `bic` : float
            The Bayesian Information Criterion (BIC), estimated as
            chi^2 + k * ln(n), where chi^2 is the chi-squared value of the
            fit, k is the number of parameters, and n is the number of data
            points to which the model is being fit.
            If `clean_output` is `True`, the BIC is rounded to the hundredths.

    Raises
    ------
    ValueError
        If a ValueError occurs during the emcee sampler run. Most likely due to
        extremely poor initial guesses or extremely poor model choice.
    RuntimeError
        Autocorrelation time for parameters could not be estimated. Most likely
        due to extremely poor initial guesses or extremely poor model choice.

    Notes
    -----
    If the number of sources to fit exceeds the number of sources with
    information in `info`, then the initial guess for the source location will
    be the a square around the origin, with side length equal to the beam major
    axis.
    """
    freq, u, v, re, im, w = vis
    all_peaks = info['all_peaks']  # Information from image fitting.
    n_peaks = info['n_peaks']
    rad_barea = info['rad_barea']
    rad_bmaj = info['rad_bmaj']
    rad_bmin = info['rad_bmin']

    if min_sep is None:
        min_sep = rad_bmaj / 10

    # For auto-detection, we do not impose additional priors, for generality,
    # simplicity, and speed.
    vis_priors = [
        [[None, None] for _ in range(6)] for _ in range(n_sources)
    ]

    # Calculate n_params and n_walkers.
    permutation = tuple(['p'] * n_sources)
    n_params = 0
    for i in range(n_sources):
        n_params += SOURCE_TYPES['p'][0]
    n_walkers = 2 * n_params

    # Determine initial guesses.
    width_p0 = None  # Since this will not be used for point-source fitting.
    ratio_p0 = None
    theta_p0 = None
    fluxes = [flux_coord[0] for flux_coord in all_peaks]
    min_index = np.argmin(fluxes)
    for i in range(n_sources):
        # If the number of sources to be fit exceeds the number of sources
        # found in the image domain, use the value of the dimmest peak detected
        # in the image domain to prompt the initial guess(es) for the
        # intensities of the remaining source(s).
        peak = all_peaks[i][0] if i < n_peaks else all_peaks[min_index][0]
        coord0 = all_peaks[i][1] if i < n_peaks else all_peaks[min_index][1]
        rad_coord = (
            float(Angle(coord0[0], units.arcsec).to(units.radian).value),
            float(Angle(coord0[1], units.arcsec).to(units.radian).value)
        )
        if i == 0:
            # For a point source, the intensity is the same as the flux.
            p0 = SOURCE_TYPES['p'][1](
                peak,
                rad_coord,
                rad_bmaj,
                width_p0,
                ratio_p0,
                theta_p0,
                n_walkers,
            )
        else:
            mini_p0 = SOURCE_TYPES['p'][1](
                peak,
                rad_coord,
                rad_bmaj,
                width_p0,
                ratio_p0,
                theta_p0,
                n_walkers,
            )
            # If the number of sources to be fit exceeds the number of sources
            # found in the image domain, allow the initial guess(es) for the
            # coordinates of the remaining source(s) to come from the a square
            # of side lengths rad_bmaj, centered around the origin.
            if i >= n_peaks:
                for j in range(n_walkers):
                    mini_p0[j,1] = np.random.uniform(
                        -rad_bmaj / 2,
                        rad_bmaj / 2,
                    )
                    mini_p0[j,2] = np.random.uniform(
                        -rad_bmaj / 2,
                        rad_bmaj / 2,
                    )
            p0 = np.append(p0, mini_p0, axis=1)

    # Set up and run MCMC.
    all_results = []
    n_steps = 100
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_params,
        log_probability,
        args=(
            permutation,
            vis_priors,
            re,
            im,
            u,
            v,
            w,
            min_sep,
        )
    )
    # Run the sampler for a bit to get an estimate for the autocorrelation
    # time.
    try:
        state = sampler.run_mcmc(p0, n_steps)
    except emcee.autocorr.AutocorrError:
        pass
    except ValueError as e:
        raise ValueError("Error encountered during MCMC run.") from e
    tau = sampler.get_autocorr_time(quiet=True)
    if np.isnan(tau).all():
        raise RuntimeError(
            "Autocorrelation time for parameters could not be estimated; "
            "all values are NaN."
        )
    int_tau = math.ceil(np.nanmax(tau))
    steps_to_50_tau = abs(int_tau * 50 - n_steps)

    # Run the sampler until it reaches 50 times the autocorrelation time.
    try:
        sampler.run_mcmc(state, steps_to_50_tau, skip_initial_state_check=True)
    except ValueError as e:
        raise ValueError("Error encountered during MCMC run.") from e
    chain = sampler.get_chain(discard = int_tau * 10, flat=True)
    log_probs = sampler.get_log_prob(discard = int_tau * 10, flat=True)
    max_prob_index = np.argmax(log_probs)

    # Find parameter estimates and uncertainties and calculate chi2.
    result = {}
    model = 0.0
    start = 0
    for i in range(n_sources):
        n_source_params = SOURCE_TYPES['p'][0]
        source_chain = chain[:, start:start + n_source_params]
        source_result = {'type': 'p'}
        temp_medians = []  # To store medians.
        temp_bests = {}  # To store best values (that maximimize probability).
        temp_max_probs = []  # To store candidate maximum probabilities.
        for j in range(n_source_params):
            samples = source_chain[:, j]
            temp_max_probs.append(samples[max_prob_index])
            samples_med = np.median(samples)
            samples_sd = np.nanstd(samples)
            param_name = SOURCE_TYPES['p'][4][j]
            source_result[param_name] = (float(samples_med), float(samples_sd))
            temp_bests[param_name] = float(samples[max_prob_index])
            temp_medians.append(samples_med)
        source_result['best'] = temp_bests
        model += SOURCE_TYPES['p'][3](
            temp_max_probs,
            u,
            v,
            rad_bmaj,
            rad_barea,
        )
        result[f'source_{i+1}'] = source_result
        start += n_source_params
    chi2 = float(np.sum(w * ((re - model.real)**2 + (im - model.imag)**2)))
    n = len(re)
    k = n_params
    bic = float(k * np.log(n) + chi2)
    all_results.append({
        'n_sources': n_sources,
        'result': result,
        'bic': bic,
        'chain': chain
    })

    if clean_output:
        result = all_results[0]['result']
        all_results[0]['bic'] = round(all_results[0]['bic'], 2)
        start = 0
        permutation_chain = all_results[0]['chain']
        for i in range(n_sources):
            source_key = f'source_{i+1}'
            source_result = result[source_key]
            source_params = SOURCE_TYPES['p'][4]
            n_source_params = SOURCE_TYPES['p'][0]
            n_walkers = 2 * n_source_params
            source_chain = permutation_chain[:, start:start+n_source_params]

            # Round flux and uncertainty and find percentiles.
            flux_chain = source_chain[:, 0]
            flux_sigmas = tuple([
                float(sigfig.round(sigma, sigfigs=3))
                for sigma in sigmas(flux_chain)
            ])
            source_result['flux'] = (
                round_tuple(
                    (source_result['flux'][0], source_result['flux'][1])
                ), flux_sigmas
            )

            # Convert ra, dec to arcsec. Round values and uncertainties and
            # find percentiles.
            ra_chain = source_chain[:, 1]
            dec_chain = source_chain[:, 2]
            ra_sigmas = tuple([
                float(
                    sigfig.round(
                        Angle(sigma, units.radian).to(units.arcsec).value,
                        sigfigs=3
                    )
                ) for sigma in sigmas(ra_chain)
            ])
            dec_sigmas = tuple([
                float(
                    sigfig.round(
                        Angle(sigma, units.radian).to(units.arcsec).value,
                        sigfigs=3
                    )
                ) for sigma in sigmas(dec_chain)
            ])
            source_result['ra'] = (round_tuple(
                tuple([
                    float(Angle(l, units.radian).to(units.arcsec).value)
                    for l in source_result['ra']
                ])), ra_sigmas
            )
            source_result['dec'] = (
                round_tuple(tuple([
                    float(Angle(m, units.radian).to(units.arcsec).value)
                    for m in source_result['dec']
                ])), dec_sigmas
            )

            del source_result['best']

            start += n_source_params

    # Create a corner plot for each source, if requested.
    if corner_plot:
        result = all_results[0]['result']
        chain = all_results[0]['chain']
        start = 0
        end = 0
        for i in range(n_sources):
            source_key = f'source_{i+1}'
            source_result = result[source_key]
            n_params = SOURCE_TYPES['p'][0]
            source_params = SOURCE_TYPES['p'][4]
            end += n_params
            if i == n_sources-1:
                fig = corner.corner(chain[:, start:], labels=source_params)
            else:
                fig = corner.corner(chain[:, start:end], labels=source_params)
            fig.suptitle(f'Source {i+1} of {n_sources}')
            start = end

    del all_results[0]['chain']

    return all_results


def best_auto_detect(
    fits_file: str | Path,
    n_sources: int | None = None,
    clean_output: bool = True,
    corner_plot: bool = True,
    min_sep: float | None = None,
    only_best: bool = True,
) -> list:
    """
    Fit visibility data from a FITS file to a given number of point sources or
    to different numbers of point sources, and summarize the results of the
    (best) fit in a list and/or corner plot.

    Paramters
    ---------
    fits_file : str | Path
        The path of the FITS file that contains the visibility data as its
        index 1 element.
    n_sources : int | None, optional
        The number of point sources in the model that will be fit to the data.
        If none, models with different numbers of point sources will be fit to
        the data and the results of the best-fit model will be reported.
    clean_output : bool, optional
        Whether to return the list of fit results in a more human-parsable
        format. See Returns section for more details.
    corner_plot : bool, optional
        Whether to create a corner plot for each source.
    min_sep : float | None, optional
        The minimum separation, in radians, to differentiate two distinct
        sources.
    only_best : bool, optional
        Whether to only return the best result.

    Returns
    -------
    list of dict
        A list of a dictionary with the following keys:
        `n_sources` : int
            The number of point sources for which the fit was performed.
        `result` : dict
            A dictionary with the following keys:
            `source_m` : dict
                A dictionary, for each source (m = 1 through m = `n_sources`),
                with the following keys:
                    `type` : str
                        The source type, as a single character code (a key of
                        SOURCE_TYPES).
                        By definition, 'p' for all the sources in
                        best_auto_detect().
                    `flux` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted flux estimate and its
                        uncertainty, respectively, in Jy. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sampled
                        parameters, in Jy, all rounded to 3 signficant figures.
                        Else, a tuple of the fitted flux estimate and its
                        uncertainty, respectively, in Jy.
                    `ra` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Right Ascension
                        estimate and its uncertainty, respectively, in
                        arcseconds relative to the field center. The
                        uncertainty is rounded to 3 significant figures and the
                        estimate is rounded to the appropriate number of digits
                        to match the uncertainty. The second tuple contains
                        the 2.5th, 16th, 50th, 84th, and 97.5th percentile of
                        sample parameters, in relative arcseconds, all rounded
                        to 3 significant figures.
                        Else, a tuple of the fitted Right Ascension estimate
                        and its uncertainty, respectively, in radians relative
                        to the field center.
                    `dec` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Declination estimate
                        and its uncertainty, respectively, in arcseconds
                        relative to the field center. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sample
                        parameters, in relative arcseconds, all rounded to 3
                        significant figures.
                        Else, a tuple of the fitted Declination estimate and
                        its uncertainty, respectively, in radians relative to
                        the field center.
                    `best` : dict
                        This key is only present if `clean_output` is `False`.
                        A dictionary, of the collection of sample parameters
                        that resulted in the lowest chi-squared value, with the
                        following keys:
                            `flux` : float
                                The flux, in Jy.
                            `ra` : float
                                The right ascension, in radians relative to the
                                field center.
                            `dec` : float
                                The declination, in radians relative to the
                                field center.
        `bic` : float
            The Bayesian Information Criterion (BIC), estimated as
            chi^2 + k * ln(n), where chi^2 is the chi-squared value of the
            fit, k is the number of parameters, and n is the  number of data
            points to which the model is being fit.
            If `clean_output` is `True`, the BIC is rounded to the hundredths.

    Raises
    ------
    ValueError
        If all models attempted fail to converge.

    Notes
    -----
    The visibility table should contain as its elements  rows of 6 elements
    each: the 0th index corresponding to the frequency or frequency bin, the
    1st index corresponding to the `u` coordinate, in wavelengths; the 2nd
    index corresponding to the `v` coordinate, in wavelengths; the 3rd index
    corresponding to the real component of the visibility, in Jy; the 4th index
    corresponding to the imaginary component of the visibility, in Jy; and the
    5th index corresponding to the visibility weight (the inverse of the
    uncertainty squared), in 1/Jy^2.

    Regardless of `clean_output`, corner plots will be of the visibility
    parameters. This means that the RA and Dec will be in radians, and the
    position angle will be in radians and rotated by pi/2 relative to the
    position angle in the image domain.

    This function employs Markov chain Monte Carlo (MCMC) via the emcee package
    to sample the parameter space.

    This function operates under several assumptions:
    1.  Assume that the presence of more than 2 internal detected peaks
        suggests the source is extended, rather than the presence of more than
        2 separate sources in the interior region.
    2.  Assume that extended external sources are extremely rare, so do not
        apply a similar assumption as (1) to cases with more than 2 external
        sources.
    3.  Assume that summary() more often has false positives than false
        negatives. As a consequence, when `n_sources` is None, begin guesses
        for the number of point sources at the number of peaks detected in the
        image domain and then try models with fewer and fewer sources.
    """
    # Extract data from the FITS file.
    fits_file = Path(fits_file)
    with fits.open(fits_file) as file:
        cunit1 = file[0].header['CUNIT1']
        data = file[1].data
        bmaj = file[0].header['BMAJ']
        bmin = file[0].header['BMIN']
        weighted_vis = False
        try:
            if file[1].header['EXTNAME'] == 'Weighted Visibilities':
                weighted_vis = True
        except KeyError:  # 'EXTNAME' is not a key.
            pass

    rad_bmaj = Angle(bmaj, cunit1).to(units.radian).value
    rad_bmin = Angle(bmin, cunit1).to(units.radian).value
    rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))

    # Get image domain fitting results.
    summ = summary(fits_file, plot=False)
    int_peaks = summ['int_peak_val']
    int_coords = summ['int_peak_coord']
    ext_peaks = summ['ext_peak_val']
    ext_coords = summ['ext_peak_coord']

    if min_sep is None:
        min_sep = rad_bmaj / 10

    # Assume the presence of more than 2 internal detected peaks suggests the
    # source is extended, rather than the presence of more than 2 separate
    # sources in the interior region.
    if len(int_peaks) > 2:
        int_peaks = int_peaks[:1]
        int_coords = int_coords[:1]

    int_info = list(zip(int_peaks, int_coords))
    if type(ext_peaks) is list:
        ext_info = list(zip(ext_peaks, ext_coords))
    else:
        ext_info = []

    # Create a list of image domain detected peak information. The list
    # elements are tuples (peak, (ra, dec)).
    all_peaks = int_info + ext_info
    all_peaks.sort(reverse=True)  # Sort by peak flux density (high to low).
    n_peaks = len(all_peaks)

    vis = np.array(data)
    freq_bin, u, v, re, im, w = [], [], [], [], [], []
    for row in vis:
        freq_bin_data, u_data, v_data, re_data, im_data, w_data = row
        freq_bin.append(int(freq_bin_data))
        u.append(int(u_data))
        v.append(int(v_data))
        if weighted_vis:
            re.append(float(re_data / w_data))
            im.append(float(im_data / w_data))
        else:
            re.append(float(re_data))
            im.append(float(im_data))
        w.append(float(w_data))

    # Add in the conjugate half of the visibility data.
    freq_bin *= 2
    neg_u = [-1 * val for val in u]
    u += neg_u
    neg_v = [-1 * val for val in v]
    v += neg_v
    re *= 2
    neg_im = [-1 * val for val in im]
    im += neg_im
    w *= 2

    freq_bin = np.array(freq_bin)
    u = np.array(u)
    v = np.array(v)
    re = np.array(re)
    im = np.array(im)
    w = np.array(w)

    input_vis = [freq_bin, u, v, re, im, w]
    info = {
        'all_peaks': all_peaks,
        'n_peaks': n_peaks,
        'rad_barea': rad_barea,
        'rad_bmaj': rad_bmaj,
        'rad_bmin': rad_bmin,
    }

    results = []
    if n_sources is None:
        # Assuming that summary() more often has false positives than false
        # negatives, start with a single-source model and increment up to the
        # number of peaks detected in the image domain.
        for i in range(1, n_peaks + 1):
            temp = auto_detect(
                vis=input_vis,
                n_sources=i,
                info=info,
                clean_output=clean_output,
                corner_plot=corner_plot,
            )
            if temp:
                if only_best:
                    if results:
                        if results[0]['bic'] - temp[0]['bic'] > 10:
                            results = temp
                    else:
                        results = temp
                else:
                    results.append(temp)
    else:
        results = auto_detect(
            vis=input_vis,
            n_sources=n_sources,
            info=info,
            clean_output=clean_output,
            corner_plot=corner_plot,
            min_sep=min_sep,
        )

    if results:
        return results
    else:
        raise ValueError("All attempts failed to converge.")


def estimate_beam_axes(
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    w: npt.ArrayLike,
) -> npt.ArrayLike:
    """
    Estimate the synthesized beam major and minor axes.

    Paramters:
    u : npt.ArrayLike
        `u` coordinate of the visibility, in wavelengths.
    v : npt.ArrayLike
        `v` coordinate of the visibility, in wavelengths.
    w : npt.ArrayLike
        The data weight(s), equal to 1 over the uncertainty squared, in 1/Jy^2.

    Returns
    -------
    npt.Arraylike
        An array [`b_maj`, `b_min`] of the estimated beam major (`b_maj`) and
        minor (`b_min`) axes, in radians.

    Notes
    -----
    Uses the method of moments with natural weighting.
    """
    uu_moment = np.sum(w * u**2) / np.sum(w)
    vv_moment = np.sum(w * v**2) / np.sum(w)
    uv_moment = np.sum(w * u * v) / np.sum(w)
    moment_mtx = np.array(
        [[uu_moment, uv_moment], [uv_moment, vv_moment]]
    )
    eigvals = np.sort(np.linalg.eigvals(moment_mtx))
    beam_axes = (
        np.sqrt(2 * np.log(2))
        / (np.pi * np.sqrt(eigvals))
    )
    return beam_axes


def uv_fit_no_image(
    vis: npt.ArrayLike,
    sources: list,
    info: dict | None = None,
    flux_guess: list | None = None,
    ra_guess: list | None = None,
    dec_guess: list | None = None,
    width_guess: list | None = None,
    ratio_guess: list | None = None,
    angle_guess: list | None = None,
    priors: list | None = None,
    clean_output: bool = True,
    corner_plot: bool = True,
    additional_runs: int = 2,
    extreme_case_check: bool = True,
    min_sep: float | None = None,
    use_image_guess : bool = False,
) -> list:
    """
    Fit visibility data from a FITS file to a model of a given source type or
    given source types.

    Parameters
    ----------
    vis : np.ArrayLike
        The visibility table. Elements of the table should themselves be
        array-like objects of 6 elements each: the 0th index corresponding to
        the frequency or frequency bin, the 1st index corresponding to the `u`
        coordinate, the 2nd index corresponding to the `v` coordinate, the 3rd
        index corresponding to the real component of the visibility, the 4th
        index corresponding to the imaginary component of the visibility, and
        the 5th index corresponding to the visibility weight.
    sources : list
        List of source types to fit. Each source type should be one of 'p'
        (point), 'c' (circular gaussian), 'g' (gaussian), 'd' (disk), or 'any'
        (try all and pick best fit).
    info : dict, optional
        Dictionary of observation information, with the following keys:
        `CUNIT1` : u.UnitBase
            The units of the first axis.
        `BMAJ` : float
            The beam major axis, in degrees.
        `BMIN` : float
            The beam minor axis, in degrees.
        `int_peak_val` : list
            A list of guesses for the internal peak intensities, in Jy/beam.
        `int_peak_coord` : tuple of float
            A list of guesses for the internal peak coordinates, in relative
            arcseconds.
        `ext_peak_val` : float
            A list of guesses for the external peak intensities, in Jy/beam.
        `ext_peak_coord` : tuple of float
            A list of guesses for the external peak coordinates, in relative
            arcseconds.
        `weighted_vis` : bool
            Whether the visibilities are weighted, i.e. whether the real and
            imaginary components of the visibility in the table have been
            multiplied by their weights.
        And optionally, the following keys:
        `U_RES` : float
            The resolution/scale of the `u` coordinates.
        `V_RES` : float
            The resolution/scale of the `v` coordinates.
    flux_guess : list | None, optional
        A guess for the flux(es) of the source(s), in Jy.
        If `flux_guess` is a list, each element is a guess for the
        corresponding source in `sources`, so the list must be the same length
        as `sources`. Elements in the list may be `None`.
    ra_guess : list | None, optional
        A guess for the Right Ascension(s) of the source(s), in arcseconds
        relative to the field center.
        If `ra_guess` is a list, each element is a guess for the corresponding
        source in `sources`, so the list must be the same length as `sources`.
        Elements in the list may be `None`.
    dec_guess : list | None, optional
        A guess for the Declination(s) of the source(s), in arcseconds relative
        to the field center.
        If `dec_guess` is a list, each element is a guess for the corresponding
        source in `sources`, so the list must be the same length as `sources`.
        Elements in the list may be `None`.
    width_guess : list | None, optional
        A guess for the width(s) of the source(s), in arcseconds.
        If `width_guess` is a list, each element is a guess for the
        corresponding source in `sources`, so the list must be the same length
        as `sources`. Elements in the list may be `None`, but a width guess is
        required for every source type except `p`.
    ratio_guess : list | None, optional
        A guess for the source's/sources' minor axis to major axis ratio(s).
        If `ratio_guess` is a list, each element is a guess for the
        corresponding source in `sources`, so the list must be the same length
        as `sources`. Elements in the list may be `None`, but a ratio guess is
        required for every source type except `p` and `c`.
    angle_guess : list | None, optional
        A guess for the source angle(s), in degrees.
        If `angle_guess` is a list, each element is a guess for the
        corresponding source in `sources`, so the list must be the same length
        as `sources`. Elements in the list may be `None`, but an angle guess is
        required for every source type except `p` and `c`.
    priors : list | None, optional
        Priors for the source(s).
        If `priors` is a list, each element may be `None` or a list of priors
        for the 6 parameters (flux, ra, dec, width, ratio, angle) for the
        corresponding source in `sources`. If `priors` is a list, it may
        either be of length 1, in which case the set of priors will be applied
        to all sources, or it must be the same length as `sources`. The list of
        priors must contain 6 elements (one for each parameter), that are
        either tuples (`min`, `max`) exclusive or lists [`min`, `max`]
        inclusive.
    clean_output : bool, optional
        Whether to return the list of fit results in a more human-parsable
        format. See Returns section for more details.
    additional_runs : int, optional
        The number of MCMC runs beyond the first to perform.
    extreme_case_check : bool, optional
        Whether to check for extreme cases: the brightest source is a very
        large Gaussian, a very large disk, or a point source.
    min_sep : float | None, optional
        The minimum separation, in radians, to differentiate two distinct
        sources.
    use_image_guess : bool, optional
        Whether to use the image domain fitting results as initial guesses for
        flux, RA, and dec in the visibility fitting.

    Returns
    -------
    list of dict
        A list of a dictionary/dictionaries with the following keys:
        `permutation` : tuple of str
            Tuple of source types corresponding to the model being fit.
        `n_params` : int
            The total number of parameters in the model.
        `result` : dict
            A dictionary with the following keys:
            `source_m` : dict
                A dictionary, for each source (m = 1 through m = `n_sources`),
                with the following keys:
                    `type` : str
                        The source type, as a single character code (a key of
                        SOURCE_TYPES).
                    `flux` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted flux estimate and its
                        uncertainty, respectively, in Jy. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sampled
                        parameters, in Jy, all rounded to 3 signficant figures.
                        Else, a tuple of the fitted flux estimate and its
                        uncertainty, respectively, in Jy.
                    `ra` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Right Ascension
                        estimate and its uncertainty, respectively, in
                        arcseconds relative to the field center. The
                        uncertainty is rounded to 3 significant figures and the
                        estimate is rounded to the appropriate number of digits
                        to match the uncertainty. The second tuple contains
                        the 2.5th, 16th, 50th, 84th, and 97.5th percentile of
                        sample parameters, in relative arcseconds, all rounded
                        to 3 significant figures.
                        Else, a tuple of the fitted Right Ascension estimate
                        and its uncertainty, respectively, in radians relative
                        to the field center.
                    `dec` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Declination estimate
                        and its uncertainty, respectively, in arcseconds
                        relative to the field center. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sample
                        parameters, in relative arcseconds, all rounded to 3
                        significant figures.
                        Else, a tuple of the fitted Declination estimate and
                        its uncertainty, respectively, in radians relative to
                        the field center.
                    `sigma` : tuple of tuple | tuple of float
                        This key is only present for sources of `type` 'c'.
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted circular Gaussian
                        standard deviation estimate and its uncertainty,
                        respectively, in arcseconds. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sample
                        parameters, in arcseconds, all rounded to 3 significant
                        figures.
                        Else, a tuple of the fitted circular Gaussian standard
                        deviation estimate and its uncertainty, respectively,
                        in radians.
                    `r` : tuple of float
                        This key is only present when `clean_output` is `False`
                        and the source is of `type` 'd'.
                        A tuple of the fitted disk radius estimate and its
                        uncertainty, respectively, in radians.
                    `theta` : tuple of tuple
                        This key is only present when `clean_output` is `True`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of 2 tuples. The first tuple contains the
                        fitted source angle estimate and its uncertainty,
                        respectively, in degrees. The uncertainty is rounded to
                        3 significant figures and the estimate is rounded to
                        the appropriate number of digits to match the
                        uncertainty. The second tuple contains the 2.5th, 16th,
                        50th, 84th, and 97.5th percentile of sample parameters,
                        in degrees, all rounded to 3 significant figures.
                    `vis_theta` : tuple of float
                        This key is only present when `clean_output` is `False`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of the fitted source angle estimate in
                        visibility space and its uncertainty, respectively, in
                        radians.
                    `ratio` : tuple of float
                        This key is only present when `clean_output` is `False`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of the fitted source minor axis to major axis
                        ratio and its uncertainty, respectively.
                    `major_axis` : tuple of tuple
                        This key is only present when `clean_output` is `True`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of 2 tuples. The first tuple contains the
                        fitted source major axis estimate and its uncertainty,
                        respectively, in arcseconds. The uncertainty is rounded
                        to 3 significant figures and the estimate is rounded to
                        the appropriate number of digits to match the
                        uncertainty. The second tuple contains the 2.5th, 16th,
                        50th, 84th, and 97.5th percentile of sample parameters,
                        in arcseconds, all rounded to 3 significant figures.
                    `minor_axis : tuple of tuple
                        This key is only present when `clean_output` is `True`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of 2 tuples. The first tuple contains the
                        fitted source minor axis estimate and its uncertainty,
                        respectively, in arcseconds. The uncertainty is rounded
                        to 3 significant figures and the estimate is rounded to
                        the appropriate number of digits to match the
                        uncertainty. The second tuple contains the 2.5th, 16th,
                        50th, 84th, and 97.5th percentile of sample parameters,
                        in arcseconds, all rounded to 3 significant figures.
                    `best` : dict
                        This key is only present if `clean_output` is `False`.
                        A dictionary, of the collection of sample parameters
                        that resulted in the lowest chi-squared value, with the
                        following keys:
                            `flux` : float
                                The flux, in Jy.
                            `ra` : float
                                The right ascension, in radians relative to the
                                field center.
                            `dec` : float
                                The declination, in radians relative to the
                                field center.
                            `r` : float
                                This key is only present if the source is of
                                `type` 'd'. The disk radius, in radians.
                            `sigma` : float
                                This key is only present if the source is of
                                `type` 'c' or 'g'. The source standard
                                deviation, in radians.
                            `ratio` : float
                                This key is only present if the source is of
                                `type` 'g' or 'd'. The ratio between the source
                                minor axis and major axis.
                            `vis_theta` : float
                                This key is only present if the source is of
                                `type` 'g' or 'd'. The source angle in
                                visibility space, in radians.

        `chi2` : float
            The chi-squared value of the fit.
            If `clean_output` is `True`, the chi-squared value is rounded to
            the hundredths.
        `bic` : float
            The Bayesian Information Criterion (BIC), estimated as
            chi^2 + k * ln(n), where chi^2 is the chi-squared value of the
            fit, k is the number of parameters, and n is the number of data
            points to which the model is being fit.
            If `clean_output` is `True`, the BIC is rounded to the hundredths.
        `red_chi2` : float
            The reduced chi-squared value of the fit.
            If `clean_output` is `True`, the reduced chi-squared value is
            rounded to the hundredths.

    Raises
    ------
    ValueError
        If `additional_runs` is negative.
        If `priors` has improper formatting.
        If no sources are specified.
        If improper source types are specified.
        If required initial guesses are missing.
        If initial guesses are outside of the proper bounds (width guesses are
        not positive, ratio guesses are not greater than 0 but less than or
        equal to 1, angle guesses are not between -90 and 90 degrees
        inclusive).

    Warns
    -----
    UserWarning
        If the number of peaks detected from the image fitting is less than the
        number of sources requested to fit.
        If the autocorrelation time for parameters of an MCMC run could not be
        estimated.
        If the best-fit model has a reduced chi-square value greater than 10.
        If `extreme_case_check` is true and the brightest source might be an
        extreme case (a very large Gaussian, a very large disk, or a point
        source).

    Notes
    -----
    If the number of peaks in `info` is less than the `n_sources`, and `info`
    is being used for initial guesses, then the remaining sources will have
    initial flux guesses calculated from the dimmest peak in `info` and initial
    position guesses randomly selected (uniformly) from a square of side
    length rad_bmaj, centered around the origin.

    Assumes that `CUNIT1` is equal to `CUNIT2`.

    If the number of sources to fit exceeds the number of sources with
    information in `info`, then the initial guess for the source location will
    be the a square around the origin, with side length equal to the beam major
    axis.

    Regardless of `clean_output`, corner plots will be of the visibility
    parameters. This means that the RA and Dec will be in radians, and the
    position angle will be in radians and rotated by pi/2 relative to the
    position angle in the image domain.

    This function employs Markov chain Monte Carlo (MCMC) via the emcee package
    to sample the parameter space.

    Additional MCMC runs use the fitting results from the previous run to
    create initial guesses.

    For the very large Gaussian and very large disk checks, the width initial
    guesses are created around twice the beam major axis. The flux initial
    guesses are created around the median visibility amplitude. If there is
    no ratio or angle guess, intial guesses will be created around 0.8 and
    around 0 radians.
    """
    n_sources = len(sources)

    # Check additional_runs.
    if additional_runs < 0:
        raise ValueError("'additional_runs' must be a non-negative integer.")

    # Check priors format.
    if priors is not None:
        if len(priors) != n_sources or len(priors) != 1:
            raise ValueError(
                "Length of priors must match length of sources, "
                "or priors must be of length 1 "
                "(one set of priors for all sources)."
            )
        for i in range(len(priors)):
            if priors[i] is not None:
                if type(priors[i]) is not list:
                    raise ValueError(
                        "Each element in priors must be None or a list "
                        "corresponding to a source."
                    )
                if len(priors[i]) != 6:
                    raise ValueError(
                        "Each prior list must have 6 elements corresponding "
                        "to ranges for flux, RA, declination, "
                        "width parameter, ratio, and position angle."
                    )
                for j in range(len(priors[i])):
                    if priors[i][j] is not None:
                        if type(priors[i][j]) not in [list, tuple]:
                            raise ValueError(
                                "Each prior range must be None, a list "
                                "(inclusive), or a tuple (exclusive)."
                            )
                        if len(priors[i][j]) != 2:
                            raise ValueError(
                                "Each prior range list or tuple must have "
                                "exactly two elements: min and max."
                            )

    # Check input source types.
    if n_sources == 0:
        raise ValueError(
            "No sources specified. Try specifying one or more sources of type "
            "'p' (point), 'c' (circular gaussian), 'g' (gaussian), "
            "'d' (disk), or 'any' (try all and pick best fit)."
        )
    for source in sources:
        if source not in SOURCE_TYPES and source != 'any':
            raise ValueError(
                f"Source type '{source}' is not recognized. Source type must "
                "be one of the following: 'p' (point), "
                "'c' (circular gaussian), 'g' (gaussian), 'd' (disk), or "
                "'any' (try all and pick best fit)."
            )

    # Make sure guesses, if they exist, have proper length.
    user_guesses = [
        flux_guess,
        ra_guess,
        dec_guess,
        width_guess,
        ratio_guess,
        angle_guess,
    ]
    for guess in user_guesses:
        if guess is not None:
            if len(guess) != n_sources:
                raise ValueError(
                    f"Length of {guess} is {len(guess)}, which does not match "
                    f"the length of sources ({n_sources})."
                )

     # Extract U_RES, V_RES information from info, if it exists.
    u_res = 1
    v_res = 1
    if info:
        if 'U_RES' in info.keys():
            u_res = info['U_RES']
        if 'V_RES' in info.keys():
            v_res = info['V_RES']

    # Extract data from visibility table.
    vis = np.array(vis)
    freq_bin, u, v, re, im, w = [], [], [], [], [], []
    for row in vis:
        freq_bin_data, u_data, v_data, re_data, im_data, w_data = row
        freq_bin.append(int(freq_bin_data))
        u.append(int(u_data * u_res))
        v.append(int(v_data * v_res))
        if info and info['weighted_vis']:
            re.append(float(re_data / w_data))
            im.append(float(im_data / w_data))
        else:
            re.append(float(re_data))
            im.append(float(im_data))
        w.append(float(w_data))

    # Add in the conjugate half of the visibility data.
    freq_bin *= 2
    neg_u = [-1 * val for val in u]
    u += neg_u
    neg_v = [-1 * val for val in v]
    v += neg_v
    re *= 2
    neg_im = [-1 * val for val in im]
    im += neg_im
    w *= 2

    freq_bin = np.array(freq_bin)
    u = np.array(u)
    v = np.array(v)
    re = np.array(re)
    im = np.array(im)
    w = np.array(w)

    # Extract beam information from info, if it exists.
    if info:
        cunit1 = info['CUNIT1']
        bmaj = info['BMAJ']
        bmin = info['BMIN']
        rad_bmaj = Angle(bmaj, cunit1).to(units.radian).value
        rad_bmin = Angle(bmin, cunit1).to(units.radian).value
    else:
        rad_bmaj, rad_bmin = estimate_beam_axes(u, v, w)
    rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))

    # Clean up priors and convert units.
    vis_priors = []
    if priors is None:
        priors = [None] * n_sources
    # If only one set of priors provided, use for it all sources.
    if len(priors) == 1 and n_sources > 1:
        priors = priors * n_sources
    for i in range(len(priors)):
        mini_vis_priors = []
        if priors[i] is not None:
            for j in range(len(priors[i])):
                if priors[i][j] is not None:
                    if j == 0:  # Flux, so keep as is.
                        mini_vis_priors.append(priors[i][j])
                    # Ra, dec, width parameter, so convert from arcsec to
                    # radians.
                    elif j in [1, 2, 3]:
                        rad_min = None
                        rad_max = None
                        if priors[i][j][0] is not None:
                            rad_min = float(
                                Angle(priors[i][j][0], units.arcsec)
                                .to(units.radian).value
                            )
                        if priors[i][j][1] is not None:
                            rad_max = float(
                                Angle(priors[i][j][1], units.arcsec)
                                .to(units.radian).value
                            )
                        if isinstance(priors[i][j], tuple):
                            mini_vis_priors.append((rad_min, rad_max))
                        else: # is list
                            mini_vis_priors.append([rad_min, rad_max])
                    elif j == 4:  # Ratio.
                        mini_vis_priors.append(priors[i][j])
                    elif j == 5:  # Angle.
                        # Convert from degrees to radians.
                        theta_min = None
                        theta_max = None
                        if priors[i][j][0] is not None:
                            if priors[i][j][0] < -90 or priors[i][j][0] > 90:
                                raise ValueError(
                                    "Position angle prior lower bound must be "
                                    "between -90 and 90 degrees, inclusive."
                                )
                            # Shift by 90 to go from image to visibility angle.
                            theta_min = (priors[i][j][0] - 90) * np.pi / 180
                            if theta_min < -90:
                                # -90 to -180 is the same as 90 to 0.
                                theta_min += 180
                        if priors[i][j][1] is not None:
                            if priors[i][j][1] < -90 or priors[i][j][1] > 90:
                                raise ValueError(
                                    "Position angle prior upper bound must be "
                                    "between -90 and 90 degrees, inclusive."
                                )
                            # Shift by 90 to go from image to visibility angle.
                            theta_max = (priors[i][j][1] - 90) * np.pi / 180
                            if theta_max < -90:
                                # -90 to -180 is the same as 90 to 0.
                                theta_max += 180
                        if isinstance(priors[i][j], tuple):
                            if theta_max is not None and theta_min is not None:
                                # Check because upper bound may now be less
                                # than lower bound due to the conversion above.
                                if theta_max >= theta_min:
                                    mini_vis_priors.append(
                                        (theta_min, theta_max)
                                    )
                                else:
                                    mini_vis_priors.append(
                                        (theta_max, theta_min)
                                    )
                        else:  # Is list.
                            if theta_max is not None and theta_min is not None:
                                # Check because upper bound may now be less
                                # than lower bound due to the conversion above.
                                if theta_max >= theta_min:
                                    mini_vis_priors.append(
                                        [theta_min, theta_max]
                                    )
                                else:
                                    mini_vis_priors.append(
                                        [theta_max, theta_min]
                                    )
                else:
                    mini_vis_priors.append([None, None])
            vis_priors.append(mini_vis_priors)
        else:  # Nothing to convert.
            vis_priors.append([[None, None]] * 6)

    # Convert initial guesses, if given, to proper units.
    # Apply default initial guesses if not given.
    rad_ra_guess = []
    rad_dec_guess = []
    rad_width_guess = []
    rad_theta_guess = []
    if flux_guess is None:
        flux_guess = [None] * n_sources
    if ra_guess is None:
        ra_guess = [None] * n_sources
    if dec_guess is None:
        dec_guess = [None] * n_sources
    if width_guess is None:
        width_guess = [None] * n_sources
    if ratio_guess is None:
        ratio_guess = [None] * n_sources
    if angle_guess is None:
        angle_guess = [None] * n_sources

    # Apply default guesses if needed.
    for i in range(n_sources):
        if flux_guess[i] is None:
            flux_guess[i] = 1
        if ra_guess[i] is None:
            rad_ra_guess.append(0)
        else:
            rad_ra_guess.append(
                Angle(ra_guess[i], units.arcsec).to(units.radian).value
            )
        if dec_guess[i] is None:
            rad_dec_guess.append(0)
        else:
            rad_dec_guess.append(
                Angle(dec_guess[i], units.arcsec).to(units.radian).value
            )

        if sources[i] != 'p':
            if width_guess[i] is None:
                rad_width_guess.append(rad_bmaj)
            # Convert width from arcsec to radians, if needed.
            elif width_guess[i] < 0:
                raise ValueError("Width initial guesses may not be negative.")
            else:
                rad_width_guess.append(
                    float(
                        Angle(width_guess[i], units.arcsec)
                        .to(units.radian).value
                    )
                )
        else:
            rad_width_guess.append(None)
        if sources[i] not in ['p', 'c']:
            if ratio_guess[i] is None:
                ratio_guess[i] = 1
            # Ensure ratio is in the correct range.
            elif ratio_guess[i] <= 0 or ratio_guess[i] > 1:
                raise ValueError(
                    "If not None, ratio initial guesses must be greater than "
                    "0 and less than or equal to 1."
                )
            if angle_guess[i] is None:
                rad_theta_guess.append(-np.pi / 2)  # 0 in image domain.
            elif angle_guess[i] < -90 or angle_guess[i] > 90:
                raise ValueError(
                    "Position angle initial guess must be between -90 and 90 "
                    "degrees, inclusive."
                )
            else:
                # Convert position angle from degrees to radians, if needed.
                # Shift by 90 degrees to go from image to visibility angle.
                temp_pa = angle_guess[i] - 90
                if temp_pa < -90:
                    temp_pa += 180  # -90 to -180 is the same as 90 to 0.
                rad_theta_guess.append(temp_pa * np.pi/180)
        else:
            rad_theta_guess.append(None)

    # Extract peak data from info, if it exists and if needed.
    peak_info = None
    if info:
        if use_image_guess:
            peak_info = info['int_peak_val']
            ext_peaks = info['ext_peak_val']
            coord_info = info['int_peak_coord']
            ext_coords = info['ext_peak_coord']
            if type(ext_peaks) is list:
                peak_info += ext_peaks
                coord_info += ext_coords
            n_peaks = len(peak_info)
            smallest_peak = np.min(peak_info)
            # Use image RA and dec guesses for the sources where we have image
            # domain results.
            if i in range(n_peaks):
                rad_ra_guess[i] = Angle(
                    coord_info[i][0], units.arcsec
                ).to(units.radian).value
                rad_dec_guess[i] = Angle(
                    coord_info[i][1], units.arcsec
                ).to(units.radian).value
            if n_peaks < n_sources:
                warnings.warn(
                    f"Number of detected peaks ({n_peaks}) from the image is "
                    f"less than the number of sources to fit ({n_sources}). "
                    "Fitting will continue but be aware."
                )
    elif use_image_guess:
        warnings.warn(
            "User requested to use image domain results as initial "
            "guesses. However, image results (`info`) was not provided. "
            "Fitting will continue with user-inputted guesses, if they "
            "exist, or with default guesses."
        )

    if min_sep is None:
        min_sep = rad_bmaj / 10

    # Find all possible permutations.
    sample_space = list(SOURCE_TYPES.keys()) * n_sources
    all_permutations = list(itertools.permutations(sample_space, n_sources))

    for i in range(n_sources):
        if sources[i] != 'any':
            all_permutations = [
                p for p in all_permutations if p[i] == sources[i]
            ]  # Remove unwanted permutations.
    all_permutations = list(set(all_permutations))  # Remove duplicates.

    all_results = []
    for permutation in all_permutations:
        # Calculate n_params and n_walkers.
        n_params = 0
        for i in range(n_sources):
            source = permutation[i]
            n_params += SOURCE_TYPES[source][0]
        n_walkers = 2 * n_params

        # Initial guesses.
        for i in range(n_sources):
            source = permutation[i]
            flux_p0 = flux_guess[i]
            ra_p0 = rad_ra_guess[i]
            dec_p0 = rad_dec_guess[i]
            width_p0 = rad_width_guess[i]
            ratio_p0 = ratio_guess[i]
            theta_p0 = rad_theta_guess[i]
            rad_coord = (ra_p0, dec_p0)
            if peak_info is not None:
                # Use image domain guesses instead for flux. (RA and dec have
                # already been handled above.)
                if i < n_peaks:
                    flux_p0 = SOURCE_TYPES[source][5](
                        peak_info[i],
                        width_p0,
                        ratio_p0,
                        rad_barea,
                    )
                else:
                    flux_p0 = SOURCE_TYPES[source][5](
                        smallest_peak,
                        width_p0,
                        ratio_p0,
                        rad_barea,
                    )
            if i == 0:
                p0 = SOURCE_TYPES[source][1](
                    flux_p0,
                    rad_coord,
                    rad_bmaj,
                    width_p0,
                    ratio_p0,
                    theta_p0,
                    n_walkers,
                )
            else:
                mini_p0 = SOURCE_TYPES[source][1](
                    flux_p0,
                    rad_coord,
                    rad_bmaj,
                    width_p0,
                    ratio_p0,
                    theta_p0,
                    n_walkers,
                )
                # If the number of sources to be fit exceeds the number of
                # sources found in the image domain and image domain results
                # are being used as initial guesses, allow the initial
                # guess(es) for the coordinates of the remaining source(s) to
                # come a square of side lengths rad_bmaj, centered around the
                # origin. (After the else statement because we are guaranteed
                # to have at least 1 source found in image.)
                if peak_info is not None and i >= n_peaks:
                    for j in range(n_walkers):
                        mini_p0[j,1] = np.random.uniform(
                            -rad_bmaj / 2, rad_bmaj / 2
                        )
                        mini_p0[j,2] = np.random.uniform(
                            -rad_bmaj / 2, rad_bmaj / 2
                        )
                p0 = np.append(p0, mini_p0, axis=1)

        # Set up and run MCMC.
        n_steps = 100
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_params,
            log_probability,
            args=(
                permutation,
                vis_priors,
                re,
                im,
                u,
                v,
                w,
                min_sep,
            )
        )
        try:
            state = sampler.run_mcmc(p0, n_steps)
        except emcee.autocorr.AutocorrError:
            pass
        except ValueError:
            print(
                "Error encountered during MCMC run for permutation "
                f"{permutation}. Skipping this permutation."
            )
            all_results.append({
                'permutation': permutation,
                'n_params': n_params,
                'result': None,
                'chi2': np.inf,
                'chain': None
            })
            continue
        tau = sampler.get_autocorr_time(quiet=True)
        if np.isnan(tau).all():
            warnings.warn(
                f"Autocorrelation time for first run of {permutation} could "
                f"not be estimated; all values are NaN.",
                RuntimeWarning
            )
            all_results.append({
                'permutation': permutation,
                'n_params': n_params,
                'result': None,
                'chi2': np.inf,
                'chain': None
            })
            continue
        int_tau = math.ceil(np.nanmax(tau))
        steps_to_50_tau = abs(int_tau * 50 - n_steps)
        try:
            sampler.run_mcmc(state, steps_to_50_tau)
        except ValueError:
            print(
                "Error encountered during second MCMC run for permutation "
                f"{permutation}. Skipping this permutation."
            )
            all_results.append({
                'permutation': permutation,
                'n_params': n_params,
                'result': None,
                'chi2': np.inf,
                'chain': None
            })
            continue
        chain = sampler.get_chain(discard = int_tau * 10, flat=True)
        log_probs = sampler.get_log_prob(discard = int_tau * 10, flat=True)
        max_prob_index = np.argmax(log_probs)

        # Find parameter estimates and uncertainties and calculate chi2.
        result = {}
        model = 0.0
        start = 0
        for i in range(n_sources):
            source = permutation[i]
            n_source_params = SOURCE_TYPES[source][0]
            source_chain = chain[:, start:start+n_source_params]
            source_result = {'type': source}
            temp_medians = []  # To store medians.
            # To store best values (that maximimize probability).
            temp_bests = {}
            temp_max_probs = [] # To store candidate maximum probabilities.
            for j in range(n_source_params):
                samples = source_chain[:, j]
                temp_max_probs.append(samples[max_prob_index])
                samples_med = np.median(samples)
                samples_sd = np.nanstd(samples)
                param_name = SOURCE_TYPES[source][4][j]
                source_result[param_name] = (
                    float(samples_med),
                    float(samples_sd)
                )
                temp_bests[param_name] = float(samples[max_prob_index])
                temp_medians.append(samples_med)
            source_result['best'] = temp_bests
            model += SOURCE_TYPES[source][3](
                temp_max_probs,
                u,
                v,
                rad_bmaj,
                rad_barea,
            )
            result[f'source_{i+1}'] = source_result
            start += n_source_params
        chi2 = float(np.sum(w * ((re - model.real)**2 + (im - model.imag)**2)))
        all_results.append({
            'permutation': permutation,
            'n_params': n_params,
            'result': result,
            'chi2': chi2,
            'chain': chain
        })

    # Do it again with refined initial guesses, if requested.
    for reps in range(additional_runs):
        second_results = []
        for permutation_info in all_results:
            # Calculate n_params and n_walkers.
            permutation = permutation_info['permutation']
            chain = permutation_info['chain']
            n_params = 0
            for i in range(n_sources):
                source = permutation[i]
                n_params += SOURCE_TYPES[source][0]
            n_walkers = 2 * n_params

            # New initial guesses from previous results.
            for i in range(n_sources):
                source = permutation[i]
                # If fitting didn't happen, we can't actually refine our
                # guesses, so we keep inputted guesses and possibly image
                # guesses since we have nothing better right now.
                if permutation_info['result'] is None:
                    flux_p1 = flux_guess[i]
                    ra_p1 = rad_ra_guess[i]
                    dec_p1 = rad_dec_guess[i]
                    width_p1 = rad_width_guess[i]
                    ratio_p1 = ratio_guess[i]
                    theta_p1 = rad_theta_guess[i]
                    rad_coord = (ra_p1, dec_p1)
                    if peak_info is not None:
                        # Use image domain guesses instead for flux. (RA and
                        # dec have already been handled.)
                        if i < n_peaks:
                            flux_p1 = SOURCE_TYPES[source][5](
                                peak_info[i],
                                width_p1,
                                ratio_p1,
                                rad_barea,
                            )
                        else:
                            flux_p1 = SOURCE_TYPES[source][5](
                                smallest_peak,
                                width_p1,
                                ratio_p1,
                                rad_barea,
                            )
                    if i == 0:
                        p1 = SOURCE_TYPES[source][1](
                            flux_p1,
                            rad_coord,
                            rad_bmaj,
                            width_p1,
                            ratio_p1,
                            theta_p1,
                            n_walkers,
                        )
                    else:
                        mini_p1 = SOURCE_TYPES[source][1](
                            flux_p1,
                            rad_coord,
                            rad_bmaj,
                            width_p1,
                            ratio_p1,
                            theta_p1,
                            n_walkers,
                        )
                        # If the number of sources to be fit exceeds the number
                        # of sources found in the image domain and image domain
                        # results are being used as initial guesses, allow the
                        # initial guess(es) for the coordinates of the
                        # remaining source(s) to come a square of side lengths
                        # rad_bmaj, centered around the origin. (After the else
                        # statement because we are guaranteed to have at least
                        # 1 source found in image.)
                        if peak_info is not None and i >= n_peaks:
                            for j in range(n_walkers):
                                mini_p1[j,1] = np.random.uniform(
                                    -rad_bmaj / 2, rad_bmaj / 2
                                )
                                mini_p1[j,2] = np.random.uniform(
                                    -rad_bmaj / 2, rad_bmaj / 2
                                )
                        p1 = np.append(p1, mini_p1, axis=1)
                else:
                    source_result = permutation_info['result'][f'source_{i+1}']
                    best_params = []
                    med_sd = []
                    for param_name in SOURCE_TYPES[source][4]:
                        med_sd.append(source_result[param_name])
                        best_params.append(source_result[param_name][0])

                    if i == 0:
                        p1 = all_p1(med_sd, n_walkers, chain)
                    else:
                        p1 = np.append(
                            p1, all_p1(med_sd, n_walkers, chain), axis=1
                        )

            # Set up and run MCMC.
            n_steps = 100
            sampler1 = emcee.EnsembleSampler(
                n_walkers,
                n_params,
                log_probability,
                args=(
                    permutation,
                    vis_priors,
                    re,
                    im,
                    u,
                    v,
                    w,
                    min_sep,
                )
            )
            try:
                state = sampler1.run_mcmc(p1, n_steps)
            except emcee.autocorr.AutocorrError:
                pass
            except ValueError:
                print(
                    f"Error encountered during MCMC run number {reps+2} for "
                    f"permutation {permutation}. Skipping this permutation."
                )
                second_results.append({
                    'permutation': permutation,
                    'n_params': n_params,
                    'result': None,
                    'chi2': np.inf,
                    'chain': None
                })
                continue
            tau = sampler1.get_autocorr_time(quiet=True)
            if np.isnan(tau).all():
                warnings.warn(
                    f"Autocorrelation time for MCMC run number {reps+2} of "
                    f"{permutation} could not be estimated; all values are "
                    "NaN.", RuntimeWarning
                )
                second_results.append({
                    'permutation': permutation,
                    'n_params': n_params,
                    'result': None,
                    'chi2': np.inf,
                    'chain': None
                })
                continue
            int_tau = math.ceil(np.nanmax(tau))
            steps_to_50_tau = abs(int_tau * 50 - n_steps)
            try:
                sampler1.run_mcmc(state, steps_to_50_tau)
            except ValueError:
                print(
                    f"Error encountered during MCMC run number {reps+2} for "
                    f"permutation {permutation}. Skipping this permutation."
                )
                second_results.append({
                    'permutation': permutation,
                    'n_params': n_params,
                    'result': None,
                    'chi2': np.inf,
                    'chain': None
                })
                continue
            chain1 = sampler1.get_chain(discard = int_tau * 10, flat=True)
            log_probs1 = sampler1.get_log_prob(
                discard = int_tau * 10,
                flat=True,
            )
            max_prob_index1 = np.argmax(log_probs1)

            # Find parameter estimates and uncertainties and calculate chi2.
            result = {}
            model = 0.0
            start = 0
            for i in range(n_sources):
                source = permutation[i]
                n_source_params = SOURCE_TYPES[source][0]
                source_chain = chain1[:, start:start+n_source_params]
                source_result = {'type': source}
                temp_medians = []  # To store medians for chi2 calculation.
                # To store best values (that maximimize probability).
                temp_bests = {}
                # To store candidate maximum probabilities.
                temp_max_probs = []
                for j in range(n_source_params):
                    samples = source_chain[:, j]
                    temp_max_probs.append(samples[max_prob_index1])
                    samples_med = np.median(samples)
                    samples_sd = np.nanstd(samples)
                    param_name = SOURCE_TYPES[source][4][j]
                    source_result[param_name] = (
                        float(samples_med),
                        float(samples_sd)
                    )
                    temp_bests[param_name] = float(samples[max_prob_index1])
                    temp_medians.append(samples_med)
                source_result['best'] = temp_bests
                model += SOURCE_TYPES[source][3](
                    temp_max_probs,
                    u,
                    v,
                    rad_bmaj,
                    rad_barea,
                )
                result[f'source_{i+1}'] = source_result
                start += n_source_params
            chi2 = float(
                np.sum(w * ((re - model.real)**2 + (im - model.imag)**2))
            )
            second_results.append({
                'permutation': permutation,
                'n_params': n_params,
                'result': result,
                'chi2': chi2,
                'chain': chain1
            })
        all_results = []
        for permutation_info in second_results:
            all_results.append(permutation_info)

    # Bayesian Information Criterion and reduced chi2.
    n = len(re)
    for permutation_info in all_results:
        k = permutation_info['n_params']
        chi2 = permutation_info['chi2']
        if chi2 is np.inf:
            permutation_info['red_chi2'] = np.inf
            permutation_info['bic'] = np.inf
            continue
        bic = k * np.log(n) + chi2
        red_chi2 = chi2 / (n-k)
        permutation_info['bic'] = float(bic)
        permutation_info['red_chi2'] = red_chi2
    all_results.sort(key=lambda x: x['bic'])  # Lowest to highest BIC.

    if extreme_case_check:
        # Ue reduced chi2 of lowest BIC model to estimate how well fitting
        # occurred.
        # Do this inside the if loop to silence the high reduced chi2 warning
        # when checking the extreme cases.
        if all_results[0]['red_chi2'] > 10:
            warnings.warn(
                "Best model has a reduced chi2 > 10. The fit may have been "
                "poor. Use these results with caution. Consider re-running "
                "with different inputted guesses."
            )

        # Case: brightest source is very resolved.
        large_width = float(
            Angle(2 * rad_bmaj, units.radian).to(units.arcsec).value
        )  # Guess 2x beam major axis for a very resolved source.
        amplitudes = np.sqrt(re**2 + im**2)
        # Use median amplitude as a flux guess for the very resolved source
        # case.
        median_amp = np.nanmedian(amplitudes)
        large_width_guess = [large_width] + width_guess[1:]

        # Keep ratio guesses if they exist.
        new_ratio_guess = ratio_guess
        if ratio_guess[0] is None:
            new_ratio_guess[0] = 1  # Use default of 1 if no ratio guess.

        # Keep angle guesses if they exist.
        new_angle_guess = angle_guess
        if angle_guess[0] is None:
            new_angle_guess[0] = -np.pi / 2

        new_priors = [pr for pr in priors]
        # Get rid of any prior on the width of this source, if it exists.
        if new_priors[0] is not None and new_priors[0][3] is not None:
            new_priors[0][3] = None

        # Subcase: very resolved Gaussian.
        large_g_flux_guess = [flux for flux in flux_guess]
        large_g_flux_guess[0] = median_amp
        large_g_sources = ['g'] + sources[1:]
        large_gaussian_result = uv_fit_no_image(
            vis=vis,
            info=info,
            sources=large_g_sources,
            flux_guess=large_g_flux_guess,
            ra_guess=ra_guess,
            dec_guess=dec_guess,
            width_guess=large_width_guess,
            ratio_guess=new_ratio_guess,
            angle_guess=new_angle_guess,
            priors=new_priors,
            clean_output=True,
            corner_plot=False,
            additional_runs=additional_runs,
            extreme_case_check=False,
            use_image_guess=False,
        )

        # Subcase: very resolved disk.
        large_d_flux_guess = [flux for flux in flux_guess]
        large_d_flux_guess[0] = median_amp
        large_d_sources = ['d'] + sources [1:]
        large_disk_result = uv_fit_no_image(
            vis=vis,
            info=info,
            sources=large_d_sources,
            flux_guess=large_d_flux_guess,
            ra_guess=ra_guess,
            dec_guess=dec_guess,
            width_guess=large_width_guess,
            ratio_guess=new_ratio_guess,
            angle_guess=new_angle_guess,
            priors=new_priors,
            clean_output=True,
            corner_plot=False,
            additional_runs=additional_runs,
            extreme_case_check=False,
            use_image_guess=False,
        )

        # Case: brightest source is a point source.
        point_result = None
        # Make sure we haven't already tested this.
        if not (sources[0] in ['p', 'any']):
            p_sources = ['p'] + sources[1:]
            point_result = uv_fit_no_image(
                vis=vis,
                info=info,
                sources=p_sources,
                flux_guess=flux_guess,
                ra_guess=ra_guess,
                dec_guess=dec_guess,
                width_guess=width_guess,
                ratio_guess=ratio_guess,
                angle_guess=angle_guess,
                priors=priors,
                clean_output=True,
                corner_plot=False,
                additional_runs=additional_runs,
                extreme_case_check=False,
                use_image_guess=False,
            )

        # Compare BICs of these extreme cases to BICs of the other results.
        best_other_bic = all_results[0]['bic']
        extreme_cases = [
            {
                'info': large_gaussian_result[0],
                'case': 'very large gaussian'
            },
            {
                'info': large_disk_result[0],
                'case': 'very large disk'
            }
        ]
        if point_result is not None:
            extreme_cases.append({
                'info': point_result[0],
                'case': 'point source'
            })
        extreme_cases.sort(key=lambda x: x['info']['bic'])
        best_extreme = extreme_cases[0]
        if best_extreme['info']['bic'] < best_other_bic:
            best_extreme_case = best_extreme['case']
            warnings.warn(
                f"The brightest source might be a {best_extreme_case}. "
                "Consider re-running with edited inputted guesses, especially "
                "if the fit was poor."
            )

    if clean_output:
        for permutation_info in all_results:
            permutation_info['chi2'] = round(
                permutation_info['chi2'], 2
            )
            permutation_info['bic'] = round(
                permutation_info['bic'], 2
            )
            permutation_info['red_chi2'] = round(
                permutation_info['red_chi2'], 2
            )
            result = permutation_info['result']
            start = 0
            permutation_chain = permutation_info['chain']
            if result is None:
                continue
            for i in range(n_sources):
                source_key = f'source_{i+1}'
                source_result = result[source_key]
                source_type = source_result['type']
                source_params = SOURCE_TYPES[source_type][4]
                n_source_params = SOURCE_TYPES[source_type][0]
                n_walkers = 2 * n_source_params
                source_chain = permutation_chain[
                    :,
                    start:start + n_source_params
                ]

                # Flux.
                flux_chain = source_chain[:, 0]
                flux_sigmas = tuple([
                    float(sigfig.round(sigma, sigfigs=3))
                    for sigma in sigmas(flux_chain)
                ])
                source_result['flux'] = (
                    round_tuple(
                        (source_result['flux'][0], source_result['flux'][1])
                    ), flux_sigmas
                )

                # Convert ra, dec to arcsec.
                ra_chain = source_chain[:, 1]
                dec_chain = source_chain[:, 2]
                ra_sigmas = tuple([
                    float(
                        sigfig.round(
                            Angle(sigma, units.radian).to(units.arcsec).value,
                            sigfigs=3
                        )
                    )
                    for sigma in sigmas(ra_chain)
                ])
                dec_sigmas = tuple([
                    float(
                        sigfig.round(
                            Angle(sigma, units.radian).to(units.arcsec).value,
                            sigfigs=3
                        )
                    )
                    for sigma in sigmas(dec_chain)
                ])
                source_result['ra'] = (
                    round_tuple(
                        tuple([
                            float(
                                Angle(l, units.radian).to(units.arcsec).value
                            )
                            for l in source_result['ra']
                        ])
                    ),
                    ra_sigmas
                )
                source_result['dec'] = (
                    round_tuple(
                        tuple([
                            float(
                                Angle(m, units.radian).to(units.arcsec).value
                            )
                        for m in source_result['dec']
                        ])
                    ),
                    dec_sigmas
                )

                # Convert visibility width to image width in arcsec.
                if source_type != 'p':
                    width_chain = source_chain[:, 3]
                    width_sigmas = tuple([
                        float(
                            sigfig.round(
                                Angle(sigma, units.radian)
                                .to(units.arcsec).value,
                                sigfigs=3
                            )
                        )
                        for sigma in sigmas(width_chain)
                    ])
                    source_result['sigma'] = (
                        round_tuple(
                            (
                                float(
                                    Angle(
                                        source_result[source_params[3]][0],
                                        units.radian
                                    ).to(units.arcsec).value
                                ),
                                float(
                                    Angle(
                                        source_result[source_params[3]][1],
                                        units.radian
                                    ).to(units.arcsec).value
                                )
                            )
                        ),
                        width_sigmas
                    )
                    if source_type == 'd':
                        del source_result['r']

                # Convert visibility angle to image angle in degrees.
                # Convert sigma and ratio into major and minor axes.
                if source_type in ['g', 'd']:
                    theta_chain = source_chain[:, 5]
                    vis_theta_sigmas = sigmas(theta_chain)
                    theta_sigmas = [
                        theta * 180/np.pi + 90 for theta in vis_theta_sigmas
                    ]
                    uvis_theta = ufloat(
                        source_result['vis_theta'][0],
                        source_result['vis_theta'][1]
                    )
                    uimg_theta = (uvis_theta * (180/np.pi))
                    # Convert visibility angle to image angle.
                    modded_theta = uimg_theta.n + 90
                    # Put angle back in [-90,90] range, if needed.
                    if modded_theta > 90:
                        modded_theta -= 180
                    for i in range(len(theta_sigmas)):
                        if theta_sigmas[i] > 90:
                            theta_sigmas[i] -= 180
                        theta_sigmas[i] = float(
                            sigfig.round(theta_sigmas[i], sigfigs=3)
                        )
                    del source_result['vis_theta']
                    source_result['theta'] = (
                        round_tuple((modded_theta, uimg_theta.s)),
                        theta_sigmas
                    )

                    uwidth_maj = ufloat(
                        source_result['sigma'][0][0],
                        source_result['sigma'][0][1]
                    )
                    uratio = ufloat(
                        source_result['ratio'][0],
                        source_result['ratio'][1]
                    )
                    uwidth_min = uwidth_maj * uratio
                    del source_result['sigma']
                    del source_result['ratio']
                    source_result['major_axis'] = (
                        round_tuple((uwidth_maj.n, uwidth_maj.s)),
                        tuple([
                            float(sigfig.round(width, sigfigs=3))
                            for width in width_sigmas
                        ])
                    )
                    source_result['minor_axis'] = (
                        round_tuple((uwidth_min.n, uwidth_min.s)),
                        tuple([
                            float(sigfig.round(width * uratio.n, sigfigs=3))
                            for width in width_sigmas
                        ])
                    )

                del source_result['best']

                start += n_source_params

    if corner_plot:
        for j in range(len(all_results)):
            permutation_info = all_results[j]
            result = permutation_info['result']
            if result is None:
                continue
            chain = permutation_info['chain']
            start = 0
            end = 0
            for i in range(n_sources):
                source_key = f'source_{i+1}'
                source_result = result[source_key]
                source_type = source_result['type']
                n_params = SOURCE_TYPES[source_type][0]
                source_params = SOURCE_TYPES[source_type][4]
                end += n_params
                if i == n_sources-1:
                    fig = corner.corner(
                        chain[:, start:],
                        labels=source_params
                    )
                else:
                    fig = corner.corner(
                        chain[:, start:end],
                        labels=source_params
                    )
                fig.suptitle(
                    f"Permutation {j+1}: {permutation_info['permutation']}, "
                    f"source {i+1} of {n_sources}"
                )
                start = end

    for permutation_info in all_results:
        del permutation_info['chain']

    return all_results


def uv_fit_with_image(
    fits_file: str | Path,
    sources: list,
    flux_guess: list | None = None,
    ra_guess: list | None = None,
    dec_guess: list | None = None,
    width_guess: list | None = None,
    ratio_guess: list | None = None,
    angle_guess: list | None = None,
    priors: list | None = None,
    clean_output: bool = True,
    corner_plot: bool = True,
    additional_runs: int = 2,
    extreme_case_check: bool = True,
    min_sep: float | None = None,
    use_image_guess: bool = True,
) -> list:
    """
    Fit visibility data from a FITS file to a model of a given source type or
    given source types.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the visibility data as the
        index 1 element.
    sources : list
        List of source types to fit. Each source type should be one of 'p'
        (point), 'c' (circular gaussian), 'g' (gaussian), 'd' (disk), or 'any'
        (try all and pick best fit).
    flux_guess : list | None, optional
        A guess for the flux(es) of the source(s), in Jy.
        If `flux_guess` is a list, each element is a guess for the
        corresponding source in `sources`, so the list must be the same length
        as `sources`. Elements in the list may be `None`.
    ra_guess : list | None, optional
        A guess for the Right Ascension(s) of the source(s), in arcseconds
        relative to the field center.
        If `ra_guess` is a list, each element is a guess for the corresponding
        source in `sources`, so the list must be the same length as `sources`.
        Elements in the list may be `None`.
    dec_guess : list | None, optional
        A guess for the Declination(s) of the source(s), in arcseconds relative
        to the field center.
        If `dec_guess` is a list, each element is a guess for the corresponding
        source in `sources`, so the list must be the same length as `sources`.
        Elements in the list may be `None`.
    width_guess : list | None, optional
        A guess for the width(s) of the source(s), in arcseconds.
        If `width_guess` is a list, each element is a guess for the
        corresponding source in `sources`, so the list must be the same length
        as `sources`. Elements in the list may be `None`, but a width guess is
        required for every source type except `p`.
    ratio_guess : list | None, optional
        A guess for the source's/sources' minor axis to major axis ratio(s).
        If `ratio_guess` is a list, each element is a guess for the
        corresponding source in `sources`, so the list must be the same length
        as `sources`. Elements in the list may be `None`, but a ratio guess is
        required for every source type except `p` and `c`.
    angle_guess : list | None, optional
        A guess for the source angle(s), in degrees.
        If `angle_guess` is a list, each element is a guess for the
        corresponding source in `sources`, so the list must be the same length
        as `sources`. Elements in the list may be `None`, but an angle guess is
        required for every source type except `p` and `c`.
    priors : list | None, optional
        Priors for the source(s).
        If `priors` is a list, each element may be `None` or a list of priors
        for the 6 parameters (flux, ra, dec, width, ratio, angle) for the
        corresponding source in `sources`. If `priors` is a list, it may
        either be of length 1, in which case the set of priors will be applied
        to all sources, or it must be the same length as `sources`. The list of
        priors must contain 6 elements (one for each parameter), that are
        either tuples (`min`, `max`) exclusive or lists [`min`, `max`]
        inclusive.
    clean_output : bool, optional
        Whether to return the list of fit results in a more human-parsable
        format. See Returns section for more details.
    additional_runs : int, optional
        The number of MCMC runs beyond the first to perform.
    extreme_case_check : bool, optional
        Whether to check for extreme cases: the brightest source is a very
        large Gaussian, a very large disk, or a point source.
    min_sep : float | None, optional
        The minimum separation, in radians, to differentiate two distinct
        sources.
    use_image_guess : bool, optional
        Whether to use the image domain fitting results as initial guesses for
        the visibility fitting.

    Returns
    -------
    list of dict
        A list of a dictionary/dictionaries with the following keys:
        `permutation` : tuple of str
            Tuple of source types corresponding to the model being fit.
        `n_params` : int
            The total number of parameters in the model.
        `result` : dict
            A dictionary with the following keys:
            `source_m` : dict
                A dictionary, for each source (m = 1 through m = `n_sources`),
                with the following keys:
                    `type` : str
                        The source type, as a single character code (a key of
                        SOURCE_TYPES).
                    `flux` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted flux estimate and its
                        uncertainty, respectively, in Jy. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sampled
                        parameters, in Jy, all rounded to 3 signficant figures.
                        Else, a tuple of the fitted flux estimate and its
                        uncertainty, respectively, in Jy.
                    `ra` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Right Ascension
                        estimate and its uncertainty, respectively, in
                        arcseconds relative to the field center. The
                        uncertainty is rounded to 3 significant figures and the
                        estimate is rounded to the appropriate number of digits
                        to match the uncertainty. The second tuple contains
                        the 2.5th, 16th, 50th, 84th, and 97.5th percentile of
                        sample parameters, in relative arcseconds, all rounded
                        to 3 significant figures.
                        Else, a tuple of the fitted Right Ascension estimate
                        and its uncertainty, respectively, in radians relative
                        to the field center.
                    `dec` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Declination estimate
                        and its uncertainty, respectively, in arcseconds
                        relative to the field center. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sample
                        parameters, in relative arcseconds, all rounded to 3
                        significant figures.
                        Else, a tuple of the fitted Declination estimate and
                        its uncertainty, respectively, in radians relative to
                        the field center.
                    `sigma` : tuple of tuple | tuple of float
                        This key is only present for sources of `type` 'c'.
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted circular Gaussian
                        standard deviation estimate and its uncertainty,
                        respectively, in arcseconds. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sample
                        parameters, in arcseconds, all rounded to 3 significant
                        figures.
                        Else, a tuple of the fitted circular Gaussian standard
                        deviation estimate and its uncertainty, respectively,
                        in radians.
                    `r` : tuple of float
                        This key is only present when `clean_output` is `False`
                        and the source is of `type` 'd'.
                        A tuple of the fitted disk radius estimate and its
                        uncertainty, respectively, in radians.
                    `theta` : tuple of tuple
                        This key is only present when `clean_output` is `True`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of 2 tuples. The first tuple contains the
                        fitted source angle estimate and its uncertainty,
                        respectively, in degrees. The uncertainty is rounded to
                        3 significant figures and the estimate is rounded to
                        the appropriate number of digits to match the
                        uncertainty. The second tuple contains the 2.5th, 16th,
                        50th, 84th, and 97.5th percentile of sample parameters,
                        in degrees, all rounded to 3 significant figures.
                    `vis_theta` : tuple of float
                        This key is only present when `clean_output` is `False`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of the fitted source angle estimate in
                        visibility space and its uncertainty, respectively, in
                        radians.
                    `ratio` : tuple of float
                        This key is only present when `clean_output` is `False`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of the fitted source minor axis to major axis
                        ratio and its uncertainty, respectively.
                    `major_axis` : tuple of tuple
                        This key is only present when `clean_output` is `True`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of 2 tuples. The first tuple contains the
                        fitted source major axis estimate and its uncertainty,
                        respectively, in arcseconds. The uncertainty is rounded
                        to 3 significant figures and the estimate is rounded to
                        the appropriate number of digits to match the
                        uncertainty. The second tuple contains the 2.5th, 16th,
                        50th, 84th, and 97.5th percentile of sample parameters,
                        in arcseconds, all rounded to 3 significant figures.
                    `minor_axis : tuple of tuple
                        This key is only present when `clean_output` is `True`
                        and the source is of `type` 'g' or 'd'.
                        A tuple of 2 tuples. The first tuple contains the
                        fitted source minor axis estimate and its uncertainty,
                        respectively, in arcseconds. The uncertainty is rounded
                        to 3 significant figures and the estimate is rounded to
                        the appropriate number of digits to match the
                        uncertainty. The second tuple contains the 2.5th, 16th,
                        50th, 84th, and 97.5th percentile of sample parameters,
                        in arcseconds, all rounded to 3 significant figures.
                    `best` : dict
                        This key is only present if `clean_output` is `False`.
                        A dictionary, of the collection of sample parameters
                        that resulted in the lowest chi-squared value, with the
                        following keys:
                            `flux` : float
                                The flux, in Jy.
                            `ra` : float
                                The right ascension, in radians relative to the
                                field center.
                            `dec` : float
                                The declination, in radians relative to the
                                field center.
                            `r` : float
                                This key is only present if the source is of
                                `type` 'd'. The disk radius, in radians.
                            `sigma` : float
                                This key is only present if the source is of
                                `type` 'c' or 'g'. The source standard
                                deviation, in radians.
                            `ratio` : float
                                This key is only present if the source is of
                                `type` 'g' or 'd'. The ratio between the source
                                minor axis and major axis.
                            `vis_theta` : float
                                This key is only present if the source is of
                                `type` 'g' or 'd'. The source angle in
                                visibility space, in radians.

        `chi2` : float
            The chi-squared value of the fit.
            If `clean_output` is `True`, the chi-squared value is rounded to
            the hundredths.
        `bic` : float
            The Bayesian Information Criterion (BIC), estimated as
            chi^2 + k * ln(n), where chi^2 is the chi-squared value of the
            fit, k is the number of parameters, and n is the number of data
            points to which the model is being fit.
            If `clean_output` is `True`, the BIC is rounded to the hundredths.
        `red_chi2` : float
            The reduced chi-squared value of the fit.
            If `clean_output` is `True`, the reduced chi-squared value is
            rounded to the hundredths.

    Raises
    ------
    ValueError
        If `additional_runs` is negative.
        If `priors` has improper formatting.
        If no sources are specified.
        If improper source types are specified.
        If required initial guesses are missing.
        If initial guesses are outside of the proper bounds (width guesses are
        not positive, ratio guesses are not greater than 0 but less than or
        equal to 1, angle guesses are not between -90 and 90 degrees
        inclusive).

    Warns
    -----
    UserWarning
        If the number of peaks detected from the image fitting is less than the
        number of sources requested to fit.
        If the autocorrelation time for parameters of an MCMC run could not be
        estimated.
        If the best-fit model has a reduced chi-square value greater than 10.
        If `extreme_case_check` is true and the brightest source might be an
        extreme case (a very large Gaussian, a very large disk, or a point
        source).

    Notes
    -----
    The visibility table should contain as its elements  rows of 6 elements
    each: the 0th index corresponding to the frequency or frequency bin, the
    1st index corresponding to the `u` coordinate, in wavelengths; the 2nd
    index corresponding to the `v` coordinate, in wavelengths; the 3rd index
    corresponding to the real component of the visibility, in Jy; the 4th index
    corresponding to the imaginary component of the visibility, in Jy; and the
    5th index corresponding to the visibility weight (the inverse of the
    uncertainty squared), in 1/Jy^2.

    If the number of sources to fit exceeds the number of sources detected in
    the image domain, then the initial guess for the source location will
    be the a square around the origin, with side length equal to the beam major
    axis.

    Regardless of `clean_output`, corner plots will be of the visibility
    parameters. This means that the RA and Dec will be in radians, and the
    position angle will be in radians and rotated by pi/2 relative to the
    position angle in the image domain.

    This function employs Markov chain Monte Carlo (MCMC) via the emcee package
    to sample the parameter space.

    Additional MCMC runs use the fitting results from the previous run to
    create initial guesses.

    For the very large Gaussian and very large disk checks, the width initial
    guesses are created around twice the beam major axis. The flux initial
    guesses are created around the median visibility amplitude. If there is
    no ratio or angle guess, intial guesses will be created around 0.8 and
    around 0 radians.
    """
    # Extract data from FITS file.
    info = {}
    fits_file = Path(fits_file)
    with fits.open(fits_file) as file:
        info['CUNIT1'] = file[0].header['CUNIT1']
        vis = file[1].data
        info['BMAJ'] = file[0].header['BMAJ']
        info['BMIN'] = file[0].header['BMIN']
        # Account for different visiblity conventions.
        if file[1].header['TTYPE2'] == 'indexU':
            info['U_RES'] = file[1].header['U_RES']
        if file[1].header['TTYPE3'] == 'indexV':
            info['V_RES'] = file[1].header['V_RES']
        weighted_vis = False
        try:
            if file[1].header['EXTNAME'] == 'Weighted Visibilities':
                weighted_vis = True
        except KeyError:  # 'EXTNAME' is not a key.
            pass
        info['weighted_vis'] = weighted_vis

    # Get image domain fitting results.
    summ = summary(fits_file, plot=False)
    info['int_peak_val'] = summ['int_peak_val']
    info['int_peak_coord'] = summ['int_peak_coord']
    info['ext_peak_val'] = summ['ext_peak_val']
    info['ext_peak_coord'] = summ['ext_peak_coord']

    uv_fit_no_image(
        vis=vis,
        info=info,
        sources=sources,
        flux_guess=flux_guess,
        ra_guess=ra_guess,
        dec_guess=dec_guess,
        width_guess=width_guess,
        ratio_guess=ratio_guess,
        angle_guess=angle_guess,
        priors=priors,
        clean_output=clean_output,
        corner_plot=corner_plot,
        additional_runs=additional_runs,
        extreme_case_check=extreme_case_check,
        min_sep=min_sep,
        use_image_guess=use_image_guess,
    )


# TODO: default guesses
# TODO: fit fluxes and report fluxes
