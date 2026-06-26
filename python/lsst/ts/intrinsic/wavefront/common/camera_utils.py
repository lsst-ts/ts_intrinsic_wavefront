"""
common/camera_utils.py — Camera geometry utilities for Rubin Observatory.

Provides coordinate transforms between pixel and focal plane systems,
and detector layout information for LSSTCam.
"""

import numpy as np
from lsst.afw import cameraGeom


def pixel_to_focal(x, y, det):
    """Convert pixel coordinates to focal plane coordinates.

    Parameters
    ----------
    x, y : array-like
        Pixel coordinates.
    det : `lsst.afw.cameraGeom.Detector`
        Detector of interest.

    Returns
    -------
    fpx, fpy : `numpy.ndarray`
        Focal plane position in millimeters in DVCS (see LSE-349).
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def focal_to_pixel(fpx, fpy, det):
    """Convert focal plane coordinates to pixel coordinates.

    Parameters
    ----------
    fpx, fpy : array-like
        Focal plane position in millimeters in DVCS (see LSE-349).
    det : `lsst.afw.cameraGeom.Detector`
        Detector of interest.

    Returns
    -------
    x, y : `numpy.ndarray`
        Pixel coordinates.
    """
    tx = det.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
    x, y = tx.getMapping().applyForward(np.vstack((fpx, fpy)))
    return x.ravel(), y.ravel()


def get_detector_centers(camera):
    """Get focal plane center coordinates for all science detectors.

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera object (e.g. from LsstCam.getCamera()).

    Returns
    -------
    det_centers : dict
        {detector_name: (fpx_mm, fpy_mm)} for each science detector.
    """
    det_centers = {}
    for det in camera:
        if det.getType() != cameraGeom.DetectorType.SCIENCE:
            continue
        # Use center pixel of detector
        bbox = det.getBBox()
        cx = (bbox.getMinX() + bbox.getMaxX()) / 2.0
        cy = (bbox.getMinY() + bbox.getMaxY()) / 2.0
        fpx, fpy = pixel_to_focal(np.array([cx]), np.array([cy]), det)
        det_centers[det.getName()] = (float(fpx[0]), float(fpy[0]))
    return det_centers


# LSSTCam science raft grid (5x5, corners are wavefront/guider rafts)
SCIENCE_RAFTS = [
    'R01', 'R02', 'R03',
    'R10', 'R11', 'R12', 'R13', 'R14',
    'R20', 'R21', 'R22', 'R23', 'R24',
    'R30', 'R31', 'R32', 'R33', 'R34',
    'R41', 'R42', 'R43',
]

RAFT_GRID = [
    [None, 'R01', 'R02', 'R03', None],
    ['R10', 'R11', 'R12', 'R13', 'R14'],
    ['R20', 'R21', 'R22', 'R23', 'R24'],
    ['R30', 'R31', 'R32', 'R33', 'R34'],
    [None, 'R41', 'R42', 'R43', None],
]

SENSOR_SLOTS = ['S00', 'S01', 'S02', 'S10', 'S11', 'S12', 'S20', 'S21', 'S22']


def get_detector_name_map(camera):
    """Build name<->number mappings for all detectors.

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera object.

    Returns
    -------
    det_names : dict
        {detector_id: detector_name}
    det_nums : dict
        {detector_name: detector_id}
    """
    det_names = {i: det.getName() for i, det in enumerate(camera)}
    det_nums = {det.getName(): i for i, det in enumerate(camera)}
    return det_names, det_nums
