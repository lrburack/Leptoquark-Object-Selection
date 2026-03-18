import os
import numpy as np
import awkward as ak


def dr(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi  # wrap into [-π, π]
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)


def dr_ak(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi1[..., None] - phi2[..., None, :] + np.pi, 2 * np.pi) - np.pi
    deta = eta1[..., None] - eta2[..., None, :]
    return np.sqrt(deta**2 + dphi**2)

def invariant_mass(pt1, eta1, phi1, pt2, eta2, phi2, m1=0.0, m2=0.0):
    # components
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), pt1*np.sinh(eta1)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), pt2*np.sinh(eta2)
    E1 = np.sqrt(m1**2 + pt1**2*np.cosh(eta1)**2)
    E2 = np.sqrt(m2**2 + pt2**2*np.cosh(eta2)**2)
    m2_LQ = (m1**2 + m2**2 + 2*(E1*E2 - (px1*px2 + py1*py2 + pz1*pz2)))
    # guard against tiny negatives from roundoff
    return np.sqrt(np.maximum(m2_LQ, 0.0))

def invariant_mass_pairwise(pt1, eta1, phi1, pt2, eta2, phi2, m1=0.0, m2=0.0, invalid_sentinel=None):
    """
    Pairwise invariant mass between (pt1,eta1,phi1) with shape (n,m)
    and (pt2,eta2,phi2) with shape (n,k).
    Returns shape (n,m,k). If invalid_sentinel is not None (e.g., -1),
    outputs np.nan where any input of a pair equals the sentinel.
    """
    # Expand to (n,m,1) and (n,1,k)
    pt1e  = pt1[..., None]
    eta1e = eta1[..., None]
    phi1e = phi1[..., None]

    pt2e  = pt2[..., None, :]
    eta2e = eta2[..., None, :]
    phi2e = phi2[..., None, :]

    # Components
    px1, py1, pz1 = pt1e*np.cos(phi1e), pt1e*np.sin(phi1e), pt1e*np.sinh(eta1e)
    px2, py2, pz2 = pt2e*np.cos(phi2e), pt2e*np.sin(phi2e), pt2e*np.sinh(eta2e)

    E1 = np.sqrt(m1**2 + (pt1e*np.cosh(eta1e))**2)
    E2 = np.sqrt(m2**2 + (pt2e*np.cosh(eta2e))**2)

    m2_pair = (m1**2 + m2**2 + 2*(E1*E2 - (px1*px2 + py1*py2 + pz1*pz2)))
    out = np.sqrt(np.maximum(m2_pair, 0.0))

    # Optional masking of padded entries (e.g., sentinel = -1)
    if invalid_sentinel is not None:
        bad1 = (pt1 == invalid_sentinel) | (eta1 == invalid_sentinel) | (phi1 == invalid_sentinel)
        bad2 = (pt2 == invalid_sentinel) | (eta2 == invalid_sentinel) | (phi2 == invalid_sentinel)
        mask = bad1[..., None] | bad2[..., None, :]
        out = out.astype(float)  # ensure we can place NaNs
        out[mask] = np.nan

    return out

def save_name(path, name):
    base, ext = os.path.splitext(name)
    candidate = name
    counter = 1

    while os.path.exists(os.path.join(path, candidate)):
        candidate = f"{base}{counter}{ext}"
        counter += 1

    return os.path.join(path, candidate)