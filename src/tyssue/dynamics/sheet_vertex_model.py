"""
Vertex model for an Epithelial sheet (see definitions).

Depends on the sheet vertex geometry functions.
"""

import numpy as np

from copy import deepcopy

from .sheet_gradients import length_grad, height_grad, area_grad
from ..utils.utils import _to_3d

def get_default_mod_specs():
    """
    Loads the default model specifications

    Returns
    -------
    defautl_mod_spec: dict, the default values for the model
      specifications
    """
    default_mod_specs = {
        "cell": {
            "contractility": (0.04, np.float),
            "vol_elasticity": (1., np.float),
            "prefered_height": (10., np.float),
            "prefered_area": (24., np.float),
            "prefered_vol": (0., np.float),
            },
        "je": {
            "line_tension": (0.12, np.float),
            },
        "jv": {
            "radial_tension": (0., np.float),
            },
        "settings": {
            'grad_norm_factor': 1.,
            'nrj_norm_factor': 1.,
            }
        }
    return default_mod_specs


def dimentionalize(mod_specs, **kwargs):
    """
    Changes the values of the input gamma and lambda parameters
    from the values of the prefered height and area.
    Computes the norm factor.
    """

    dim_mod_specs = deepcopy(mod_specs)
    dim_mod_specs.update(**kwargs)

    Kv = dim_mod_specs['cell']['vol_elasticity'][0]
    A0 = dim_mod_specs['cell']['prefered_area'][0]
    h0 = dim_mod_specs['cell']['prefered_height'][0]
    gamma = dim_mod_specs['cell']['contractility'][0]

    dim_mod_specs['cell']['contractility'] = (gamma * Kv*A0 * h0**2,
                                              np.float)

    dim_mod_specs['cell']['prefered_vol'] = (A0 * h0, np.float)

    lbda = dim_mod_specs['je']['line_tension'][0]
    dim_mod_specs['je']['line_tension'] = (lbda * Kv * A0**1.5 * h0**2,
                                           np.float)

    dim_mod_specs['settings']['grad_norm_factor'] = Kv * A0**1.5 * h0**2
    dim_mod_specs['settings']['nrj_norm_factor'] = Kv * (A0*h0)**2

    return dim_mod_specs


def elastic_force(element_df,
                  var='vol',
                  elasticity='vol_elasticity',
                  prefered='prefered_vol'):
    params = {'x': var,
              'K': elasticity,
              'x0': prefered}
    force = element_df.eval('{K} * ({x} - {x0})'.format(**params))
    return force


def elastic_energy(element_df,
                   var='vol',
                   elasticity='vol_elasticity',
                   prefered='prefered_vol'):
    params = {'x': var,
              'K': elasticity,
              'x0': prefered}
    energy = element_df.eval('0.5 * {K} * ({x} - {x0}) ** 2'.format(**params))
    return energy


def compute_energy(sheet, full_output=False):
    '''
    Computes the tissue sheet mesh energy.

    Parameters
    ----------
    * mesh: a :class:`tyssue.object.sheet.Sheet` instance
    * full_output: if True, returns the enery components
    '''
    # consider only live cells:
    live_cell_df = sheet.cell_df[sheet.cell_df.is_alive == 1]
    upcast_alive = sheet.upcast_cell(sheet.cell_df.is_alive)
    live_je_df = sheet.je_df[upcast_alive == 1]

    E_t = live_je_df.eval('line_tension * length / 2')
    E_v = elastic_energy(live_cell_df,
                         var='vol',
                         elasticity='vol_elasticity',
                         prefered='prefered_vol')
    E_c = live_cell_df.eval('0.5 * contractility * perimeter ** 2')
    if full_output:
        return (E / sheet.nrj_norm_factor for E in (E_t, E_c, E_v))
    else:
        return (E_t.sum() + (E_c+E_v).sum()) / sheet.nrj_norm_factor

def compute_gradient(sheet, components=False,
                     dcoords=None, ncoords=None):
    '''
    If components is True, returns the individual terms
    (grad_t, grad_c, grad_v)
    '''

    if dcoords is None:
        dcoords = ['d'+c for c in sheet.coords]
    if ncoords is None:
        ncoords = ['n'+c for c in sheet.coords]
    norm_factor = sheet.nrj_norm_factor

    grad_lij = length_grad(sheet)

    grad_t = tension_grad(sheet, grad_lij)
    grad_c = contractile_grad(sheet, grad_lij)
    grad_v_srce, grad_v_trgt = elastic_grad(sheet, sheet.coords)# * _to_3d(live_je)

    grad_i = ((grad_t.sum(level='srce') - grad_t.sum(level='trgt'))/2 +
              grad_c.sum(level='srce') - grad_c.sum(level='trgt') +
              grad_v_srce.sum(level='srce') + grad_v_trgt.sum(level='trgt'))
    if components:
        return grad_t, grad_c, grad_v_srce, grad_v_trgt
    #return (grad_i.sum(level='srce') - grad_i.sum(level='trgt')) / norm_factor
    return grad_i / norm_factor

def tension_grad(sheet, grad_lij):

    live_je = sheet.upcast_cell(sheet.cell_df['is_alive'])
    grad_t = (grad_lij
              * _to_3d(sheet.je_df['line_tension'] * live_je))

    #grad_t = _grad_t.sum(level='srce').loc[sheet.jv_idx]
    return grad_t


def contractile_grad(sheet, grad_lij):

    gamma_ = sheet.cell_df.eval('contractility * perimeter * is_alive')
    gamma = sheet.upcast_cell(gamma_)
    grad_c = grad_lij * _to_3d(gamma)
    return grad_c


def elastic_grad(sheet, coords=None):
    ''' Computes
    :math:`\nabla_i \left(K (V_\alpha - V_0)^2\right)`:
    '''
    if coords is None:
        coords = sheet.coords

    # volumic elastic force
    # this is K * (V - V0)
    kv_v0_ = elastic_force(sheet.cell_df,
                           var='vol',
                           elasticity='vol_elasticity',
                           prefered='prefered_vol')

    kv_v0_ = kv_v0_ * sheet.cell_df['is_alive']
    kv_v0 = _to_3d(sheet.upcast_cell(kv_v0_))

    je_h = _to_3d(sheet.upcast_srce(sheet.jv_df['height']))
    area_ = sheet.je_df['sub_area']
    area = _to_3d(area_)
    grad_a_srce, grad_a_trgt = area_grad(sheet, coords)

    grad_v_srce = kv_v0 * (je_h * grad_a_srce +
                           area * height_grad(sheet, coords))
    grad_v_trgt = kv_v0 * (je_h * grad_a_trgt)

    return grad_v_srce, grad_v_trgt