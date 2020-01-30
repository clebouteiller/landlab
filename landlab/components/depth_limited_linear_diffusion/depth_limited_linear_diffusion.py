# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:15:54 2020

@author: caroline.le-bouteill
"""

import numpy as np

from landlab import Component, LinkStatus


class DepthLimitedLinearDiffuser(Component):

    """This component implements a linear diffusion which is limited 
    by the amount of soil available. (It is a simplified version of the 
    DepthDependantDiffuser component)

    The flux q_s is given as q_s = D S where D is is the diffusivity and
    S is the slope. 
    
    The component works as follows:
        - computes a diffusion flux 
        - computes a soil depth from the divergence of this flux 
        combined to the soil production rate
        - clips the resulting soil depth to zero (no negative soil depth)

    Clipping the resulting soil depth to zero means that the amount of material 
    moved by diffusion during one time step is limited by the amount of soil 
    available (previously there + created by weathering during the time step)

    """

    _name = "DepthLimitedLinearDiffuser"


    _info = {
        "bedrock__elevation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "elevation of the bedrock surface",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of soil or weathered bedrock",
        },
        "soil__flux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m^2/yr",
            "mapping": "link",
            "doc": "flux of soil in direction of link",
        },
        "soil_production__rate": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/yr",
            "mapping": "node",
            "doc": "rate of soil production at nodes",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__slope": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "link",
            "doc": "gradient of the ground surface",
        },
    }

    def __init__(self, grid, linear_diffusivity=1.0):
        """
        Parameters
        ----------
        grid: ModelGrid
            Landlab ModelGrid object
        linear_diffusivity: float
            Hillslope diffusivity, m**2/yr
        """
        super(DepthLimitedLinearDiffuser, self).__init__(grid)
        # Store grid and parameters

        self._K = linear_diffusivity
 
        # create fields
        # elevation
        self._elev = self._grid.at_node["topographic__elevation"]
        self._depth = self._grid.at_node["soil__depth"]
        self._soil_prod_rate = self._grid.at_node["soil_production__rate"]

        # slope
        if "topographic__slope" in self._grid.at_link:
            self._slope = self._grid.at_link["topographic__slope"]
        else:
            self._slope = self._grid.add_zeros("topographic__slope", at="link")

        # soil flux
        if "soil__flux" in self._grid.at_link:
            self._flux = self._grid.at_link["soil__flux"]
        else:
            self._flux = self._grid.add_zeros("soil__flux", at="link")

        # bedrock elevation 
        if "bedrock__elevation" in self._grid.at_node:
            self._bedrock = self._grid.at_node["bedrock__elevation"]
        else:
            self._bedrock = self._grid.add_zeros("bedrock__elevation", at="node")

    def soilflux(self, dt):
        """Calculate soil flux for a time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """

        # update soil thickness 
        # (be careful if bedrock__elevation was not defined before!)
        self._grid.at_node["soil__depth"][:] = (
            self._grid.at_node["topographic__elevation"]
            - self._grid.at_node["bedrock__elevation"]
        )
 
        # Calculate gradients
        slope = self._grid.calc_grad_at_link(self._elev)
        slope[self._grid.status_at_link == LinkStatus.INACTIVE] = 0.0

        # Calculate flux
        self._flux[:] = (
            -self._K
            * slope
        )

        # Calculate flux divergence
        dqdx = self._grid.calc_flux_div_at_node(self._flux)
        print('soilflux is',np.mean(self._flux))

        # Calculate change in soil depth
        dhdt = self._soil_prod_rate - dqdx
        print('dhdt is',np.mean(dhdt))

        # Calculate soil depth at nodes
        self._depth[self._grid.core_nodes] += dhdt[self._grid.core_nodes] * dt
        print('soildepth is',np.mean(self._depth[self._grid.core_nodes]))

        # prevent negative soil thickness
        self._depth[self._depth < 0.0] = 0.0
        print('soildepth  after clip is',np.mean(self._depth[self._grid.core_nodes]))

        # Calculate bedrock elevation
        self._bedrock[self._grid.core_nodes] -= (
            self._soil_prod_rate[self._grid.core_nodes] * dt
        )
        print('soil prod rate is',np.mean(self._soil_prod_rate[self._grid.core_nodes]))

        # Update topography
        self._elev[self._grid.core_nodes] = (
            self._depth[self._grid.core_nodes] + self._bedrock[self._grid.core_nodes]
        )

    def run_one_step(self, dt):
        """

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """
        self.soilflux(dt)

