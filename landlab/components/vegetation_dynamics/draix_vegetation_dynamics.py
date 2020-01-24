"""
Created CLB April 2017
"""

from landlab import Component
from landlab import RasterModelGrid
from landlab.utils.decorators import use_file_name_or_kwds
import numpy as np
from landlab.grid.gradients import calc_grad_at_link
from landlab.grid.divergence import calc_flux_div_at_node
import scipy.optimize as scopt

class DraixVegetation(Component):
    """
    Landlab component that simulates spatial diffusion, growth and death of a
    vegetation field, interacting (when d is non zero) with geomorphic processes,
    according to equation:
    
    dV/dt = a*laplacian(V) + b*V*(1-V) - c*V - d*abs(deltaz)*V
    
    Where V is normalized vegetation density at each cell (Vmax=1), a is a 
    diffusion rate, b is a growth rate (logistic growth), c is an aging death 
    rate and d is a geomorphic death rate (by erosion/burrying of size deltaz).
    
    The component takes as input a "deltaz_" field which represents the change 
    in topographic elevation due to geomorphic processes. This field needs to 
    be populated in the main script when calling to geomorphic components and 
    before calling the present vegetation component.
    
    Note that geomorphic death is a function of abs(deltaz) meaning that both 
    erosion and burrying are represented here. This could be decoupled later
    if needed.
    
    Construction::
        Vegetation(grid, diffusion_rate=1., growth_rate=1., aging_death_rate=1.,
        geomorphic_deat_rate=1.)
        
    Note: In future developments parameters a, b and c could be spatialized 
    and depend on environment (elevation, slope, aspect...)
        
    Caution: compared to the vegetation_dynamics component, vegetation here is 
    node-based instead of cell-based.
       

    Parameters
    ----------
    grid: RasterModelGrid
        A grid.
    diffusion_rate: float, optional
        Rate of diffusion of vegetation (m2/year).
    growth_rate: float, optional
        Growth coefficient in logistic equation (year-1)
    aging_death_rate: float, optional
        Death rate (year-1). Note that this only accounts for vegetation death 
        due to its own dynamics (i.e. aging). 
    geomorphic_death_rate: float, optional
        Death rate (Geomorphic mecanisms need to be 
        accounted for by coupling other components
   
    """
    _name = 'Vegetation'

    _input_var_names = (
        'vegetation__density',
        'delta_z'
    )

    _output_var_names = (
        'vegetation__density',
    )

    _var_units = {
        'vegetation__density': 'None',
    }

    _var_mapping = {
       'vegetation__density': 'node',
    }

    _var_doc = {
        'vegetation__density':
            'normalized vegetation density at cell, from 0 to 1',
    }

    @use_file_name_or_kwds

    def __init__(self, grid, 
                 colonization_elevation_coeff=0., 
                 colonization_slope_coeff=0.,
                 colonization_veg_coeff=0.,
                 colonization_constant_coeff=1.,
                 growth_slope_coeff=0.,
                 growth_elevation_coeff=0.,
                 growth_constant_coeff=1.,
                 growth_power=0.5,
                 max_veg_density=40,
                 geomorphic_death_coeff=0.,
                 debug_mode=0.,
                 **kwds):
        """
        Parameters
        ----------
        grid: RasterModelGrid
            A grid.
        colonization_elevation_coeff: float, optional
            Coefficient for the impact of elevation in the probability of colonization (m-1).
        colonization_slope_coeff: float, optional
            Coefficient for the impact of slope (in %) in the probability of colonization (no unit).
        colonization_veg_coeff: float, optional
            Coefficient for the impact of previously vegetated neighbors in the probability of colonization (no unit).
        colonization_sconstant_coeff: float, optional
            Constant coefficient in the probability of colonization (no unit).
        growth_slope_coeff: float, optional
            Coefficient for the impact of slope (in %) in the growth rate (no_unit) 
        growth_slope_coeff: float, optional
            Coefficient for the impact of elevation in the growth rate (m-1) 
        growth_constant_coeff: float, optional
            Constant coefficient in the growth rate (no_unit) 
        growth_power: float, optional
            Power applied to vegetation basal area in the growth rate (no unit)
        max_veg_density/ float, optional
            Maximum value (vegetation basal area) allowed in the model (m2/ha)

        """
        self._grid = grid
        self.colonization_elevation_coeff = colonization_elevation_coeff
        self.colonization_slope_coeff = colonization_slope_coeff
        self.colonization_veg_coeff = colonization_veg_coeff
        self.colonization_constant_coeff = colonization_constant_coeff
        print('colo_const_coeff', self.colonization_constant_coeff)
        self.growth_elevation_coeff = growth_elevation_coeff
        self.growth_slope_coeff = growth_slope_coeff
        self.growth_constant_coeff = growth_constant_coeff
        self.growth_power = growth_power
        self.max_veg_density = max_veg_density
        self.geomorphic_death_coeff = geomorphic_death_coeff
        self.debug_mode = debug_mode
        
        self.topographic__elevation = grid.at_node['topographic__elevation']
        self.slope = grid.at_node['topographic__steepest_slope']


        # creating fields

        if 'vegetation__density' in self.grid.at_node:
            self.veg_density = self.grid.at_node['vegetation__density']
        else:
            self.veg_density = self.grid.add_zeros('node', 'vegetation__density')
            
        if 'deltaz' in self.grid.at_node:
            self.deltaz = self.grid.at_node['deltaz']
        else:
            self.deltaz = self.grid.add_zeros('node', 'deltaz')
            
        if 'all_neighbors' in self.grid.at_node:
            self.all_neighbors = self.grid.at_node['all_neighbors']
        else:
            self.all_neighbors=np.concatenate((self.grid.neighbors_at_node,self.grid._RasterModelGrid__diagonal_neighbors_at_node),axis=1)
            self.grid.add_field('all_neighbors',self.all_neighbors,at='node')
            

#%% computes probability at the appropriate time step
#    def proba(self,p,T,p20):
#        s = np.zeros(np.size(p20))
#        for n in range(T):
#            s += p*((1-p)**n)
#        return s
#    
#    def func(self,p,T,p20):
#        return p20-self.proba(p,T,p20)
    

#%%
    def run_one_step(self, dt, **kwds):
        """
        Calculate the evolution of vegetation density for a given period dt (yrs)
        
        """

        # initializes spatialized variables
        veg_density = self.veg_density
        grid = self.grid
        debug_mode = self.debug_mode
        deltaz = self.deltaz
        
        # for each node, search for the number of neighbors that are already vegetated, i.e. where veg_density != 0
        # the resulting array neighbor_veg takes values from 0 to 8
        neighbor_veg = np.sum(veg_density[grid.at_node['all_neighbors']]!=0,axis=1)
        
        
        if debug_mode == 1:
            print('mean neighbor_veg is '+repr(np.mean(neighbor_veg)))
            print('max number of vegetated neighbors for non-vegetated points is '+repr(np.max(neighbor_veg[veg_density==0])))
            print('max slope of non-vegetated points is' + repr(np.max(self.slope[veg_density == 0])))
            print('max slope of vegetated points is' + repr(np.max(self.slope[veg_density != 0])))
        
        # where there is no vegetation: computes colonization probability over 20 yrs
        # (slope is multiplied by 100 because proba was computed based on percent slope)
        colonization_probability_20yrs = np.zeros(grid.number_of_nodes)
        colonization_probability_dt = np.zeros(grid.number_of_nodes)
        colonization_probability_20yrs[veg_density == 0] = np.exp(self.colonization_constant_coeff + \
                                                                  self.colonization_slope_coeff * self.slope[veg_density == 0] * 100 + \
                                                                  self.colonization_elevation_coeff * self.topographic__elevation[veg_density == 0] + \
                                                                  self.colonization_veg_coeff * neighbor_veg[veg_density == 0]) / \
                                                           (1 + np.exp(self.colonization_constant_coeff + \
                                                                       self.colonization_slope_coeff * self.slope[veg_density == 0] * 100 + \
                                                                       self.colonization_elevation_coeff * self.topographic__elevation[veg_density == 0] + \
                                                                       self.colonization_veg_coeff * neighbor_veg[veg_density == 0]))

        # dt in years. Computes the colonization probability over time dt, knowing colonization probability over 20 years
        #colonization_probability_dt[veg_density == 0] = scopt.fsolve(self.func,
        #                                                             x0 = np.zeros(np.size(colonization_probability_20yrs[veg_density == 0])),
        #                                                             args = (int(20./dt),colonization_probability_20yrs[veg_density == 0]))
        #memory issue with this computation ... so use an approximation which is only correct for dt = one year:
        #colonization_probability_dt[veg_density == 0] = 0.05*colonization_probability_20yrs[veg_density == 0]/(1.-colonization_probability_20yrs[veg_density == 0])**0.3
        #colonization_probability_dt[colonization_probability_dt > 1]=1
        #exact formulation using survival probability
        colonization_probability_dt[veg_density == 0] = 1 - (1-colonization_probability_20yrs[veg_density == 0])**(dt/20.)

        if debug_mode == 1:
            print('max colonization probability 20 yrs is '+repr(np.max(colonization_probability_20yrs)))
            print('max colonization probability 1 yr is '+repr(np.max(colonization_probability_dt)))
            print('mean colonization probability 1 yr is '+repr(np.mean(colonization_probability_dt)))

        # compare colonization probability with uniform draw to define node to be colonized
        # attribute a veg_density of 1 to newly colonized nodes
        random_draw = np.random.uniform(0,1,np.sum(veg_density == 0))
        veg_density[veg_density == 0] += 1 * (random_draw < colonization_probability_dt[veg_density==0])

        if debug_mode==1:
            print('examples of the first random draws: '+repr(random_draw[0:5]))
            print('max veg density was before '+repr(np.max(veg_density)))
            print('mean veg density was before '+repr(np.mean(veg_density)))

        # where there is already vegetation: computes vegetation growth
        veg_density[veg_density != 0] += dt * veg_density[veg_density != 0] ** self.growth_power * \
                                        (self.growth_constant_coeff + \
                                         self.growth_elevation_coeff * self.topographic__elevation[veg_density != 0] + \
                                         self.growth_slope_coeff * self.slope[veg_density != 0] * 100) * \
                                        (self.max_veg_density - veg_density[veg_density != 0]) - \
                                        self.geomorphic_death_coeff * deltaz[veg_density != 0] * veg_density[veg_density !=0]

        veg_density[veg_density != 0] = np.minimum(veg_density[veg_density != 0],self.max_veg_density)  

        if debug_mode == 1:
            print('max veg density is now '+repr(np.max(veg_density)))
            print('mean veg density is now '+repr(np.mean(veg_density)))
            print(' ')
            
       
       
        

