####################################
# File name: creating_meshes.py
# Author: Ishita Agarwal
# email id: agarw467@purdue.edu
# Created Date: 02/20/2025 22:22
# Description: Surface Creation code
####################################

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import nibabel as nib
import k3d
import matplotlib.pyplot as plt
from lameg.viz import show_surface, rgbtoint
from lameg.surf import postprocess_freesurfer_surfaces


# In[2]:


get_ipython().run_line_magic('env', 'SUBJECTS_DIR=/home/common/bonaiuto/cued_action_meg/derivatives/processed/fs/')


# In[3]:


# Create an 11-layer surface
postprocess_freesurfer_surfaces(
    'sub-001',                                
    '/home/common/bonaiuto/cued_action_meg/derivatives/processed/sub-001/surf/', 
    'multilayer.11.ds.ds_surf_norm.fixed.gii', # name of output file
    n_surfaces=11, 
    ds_factor=0.1, 
    orientation='ds_surf_norm', 
    fix_orientation=True,
    remove_deep=True,
)


# In[4]:


# Create an 11-layer surface
postprocess_freesurfer_surfaces(
    'sub-001',                                
    '/home/common/bonaiuto/cued_action_meg/derivatives/processed/sub-001/surf/', 
    'multilayer.11.ds.ds_surf_norm.not_fixed.gii', # name of output file
    n_surfaces=11, 
    ds_factor=0.1, 
    orientation='ds_surf_norm', 
    fix_orientation=False,
    remove_deep=True,
)


# In[5]:


postprocess_freesurfer_surfaces(
    'sub-001',                                
    '/home/common/bonaiuto/cued_action_meg/derivatives/processed/sub-001/surf/', 
    'multilayer.11.ds.orig_surf_norm.fixed.gii', # name of output file
    n_surfaces=11, 
    ds_factor=0.1, 
    orientation='orig_surf_norm', 
    fix_orientation=True,
    remove_deep=True,
)


# In[6]:


postprocess_freesurfer_surfaces(
    'sub-001',                                
    '/home/common/bonaiuto/cued_action_meg/derivatives/processed/sub-001/surf/', 
    'multilayer.11.ds.orig_surf_norm.not_fixed.gii', # name of output file
    n_surfaces=11, 
    ds_factor=0.1, 
    orientation='orig_surf_norm', 
    fix_orientation=False,
    remove_deep=True,
)


# In[7]:


postprocess_freesurfer_surfaces(
    'sub-001',                                
    '/home/common/bonaiuto/cued_action_meg/derivatives/processed/sub-001/surf/', 
    'multilayer.11.ds.cps.fixed.gii', # name of output file
    n_surfaces=11, 
    ds_factor=0.1, 
    orientation='cps', 
    fix_orientation=True,
    remove_deep=True,
)


# In[8]:


postprocess_freesurfer_surfaces(
    'sub-001',                                
    '/home/common/bonaiuto/cued_action_meg/derivatives/processed/sub-001/surf/', 
    'multilayer.11.ds.cps.not_fixed.gii', # name of output file
    n_surfaces=11, 
    ds_factor=0.1, 
    orientation='cps', 
    fix_orientation=False,
    remove_deep=True,
)


# In[ ]:




