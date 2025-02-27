import glob
import shutil
import sys
import time

import numpy as np
import nibabel as nib
import pickle
from lameg.invert import coregister, invert_ebb
from lameg.laminar import model_comparison
from lameg.simulate import run_dipole_simulation, run_current_density_simulation
from lameg.util import spm_context
import json
import os
import os.path as op

from utilities.utils import gaussian, get_fiducial_coords


def run(subject_id, session_id, base_vertex, sim_vertices, snr, dipole_moment, win_size, json_file, layers_simd,
        analysis_name, sim_key):
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    out_path = os.path.join(parameters["output_path"], analysis_name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    sim_id = "_".join(["lay"] + [str(i) for i in layers_simd] + ["vx"] + [str(i) for i in sim_vertices])

    output_file = os.path.join(
        out_path,
        f"{analysis_name}_sim_ix_{sim_key}_{sim_id}_snr_{snr}.pickle"
    )

    if not os.path.exists(output_file):
        path = parameters["dataset_path"]
        der_path = op.join(path, "derivatives")
        proc_path = op.join(der_path, "processed")

        print("ID:", subject_id)

        sub_path = op.join(proc_path, subject_id)

        ses_path = op.join(sub_path, session_id)

        lpa, nas, rpa = get_fiducial_coords(subject_id, json_file)

        # Patch size to use for inversion
        patch_size = 5
        # Number of temporal modes
        n_temp_modes = 4
        # Window of interest
        woi = [-int(.5 * win_size), int(.5 * win_size)]

        tmp_dir = op.join(out_path, f'{analysis_name}_{sim_id}_snr_{snr}_{win_size}')
        if not op.exists(tmp_dir):
            os.mkdir(tmp_dir)

        # Native space MRI to use for coregistration
        shutil.copy(os.path.join(sub_path, 't1w.nii'),
                    os.path.join(tmp_dir, 't1w.nii'))
        surf_files = glob.glob(os.path.join(sub_path, 't1w*.gii'))
        for surf_file in surf_files:
            shutil.copy(surf_file, tmp_dir)
        mri_fname = os.path.join(tmp_dir, 't1w.nii')

        surf_dir = os.path.join(sub_path, 'surf')

        n_layers = 11

        # Get name of each mesh that makes up the layers of the multilayer mesh - these will be used for the source
        # reconstruction
        orientation_method = 'link_vector.fixed'

        shutil.copy(os.path.join(surf_dir, 'multilayer.11.ds.link_vector.fixed.gii'),
                    os.path.join(tmp_dir, 'multilayer.11.ds.link_vector.fixed.gii'))
        multilayer_mesh_fname = os.path.join(tmp_dir, 'multilayer.11.ds.link_vector.fixed.gii')
        shutil.copy(os.path.join(surf_dir, 'FWHM5.00_multilayer.11.ds.link_vector.fixed.mat'),
                    os.path.join(tmp_dir, 'FWHM5.00_multilayer.11.ds.link_vector.fixed.mat'))

        layers = np.linspace(1, 0, n_layers)
        layer_fnames = []
        for layer in layers:

            if layer == 1:
                name = f'pial.ds.{orientation_method}'
            elif layer == 0:
                name = f'white.ds.{orientation_method}'
            else:
                name = f'{layer:.3f}.ds.{orientation_method}'

            shutil.copy(os.path.join(surf_dir, f'{name}.gii'),
                        os.path.join(tmp_dir, f'{name}.gii'))
            layer_fnames.append(os.path.join(tmp_dir, f'{name}.gii'))
            shutil.copy(os.path.join(surf_dir, f'FWHM5.00_{name}.mat'),
                        os.path.join(tmp_dir, f'FWHM5.00_{name}.mat'))

        data_file = os.path.join(ses_path,
                                 f'spm/pcspm_converted_autoreject-{subject_id}-{session_id}-motor-epo.mat'
                                 )
        data_path, data_file_name = os.path.split(data_file)
        data_base = os.path.splitext(data_file_name)[0]

        # Copy data files to tmp directory
        shutil.copy(
            os.path.join(data_path, f'{data_base}.mat'),
            os.path.join(tmp_dir, f'{data_base}.mat')
        )
        shutil.copy(
            os.path.join(data_path, f'{data_base}.dat'),
            os.path.join(tmp_dir, f'{data_base}.dat')
        )

        # Construct base file name for simulations
        base_fname = os.path.join(tmp_dir, f'{data_base}.mat')

        # simulation signal
        time = np.linspace(-1, 1, num=1201)
        sim_signal = gaussian(time, 0.0, 0.4, dipole_moment)

        mesh = nib.load(multilayer_mesh_fname)
        verts_per_surf = int(mesh.darrays[0].data.shape[0] / n_layers)

        sim_signals = np.vstack([sim_signal] * len(sim_vertices))
        dipole_moments = [dipole_moment] * len(sim_vertices)
        patch_sizes = [patch_size] * len(sim_vertices)

        sim_vx_res = {}
        sim_vx_res["woi"] = woi
        sim_vx_res["sim_vertex"] = sim_vertices
        sim_vx_res["base_vertex"] = base_vertex
        sim_vx_res["time"] = time
        sim_vx_res["snr"] = snr
        sim_vx_res["hann"] = True
        sim_vx_res["signal"] = sim_signal
        sim_vx_res["sim_name"] = analysis_name
        sim_vx_res["sim_id"] = sim_id
        sim_vx_res["layers_used"] = layers_simd

        with spm_context() as spm:
            coregister(
                nas, lpa, rpa, mri_fname,
                multilayer_mesh_fname,
                base_fname, spm_instance=spm,
                viz=False
            )

            [_, _] = invert_ebb(
                multilayer_mesh_fname, base_fname,
                n_layers, patch_size=patch_size,
                n_temp_modes=n_temp_modes,
                spm_instance=spm,
                viz=False
            )

            prefix = f'{analysis_name}_{sim_id}_'
            sim_fname = run_current_density_simulation(
                base_fname,
                prefix,
                sim_vertices,
                sim_signals,
                dipole_moments,
                patch_sizes,
                snr,
                average_trials=True,
                spm_instance=spm
            )

            [layerF, _] = model_comparison(
                nas,
                lpa,
                rpa,
                mri_fname,
                layer_fnames,
                sim_fname,
                method='MSP',
                viz=False,
                spm_instance=spm,
                invert_kwargs={
                    'priors': [base_vertex],
                    'woi': woi,
                    'hann_windowing': True,
                    'patch_size': patch_size,
                    'n_temp_modes': n_temp_modes
                }
            )
            sim_vx_res['layerF'] = layerF

        with open(output_file, "wb") as fp:
            pickle.dump(sim_vx_res, fp)

        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    # parsing command line arguments
    try:
        sim_idx = int(sys.argv[1])
    except:
        print("incorrect simulation index")
        sys.exit()

    try:
        json_file = sys.argv[2]
        print("USING:", json_file)
    except:
        json_file = "settings.json"
        print("USING:", json_file)

    try:
        iter_file = sys.argv[3]
        print("USING:", json_file)
    except:
        iter_file = "mid_sd_iteration_2_source_snr.json"
        print("USING:", iter_file)

    with open(iter_file) as jsfile:
        iteration = json.load(jsfile)

    subject_id = "sub-001"
    session_id = "ses-01"

    snr, base_vertex, layers, sim_vertices, dipole_moment = iteration[str(sim_idx)]
    win_size = 50

    analysis_name = "mid_sd_iteration_2_source_snr"

    run(subject_id, session_id, base_vertex, sim_vertices, snr, dipole_moment, win_size, json_file, layers,
        analysis_name, sim_idx)

