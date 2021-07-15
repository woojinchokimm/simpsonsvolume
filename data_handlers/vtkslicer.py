import logging
import os, sys, re
from pathlib import Path
from data_handlers.heart import Heart
import numpy as np
from data_handlers.vtkfunctions import *


# def clear_screen():
#     os.system('cls')

def extract_rv_dirs(opts):
    dataset = opts.dataset_type
    if dataset=='HFC':
        # check that if HFC, the path to Orientations3D is present
        if (not os.path.exists(opts.rv_path)) or (opts.rv_path == None):
            raise ValueError('For HFC, must provide path to Orientations3D folder.')
            sys.exit()

        rv_dict = dict()
        for textfile in os.listdir(opts.rv_path):
            case_num = int(re.findall(r'\d+', textfile)[0].lstrip("0"))
            with open(os.path.join(opts.rv_path, textfile)) as f:
                lines = f.readlines()

            for line in lines:
                if 'Orientation LV to RV centre' in line:
                    rv_dir = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    rv_dict[case_num] = np.asarray(rv_dir, dtype=float, order='C')
                    break
        rv_data = dict(dictionary=rv_dict)
    elif dataset=='UKBB':
        rv_data = dict(node=1149)
    elif dataset=='YHC':
        rv_data = dict(vec=np.array([1,0,0]))
    else:
        raise ValueError('dataset must be either UKBB, YHC or HFC.')
        sys.exit()

    return rv_data

def manual_apex_nodes(opts):
    apex_dir = os.path.join(opts.vtk_mesh_dir, 'apex_manual.txt')
    manual_dict = dict()
    with open(apex_dir, encoding='utf8') as f:
        for i, line in enumerate(f):
            case = line.split()[0]
            endo_apex_id = int(line.split()[1])
            epi_apex_id = int(line.split()[2])
            manual_dict[case] = [endo_apex_id, epi_apex_id]
    return manual_dict

def get_endo_epi_apex_ids(opts):
    # set epi apex node (checked with paraview)
    apex_dir = os.path.join(opts.vtk_mesh_dir, 'apex_nodes.txt')
    with open(apex_dir, encoding='utf8') as f:
        for i, line in enumerate(f):
            endo_apex_id = int(line.split()[0])
            epi_apex_id = int(line.split()[1])
    return [endo_apex_id, epi_apex_id]

def get_anomalous_cases(opts):
    apex_dir = os.path.join(opts.vtk_mesh_dir, 'anomaly.txt')
    anomalous_cases = []
    with open(apex_dir, encoding='utf8') as f:
        for i, line in enumerate(f):
            for case in line.split():
                anomalous_cases.append(case)
    return anomalous_cases

def is_anomaly(anomalous_cases, case):
    is_anomalous = False
    for anomaly_case in anomalous_cases:
        if case == anomaly_case:
            is_anomalous=True
    return is_anomalous

class VTKSlicer(object):
    def __init__(self, opts):
        input_dir = opts.vtk_mesh_dir + '{}_meshes/'.format(opts.dataset_type)
        vtk_files = sorted([f for f in sorted(os.listdir(input_dir)) if f.endswith(".vtk")])
        self.opts = opts
        self.vtk_files = vtk_files
        self.rv_data = extract_rv_dirs(opts)
        self.opts.vtk_mesh_dir = input_dir
        self.anomalous_cases = get_anomalous_cases(opts)
        self.endo_epi_apex_ids = get_endo_epi_apex_ids(opts)
        self.manu_dict = manual_apex_nodes(opts)

    def create_slices(self):
        for c, case in enumerate(self.vtk_files):

            if is_anomaly(self.anomalous_cases, case):
                print('anomalous case, ignoring..')
                continue

            curr_endo_epi_idxs = self.endo_epi_apex_ids
            if case in list(self.manu_dict.keys()):
                curr_endo_epi_idxs = self.manu_dict[case]
                print('using manual apex: ', curr_endo_epi_idxs)

            ht = Heart(str(case), self.rv_data, curr_endo_epi_idxs, opts=self.opts)
            if self.opts.verbose: self.print_curr_iter(case, c, ht.simp_perr)

    def print_curr_iter(self, case, c, simp_perr):
        print('i {}, dataset {}, typ {}, ang_sep {}, perr {}, case {}'.format(c, self.opts.dataset_type,
                                                                                self.opts.simpson_type,
                                                                                self.opts.view_name,
                                                                                simp_perr,
                                                                                case))

