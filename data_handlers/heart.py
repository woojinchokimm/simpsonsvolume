import logging
import os, re, sys
from pathlib import Path

import numpy as np
import vtk
vtk.vtkObject.GlobalWarningDisplayOff() # to remove the annoying warnings output window
import pickle
from PIL import Image
from vtk.util.numpy_support import vtk_to_numpy
from data_handlers.vtkfunctions import *
import math
from scipy.spatial.distance import cdist

class Heart:
    try:
        def __init__(self, filename, rv_data, endo_epi_apex_ids, opts=None):
            # set filename
            self.opts = opts
            self.filename, self.input_type = filename.split('.')
            self.output_dir = opts.output_dir
            self.rv_data = rv_data
            self.case_num =  re.findall(r'\d+', self.filename)[0]
            self.disp  = opts.debug
            self.show_top_pts = self.disp

            # ignore this if current case is anomalous
            if self.case_num in opts.anomaly:
                print('skipping anomalous case :', self.case_num)
                return

            if opts is not None:
                self.verbose = opts.verbose
            else:
                logging.warning("WARNING: opts not used to initialize, using default values")
                self.verbose = True

            # set base parameters
            self.typ = opts.simpson_type
            self.dL_typ = 'max'
            self.data_typ = opts.dataset_type
            self.num_landmark_pairs = opts.num_disks
            self.set_viewing_angle(opts.view_name)
            self.enclosed = opts.enclosed
            self.needle = opts.ptsz_disp
            self.set_weights()
            self.segment_fnms_rootdir = opts.nodelabels_dir
            self.rv_data = rv_data
            self.a4ch_offset = opts.a4c_offset
            self.typ_dict = {'0': 'BOD', '1': 'TBC', '2': 'SBR'}
            self.landmark_path = os.path.join(self.output_dir,'{}/{}/{}'.format(self.data_typ, self.typ_dict[str(self.typ)], self.ang_sep))


            # logging vergbose
            if self.verbose:
                logging.info(f"filename = {self.filename}")
                logging.info('Reading the data from {}.{}...'.format(self.filename, self.input_type))

            # set mesh poly, triangles and points
            if self.input_type == 'vtk':
                self.mesh, self.meshActor = self.read_vtk()
                self.mesh_poly = self.mesh.GetOutput()
                self.triangles = self.mesh_poly.GetPolys().GetData()
                self.points = self.mesh_poly.GetPoints()

            # segment endo/epi + inner/outer rim points
            self.endo_poly, self.epi_poly, self.inner_rim_poly, self.outer_rim_poly = self.segment_data(self.segment_fnms_rootdir)
            self.endoActor, self.epiActor, self.irActor, self.orActor = self.get_actors_from_segments(disp=self.disp)
            self.C = self.set_center_irp()
            self.endo_apex_node = self.points.GetPoint(endo_epi_apex_ids[0])
            self.epi_apex_node = self.points.GetPoint(endo_epi_apex_ids[1])
            self.rv_dir = self.set_rv_dir()


            # apical view extraction
            if not opts.exclude_landmarks:
                # print('extracting landmarks..')
                self.plane_colors = [(0, 255, 0), (0, 0, 255)]  # 4,2,3 : green, blue
                self.orig_view_angles = [self.a4ch_offset, self.a4ch_offset + self.ang_sep]
                self.axis_of_rot_normalized, self.original_planes = self.set_original_planes(disp=self.disp)
                self.orig_cut_poly_array, self.orig_planeActor, ren = self.get_cut_poly_array(disp=self.disp)

                # landmark extraction
                self.compute_ground_truth_volume(disp=0)
                self.lines_act = self.get_landmarks([self.disp, self.disp])
                self.save_pickle()

            # volume extraction
            if not opts.exclude_volume:
                # print('extracting volumes..')
                self.simp_perr = self.compute_volume([self.disp, self.disp])
                self.save_pickle()

            # stat extraction
            if not opts.exclude_mesh_traits:
                # print('extracting mesh traits..')
                self.angs_a4c_to_major = self.get_Am_and_mu(inner_rim_poly = self.inner_rim_poly, disp=self.disp)
                self.save_pickle()

            # plot eccentricity profile
            if not opts.exclude_eccent_profile:
                self.plot_ellipticity_profile_slices(disp=1, display_slices=1)


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, exc_obj, fname, exc_tb.tb_lineno)
        sys.exit()

    def setVar(self, var):
        for key, value in var.items():
            setattr(self, key, value)


    ## initialization functions
    def read_vtk(self):
        """
        Read the source file
        """
        full_path = os.path.join(self.opts.vtk_mesh_dir, '.'.join([self.filename, self.input_type]))
        assert os.path.isfile(full_path), 'File {} does not exist!'.format(self.filename)

        reader = vtk.vtkDataReader()
        reader.SetFileName(full_path)
        reader.Update()

        if reader.IsFileUnstructuredGrid():
            reader = vtk.vtkUnstructuredGridReader()
        elif reader.IsFilePolyData():
            reader = vtk.vtkPolyDataReader()
        elif reader.IsFileStructuredGrid():
            reader = vtk.vtkStructuredGridReader()
        elif reader.IsFileStructuredPoints():
            reader = vtk.vtkStructuredPointsReader()
        elif reader.IsFileRectilinearGrid():
            reader = vtk.vtkRectilinearGridReader()
        else:
            print('Data format unknown...')
            sys.exit()

        reader.SetFileName(full_path)
        reader.Update()  # Needed because of GetScalarRange
        scalar_range = reader.GetOutput().GetScalarRange()
        if self.verbose:
            logging.info('Scalar range: \n{}'.format(scalar_range))

        meshActor = get_actor_from_polydata(reader.GetOutput(), (1,0,0), dataset=reader.IsFileUnstructuredGrid())
        return reader, meshActor

    def set_viewing_angle(self, view_name):
        if view_name == 'a3ch':
            self.ang_sep = 90
        elif view_name == 'a2ch':
            self.ang_sep = 60
        else:
            raise ValueError('--view_name must be either a3ch or a2ch')

    def set_weights(self):
        # These are the weights for the optimization problem of finding the lowest and top points of each contour.
        if self.data_typ in ['UKBB', 'YHC']:
            self.vw = 0.6
            self.hw = 0.4
            self.tw = 0.5
        elif self.data_typ == 'HFC': # must be different because more spherical
            self.vw = 0.5
            self.hw = 0.5
            self.tw = 0.5
        else:
            raise ValueError('--dataset_type must be UKBB, YHC or HFC.')
            sys.exit()

    def set_rv_dir(self):
        key = next(iter(self.rv_data))
        if key == 'node':
            rv_node = np.asarray(self.points.GetPoint(self.rv_data[key]))
            rv_dir = normalize(rv_node - self.C)
        elif key == 'dictionary':
            rv_dict = self.rv_data[key]
            rv_dir = rv_dict[int(self.case_num)]
        elif key == 'vec':
            rv_dir = self.rv_data[key]
        else:
            raise ValueError('rv_data must have 3 possible keys: node, dictionary or vec')
            sys.exit()

        return rv_dir

    ## segmentation based function

    def segment_data(self, fnms_root_dir):
        """
        Extract the endo and epi labels based on the nodelabel.csv file provided.
        """

        fnms = [os.path.join(fnms_root_dir, "{}_endo_labels.csv".format(self.data_typ)),
                os.path.join(fnms_root_dir, "{}_epi_labels.csv".format(self.data_typ)),
                os.path.join(fnms_root_dir, "{}_inner_rim_points.csv".format(self.data_typ)),
                os.path.join(fnms_root_dir, "{}_outer_rim_points.csv".format(self.data_typ))]

        endo_fnm = fnms[0]
        epi_fnm = fnms[1]
        inner_rim_fnm = fnms[2]
        outer_rim_fnm = fnms[3]

        # import the text file and set all numbers as int (since we only have ids)
        endo_arr = np.genfromtxt(endo_fnm, delimiter=",", skip_header=1, dtype=int)
        epi_arr = np.genfromtxt(epi_fnm, delimiter=",", skip_header=1, dtype=int)
        inner_rim_arr = np.genfromtxt(inner_rim_fnm, delimiter=",", skip_header=1, dtype=int)
        outer_rim_arr = np.genfromtxt(outer_rim_fnm, delimiter=",", skip_header=1, dtype=int)

        self.inner_num_rim_pts = inner_rim_arr.shape[0]

        # create the polydata from these arrays
        endo_poly = self.get_endo_epi_myocardium(endo_arr)
        epi_poly = self.get_endo_epi_myocardium(epi_arr)
        inner_rim_poly = self.get_inner_outer_rim(inner_rim_arr)
        outer_rim_poly = self.get_inner_outer_rim(outer_rim_arr)

        return endo_poly, epi_poly, inner_rim_poly, outer_rim_poly

    def get_endo_epi_myocardium(self, arr):
        """  This function extracts the endo and epi layer of the mesh.

        Inputs:
            arr : the information on cell ids and point ids
            typ : 0 is polygons are required (for endo and epi poly)
                  1 means only points are required (for rim)

        Output:
            polydata (labelled for endo/epi_poly)
        """

        tot_num_cells = arr.shape[0]

        if self.data_typ in ['YHC', 'HFC']:
            arr = arr[:, 4:]
            selected_idx = []
            for i in range(tot_num_cells):
                if arr[i, 1] == 1:  # if vtkIsSelected == True
                    selected_idx.append(arr[i, 0])
        elif self.data_typ == 'UKBB':
            selected_idx = arr
        else:
            raise ValueError('--dataset_typ should be either UKBB, YHC or HFC.')
            sys.exit()

        num_selected = len(selected_idx)
        polydata = vtk.vtkPolyData()
        points = self.points  # you need to include all the points from the original data set, the only thing that differs is which triangles are shown
        triangles = vtk.vtkCellArray()
        for i in range(num_selected):
            cellId = selected_idx[i]
            cell = self.mesh_poly.GetCell(cellId)
            triangles.InsertNextCell(cell)

        # create the polydata
        # you need to put it ALL the points of the mesh (not just the ones labelled)
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)

        # make sure now you clean the polydata to remove unused points
        cpd = clean_data(polydata)
        polydata = cpd.GetOutput()

        return polydata

    def get_inner_outer_rim(self, rim_arr):
        """  This function extracts the inner and outer rim of the base of the mesh.

        Inputs:
            typ : 0 is polygons are required (for endo and epi poly)
                  1 means only points are required (for rim)
            rim_arr : contains pointid information (rather than cellid)

        Output:
            rim polydata
        """

        tot_num_cells = rim_arr.shape[0]

        if self.data_typ in ['YHC', 'HFC']:
            rim_arr = rim_arr[:, 3:]
            selected_idx = []
            for i in range(tot_num_cells):
                if rim_arr[i, 1] == 1:  # if vtkIsSelected == True
                    selected_idx.append(rim_arr[i, 0])
        elif self.data_typ == 'UKBB':
            selected_idx = rim_arr
        else:
            raise ValueError('--dataset_typ should be either UKBB, YHC or HFC.')
            sys.exit()

        num_selected = len(selected_idx)

        rimpoly = vtk.vtkPolyData()  # rim polydata
        points = vtk.vtkPoints()
        for i in range(num_selected):
            pointId = selected_idx[i]
            pt = self.mesh_poly.GetPoints().GetPoint(pointId)
            points.InsertNextPoint(pt)

        rimpoly.SetPoints(points)
        return rimpoly

    def get_actors_from_segments(self, disp=0):
        """
        Converts polydata segment (e.g. epi_poly, endo_poly, inner_rim_poly, ...etc.)
        into actor and displays it.
        """
        # if opt = 0: polydata has only point data (i.e. requires vertex filter)
        # if opt = 1: polydata has point + cell data

        self.meshActor.GetProperty().SetOpacity(0.3)
        self.meshActor.GetProperty().SetColor(1,0,0)
        pointdata = [1,1,0,0]
        for i, polydata in enumerate([self.endo_poly, self.epi_poly, self.inner_rim_poly, self.outer_rim_poly]):
            if pointdata[i] == 0:  # point data
                # add vertex at each point (so no need to insert vertices manually)
                vertexFilter = vtk.vtkVertexGlyphFilter()
                vertexFilter.SetInputData(polydata)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(vertexFilter.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1, 1, 0)
                actor.GetProperty().SetPointSize(7)

                if disp:
                    axes = get_axes_actor([80,80,80], [0,0,0])
                    ren = vtk.vtkRenderer()
                    ren.SetBackground(1.0, 1.0, 1.0)
                    ren.AddActor(axes)
                    ren.AddActor(actor)
                    ren.AddActor(self.meshActor)
                    vtk_show(ren)

            else:  # point data + cell data
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1, 1, 0)
                actor.GetProperty().SetPointSize(7)

                if disp:
                    axes = get_axes_actor([80,80,80], [0,0,0])
                    ren = vtk.vtkRenderer()
                    ren.SetBackground(1.0, 1.0, 1.0)
                    ren.AddActor(actor)
                    ren.AddActor(self.meshActor)
                    ren.AddActor(axes)
                    vtk_show(ren)

            yield actor

    def set_center_irp(self):
        """
        Sets center self.C point using com of vtkpoly.
        'vtkpoly' can be inner ring points, outer ring points or entire base_poly.
        """
        # step 1: find center of base
        numpy_base = vtk_to_numpy(self.inner_rim_poly.GetPoints().GetData())
        return np.mean(numpy_base, axis=0)

    def visualize_mesh(self, display=True):
        # Create the mapper that corresponds the objects of the vtk file into graphics elements
        mapper = vtk.vtkDataSetMapper()
        try:
            mapper.SetInputData(self.mesh.GetOutput())
        except TypeError:
            print('Can\'t get output directly')
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(self.mesh.GetOutputPort())
        mapper.SetScalarRange(self.scalar_range)

        # Create the Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create the Renderer
        renderer = vtk.vtkRenderer()
        renderer.ResetCameraClippingRange()
        renderer.AddActor(actor)  # More actors can be added
        renderer.SetBackground(1, 1, 1)  # Set background to white

        # Create the RendererWindow
        renderer_window = vtk.vtkRenderWindow()
        renderer_window.AddRenderer(renderer)

        # Display the mesh
        # noinspection PyArgumentList
        if display:
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(renderer_window)
            interactor.Initialize()
            interactor.Start()
        else:
            return renderer_window


    ## view specific functions
    def set_original_planes(self, disp=0):
        """
        Function computes the planes for 4, 2 and 3 chamber views.

        How do we set the LV midline (i.e. the axis of rotation)?

        In clinical practice: "straight line was traced between the attachment points of the mitral annulus with the valve leaflets."
             see: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2929292/

        Since we do not have information on ventricles, Paul says
        to simply take midline as apex to center of inner rim points.
        """

        # get 4-chamber view
        plane_pts, four_ch_view_plane_normal = self.find_4ch_view(disp=disp)

        # set rodriguez rotation around midline (apex to C)
        axis_of_rot = np.array(self.endo_apex_node - self.C)
        axis_of_rot_normalized = axis_of_rot/np.linalg.norm(axis_of_rot)

        # get secondary apical-chamber view (90-counterclock rotation from 4ch)
        new_P = my_rodriguez_rotation(plane_pts, axis_of_rot_normalized,
                                    math.radians(self.orig_view_angles[1]))  # rodriguez rotation around midline
        sec_ch_view_plane_normal = find_plane_eq(new_P[0, :], new_P[1, :], new_P[2, :])

        original_planes = np.vstack((four_ch_view_plane_normal,
                                            sec_ch_view_plane_normal))

        return axis_of_rot_normalized, original_planes

    def find_4ch_view(self, disp=0):
        """ +y dir points towards RV -> use this as 4ch view
            more reliable than finding mitral and aortic valve points

            the three points on the 4ch view are:
                1) center of inner rim poly
                2) apex point
                3) the y direction
        """
        # step 1: check if self.C exists
        if self.C is None:
            raise ValueError('Center of inner rim poly is not set! exiting..')
            sys.exit()

        # step 2: find rv direction
        pt_rv_dir = self.C + 50.0*self.rv_dir

        # set plane_pts :
        plane_pts = np.vstack((self.C, pt_rv_dir, self.epi_apex_node))

        # construct plane using p1, p2 and the apex node
        four_ch_view_plane_normal = find_plane_eq(self.C, pt_rv_dir, self.epi_apex_node)

        # if 4ch angle is not 0.0:
        if np.abs(self.orig_view_angles[0]) > 0.0:
            axis_of_rot = np.array(self.endo_apex_node - self.C)
            axis_of_rot_normalized = axis_of_rot/np.linalg.norm(axis_of_rot)

            # get 4-chamber view (90 - counterclock rotation from 4ch looking from base to top)
            new_P = my_rodriguez_rotation(plane_pts, axis_of_rot_normalized,
                                        math.radians(self.orig_view_angles[0]))  # rodriguez rotation around midline
            four_ch_view_plane_normal = find_plane_eq(new_P[0, :], new_P[1, :], new_P[2, :])

        if disp:
            # display x-y-z actor
            axes = get_axes_actor([50,50,50], [0,0,0])

            c_irp_act = include_points(self.C, 1, 2, (1,0,1)) # pink = self.C
            pt_rv_dir_act = include_points(pt_rv_dir, 1, 2, (1,1,0)) # yellow = rv dir
            epi_apex_act = include_points(list(self.endo_apex_node), 1, 2, (1,0,1)) # pink = endo apex

            ren = vtk.vtkRenderer()
            ren.AddActor(self.endoActor)
            ren.AddActor(c_irp_act)
            ren.AddActor(epi_apex_act)
            ren.AddActor(pt_rv_dir_act)
            ren.AddActor(axes)
            vtk_show(ren)

        return plane_pts, four_ch_view_plane_normal

    def get_cut_poly_array(self, disp=0):
        """Function to obtain slice of 3D mesh from given plane (using intersection of point to plane)
            Cuts through endocardium only (not the entire mesh)
        Args:
            planes (2D array) : coefficients of the planes (a | b | c | d) --> each row is a new plane
                              : Is in order 4ch --> 2ch --> 3ch
            offset (float array): 4ch and 2ch specific angles (variability test) or specific offsets (foreshortening test)
            disp (boolean) : 0 for no display, 1 for display
            fix_pts (boolean,point OR boolean, 3 points) : for variability test, we need to make sure that the views are fixed to the apex_node, but for foreshortening test we fix on to the varying offsets!
        Returns:
            cut_poly_array : array 3x2 where rows is 4,2,3ch and columns is endo, epi.
            Displays of the chamber views in different colours
        """
        planes = self.original_planes
        noPlanes = len(planes)
        plane_storer = [] #4, 2, 3ch in this order
        cut_poly_array = []  #4, 2, 3ch in this order

        view_type = ['4ch', '2ch']

        for i in range(noPlanes):
            origin = self.endo_apex_node

            cutPoly_endo, planeActor_endo= get_edges_strips(self.endo_poly, planes[i], origin,
                                                            view_type[i], self.plane_colors[i])
            cutPolys = cutPoly_endo
            planeActors = planeActor_endo

            # color so that 4ch is green, 2ch is blue and 3ch is orange
            if i == 0: #4ch
                planeActor_endo.GetProperty().SetColor(0,1,0) # green
            elif i==1: # 2ch
                planeActor_endo.GetProperty().SetColor(0,0,1) # blue
            else:
                planeActor_endo.GetProperty().SetColor(1,0.5,0) # orange


            if self.epi_poly != None:
                cutPoly_epi, planeActor_epi = get_edges_strips(self.epi_poly, planes[i], origin,
                                                                        view_type[i], self.plane_colors[i])
                cutPolys = [cutPoly_endo, cutPoly_epi]

                if i == 0:  # 4ch
                    planeActor_epi.GetProperty().SetColor(0, 1, 0)  # green
                elif i == 1:  # 2ch
                    planeActor_epi.GetProperty().SetColor(0, 0, 1)  # blue
                else:
                    planeActor_epi.GetProperty().SetColor(1, 0.5, 0)  # orange

                planeActors = [planeActor_endo, planeActor_epi]



            cut_poly_array.append(cutPolys) # 4, 2, 3
            plane_storer.append(planeActors)

        ### display purposes
        ren = self.get_orig_slices_display(plane_storer, disp)
        self.orig_slice_ren = ren


        return cut_poly_array, plane_storer, ren

    def get_orig_slices_display(self, plane_storer, disp):
        # include apex_node

        apexA = include_points(list(self.epi_apex_node), 1, self.needle, (0, 0, 0))

        ## create legend box ##
        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(3)

        legendBox = vtk.vtkCubeSource()
        legendBox.SetXLength(2)
        legendBox.SetYLength(2)
        legend.SetEntry(0, legendBox.GetOutput(), "4 ch", (0, 1, 0)) #green
        legend.SetEntry(1, legendBox.GetOutput(), "2 ch", (0, 0, 1)) #blue

        legend.UseBackgroundOn()
        legend.LockBorderOn()
        legend.SetBackgroundColor(0.5, 0.5, 0.5)

        # create text box to display the angles ..
        textActor = vtk.vtkTextActor()
        textActor.SetInput("4ch" + "\n" + "2ch" )
        textActor.SetPosition2(10, 40)
        textActor.GetTextProperty().SetFontSize(24)
        textActor.GetTextProperty().SetColor(1.0, 0.0, 0.0)

        # display x-y-z actor
        axes = get_axes_actor([self.needle*10,self.needle*10,self.needle*10], [0,0,0])

        # lets display the rv_dir
        rv_actor_length = 1
        rv_dir_act = include_points(list(rv_actor_length*self.rv_dir), 1, self.needle, (1, 0 ,1))

        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)
        self.endoActor.GetProperty().SetColor(1, 1, 1)
        ren.AddActor(self.endoActor)

        try:
            ren.AddActor(plane_storer[0]) # 4ch endo
            ren.AddActor(plane_storer[1]) # 2ch endo
        except:
            ren.AddActor(plane_storer[0][0]) # 4ch endo
            ren.AddActor(plane_storer[0][1]) # 4ch epi

            ren.AddActor(plane_storer[1][0]) # 2ch endo
            ren.AddActor(plane_storer[1][1]) # 2ch epi

        self.endoActor.GetProperty().SetOpacity(0.4)
        ren.AddActor(legend)
        # ren.AddActor2D(textActor)
        # ren.AddActor(axes)
        # ren.AddActor(apexA)

        if len(self.rv_dir) > 1:
            rv_vec_act = get_line_act2(self.C, self.C + 1.7*self.rv_dir, (1,0,1))
            rv_vec_act.GetProperty().SetLineWidth(5)
            com_act = include_points(self.C, 1, self.needle, (0,0,0))
            ren.AddActor(rv_vec_act)
            ren.AddActor(com_act)

        # ren.AddActor(rv_dir_act)

        # # add rv dir point for ukbiobank (checking purposes)
        # rv_dir_pt_ukbb = include_points(list(self.points.GetPoint(1149)), 1, 1, (1,0,1))
        # ren.AddActor(rv_dir_pt_ukbb)

        if disp:
            vtk_show(ren)

        return ren



    ## landmark specific functions
    def get_landmarks(self, display_opts, **kwargs):
        """
        Function wrapper for estimating Simpson's volume depending on typ value.
        typ = 0 : basal
        typ = 1 : vendor (GE)
        typ = 2 : bare

        Important: check if self.orig_cut_poly_array has both endo and epi.

        kwargs : polys = [a4c, a2c]
                planeActors = [a4c, a2c]
        """
        if 'polys' in kwargs:
            polys = kwargs.pop('polys')
            a4c_poly = polys[0]
            a2c_poly = polys[1]
        else:
            if isinstance(self.orig_cut_poly_array[0], list):  # if 2D (i.e. endo and epi)
                a4c_poly = self.orig_cut_poly_array[0][0]
                a2c_poly = self.orig_cut_poly_array[1][0]
            else:  # if 1D
                a4c_poly = self.orig_cut_poly_array[0]
                a2c_poly = self.orig_cut_poly_array[1]

        if 'planeActors' in kwargs: planeActors = kwargs.pop('planeActors')
        else: planeActors = self.orig_planeActor

        params = {}
        if 'low_pts' in kwargs:
            low_pts_kw = kwargs.pop('low_pts')
            low_pts_a2c = low_pts_kw[0]
            low_pts_a4c = low_pts_kw[1]
            params['low_pts'] = [low_pts_a2c, low_pts_a4c]
        if 'top_pts' in kwargs:
            top_pts_kw = kwargs.pop('top_pts')
            top_pts_a2c = top_pts_kw[0]
            top_pts_a4c = top_pts_kw[1]
            params['top_pts'] = [top_pts_a2c, top_pts_a4c]

        ############################################################################

        pts_endo_2ch = vtk_to_numpy(a2c_poly.GetPoints().GetData())
        pts_endo_4ch = vtk_to_numpy(a4c_poly.GetPoints().GetData())

        # initialize
        lowest_point_2ch = None
        lowest_point_4ch = None
        top_pts_2ch = None
        top_pts_4ch = None

        if kwargs:
            for key, value in kwargs.items():
                if key == 'low_pts':
                    lowest_point_2ch = value[0]
                    lowest_point_4ch = value[1]
                if key == 'top_pts':
                    top_pts_2ch = value[0]
                    top_pts_4ch = value[1]

        if (lowest_point_2ch is None) and (lowest_point_4ch is None):
            lowest_point_2ch = self.find_furthest_point_from_top(pts_endo_2ch, None)
            lowest_point_4ch = self.find_furthest_point_from_top(pts_endo_4ch, None)

        if (top_pts_2ch is None) and (top_pts_4ch is None):
            top_pts_2ch, a2c_tp1_idx, a2c_tp2_idx = find_top_points_optimize(pts_endo_2ch,
                                                                             self.C,
                                                                             lowest_point_2ch,
                                                                             self.vw,
                                                                             self.hw,
                                                                             self.tw,
                                                                             self.show_top_pts)
            top_pts_4ch, a4c_tp1_idx, a4c_tp2_idx = find_top_points_optimize(pts_endo_4ch,
                                                                             self.C,
                                                                             lowest_point_4ch,
                                                                             self.vw,
                                                                             self.hw,
                                                                             self.tw,
                                                                             self.show_top_pts)
            # a2c_top_pts_idx = [a2c_tp1_idx, a2c_tp2_idx]
            # a4c_top_pts_idx = [a4c_tp1_idx, a4c_tp2_idx]

        horiz_2ch_a, horiz_2ch_b, Ls_2ch = self.get_landmarks_interpolation(a2c_poly,
                                                    top_pts_2ch, lowest_point_2ch,
                                                    self.typ_dict[str(self.typ)], display_opts[0], '2ch')
        horiz_4ch_a, horiz_4ch_b, Ls_4ch = self.get_landmarks_interpolation(a4c_poly,
                                                    top_pts_4ch, lowest_point_4ch,
                                                    self.typ_dict[str(self.typ)], display_opts[1], '4ch')
        Ls_2ch = Ls_2ch
        Ls_4ch = Ls_4ch

        # first dictionary
        base_dict = dict(typ=self.typ,
                            dL_typ=self.dL_typ,
                            ang_sep=self.ang_sep,
                            case=self.case_num,
                            endo_apex_node=self.endo_apex_node,
                            original_planes=self.original_planes,
                            ideal_horiz_2ch_a=horiz_2ch_a,
                            ideal_horiz_2ch_b=horiz_2ch_b,
                            ideal_horiz_4ch_a=horiz_4ch_a,
                            ideal_horiz_4ch_b=horiz_4ch_b,
                            Ls_2ch=Ls_2ch,
                            Ls_4ch=Ls_4ch,
                            ideal_top_pts_2ch=top_pts_2ch,
                            ideal_top_pts_4ch=top_pts_4ch,
                            ideal_lowest_point_2ch=lowest_point_2ch,
                            ideal_lowest_point_4ch=lowest_point_4ch,
                            ground_truth_vol=self.ground_truth_vol,
                            )

        self.setVar(base_dict)  # update attributes

        ## display
        lines_act = self.display_landmarks_a2ch_a4ch(display_opts)
        return lines_act

    def display_landmarks_a2ch_a4ch(self, display_opts):
        _, lines_2ch_act = display_landmarks(self.ideal_horiz_2ch_a, self.ideal_horiz_2ch_b,
                                             None, self.ideal_top_pts_2ch[0],
                                             self.ideal_top_pts_2ch[1],
                                             self.needle,
                                             display_opts[0])
        _, lines_4ch_act = display_landmarks(self.ideal_horiz_4ch_a, self.ideal_horiz_4ch_b,
                                             None, self.ideal_top_pts_4ch[0],
                                             self.ideal_top_pts_4ch[1],
                                             self.needle,
                                             display_opts[1])
        change_assembly_properties(lines_4ch_act, 1.0, (0, 1, 0))
        change_assembly_properties(lines_2ch_act, 1.0, (0, 0, 1))
        return [lines_4ch_act, lines_2ch_act]

    def find_furthest_point_from_top(self, sorted_pts_endo, top_pts):
        """
        This function finds the lowest point for each view by using the recommendations
        of the ACE (American Committee of Echocardiography):

        https://www.asecho.org/wp-content/uploads/2015/01/ChamberQuantification2015.pdf

        "At the mitral valve (basal) level, the contour is closed by connecting the two opposite sections (top_pts)
        of the mitral ring with a straight line. LV length is defined as the distance between the bisector
        of this line and the apical point of the LV contour, which is most distant to it. The use of the
        longer LV length between the apical two- and four-chamber views is recommended."

        sorted_pts_endo : doesn't really need to be sorted
        """
        if top_pts == None: # if not given then use self.C which is middle of inner rim poly points.
            middle_point = self.C
        else:
            # if top most basal points given, lowest point is found by finding furthest distant point from the center point
            middle_point = (top_pts[0] + top_pts[1])/2.0

        dists = cdist(sorted_pts_endo, middle_point.reshape((1,3))).flatten()

        # 3. Rather than using argmax which returns first occurence of maximum,
            # use np.argwhere and np.amax to get multiple occurences of max.
            # This is useful for views where the apical region is flat.
            # Strategy now is to get the centre of those furthest points.

        lowest_pt_idxs = np.argwhere(dists == np.amax(dists)).flatten().tolist()

        if len(lowest_pt_idxs) > 1:
            lowest_pt = np.mean(sorted_pts_endo[lowest_pt_idxs], axis=0)
        else:
            lowest_pt = sorted_pts_endo[np.argmax(dists)]

        # make function to display top points and lowest point
        if 0:
            sendo = include_points(list(sorted_pts_endo),
                        len(list(sorted_pts_endo)), self.needle/2.0, (0,1,0))
            a1 = include_points(list(lowest_pt), 1, self.needle/2.0, (1,0,0))
            # a2 = include_points(list(top_pts[0]), 1, self.needle, (1,0,0))
            # a3 = include_points(list(top_pts[1]), 1, self.needle, (1,0,0))
            C_act = include_points(list(self.C), 1, self.needle/2.0, (1,0,0))
            ren = vtk.vtkRenderer()
            ren.AddActor(sendo)
            ren.AddActor(a1)
            # ren.AddActor(a2)
            # ren.AddActor(a3)
            ren.AddActor(C_act)
            vtk_show(ren)

        return lowest_pt

    def get_landmarks_interpolation(self, polydata, top_pts, lowest_pt, method, display_opt, view_type):
        """
        Only works if cutpoly views have basal line included.
        Otherwise, we have to add ourself.
        Gets landmarks using linear interpolation?
        method: {'basal', 'GE', 'bare'}


        1. Split polydata into left and right points
        """
        lowest_pt = np.asarray(lowest_pt)
        tp1 = np.asarray(top_pts[0])
        tp2 = np.asarray(top_pts[1])

        # convert to numpy points
        pts = vtk_to_numpy(polydata.GetPoints().GetData())
        # num_pts = pts.shape[0]

        # project onto xy-plane and rotate such that midline is y axis
        com = np.mean(pts, axis=0)
        pts_2d, R_2d = project_onto_xy_plane(pts, com)

        lowest_pt_2d = np.dot(R_2d, (lowest_pt - com))[:2]
        tp1_2d = np.dot(R_2d, (tp1 - com))[:2]
        tp2_2d = np.dot(R_2d, (tp2 - com))[:2]
        tp_center_2d = (tp1_2d + tp2_2d)/2.0
        midline = normalize(tp_center_2d -  lowest_pt_2d)

        # midline must point upwards for proper alignment with y-axis
        # for basal, align line perpendicular to basal line with the y-axis
        basal_vec = normalize(tp1_2d - tp2_2d)
        basal_perp = np.array([basal_vec[1], - basal_vec[0]])
        if np.dot(basal_perp, midline) < 0.0: # make sure basal_perp is pointing upwards
            basal_perp = -basal_perp

        disp_2d_landmarks = 0

        if method == 'BOD':
            aligned_pts, top_pts_al, lowest_pt_al, top_LR_idxs, R_rot, shift_factor = align_2d_pts(pts_2d, lowest_pt_2d,
                                                        [tp1_2d, tp2_2d], basal_perp, disp_2d_landmarks)


            left_pts_2d, right_pts_2d = interpolate_scheme(aligned_pts, self.num_landmark_pairs,
                                                    top_pts_al, lowest_pt_al,
                                                    method,  self.enclosed, display_opt)

            # project back to 3d space
            left_pts = reverse_project_2d_to_3d(left_pts_2d, shift_factor, R_rot, R_2d, com)
            right_pts = reverse_project_2d_to_3d(right_pts_2d, shift_factor, R_rot, R_2d, com)
        elif (method=='TBC') or (method=='SBR'):
            aligned_pts, top_pts_al, lowest_pt_al, top_LR_idxs, R_rot, shift_factor = align_2d_pts(pts_2d, lowest_pt_2d,
                                                        [tp1_2d, tp2_2d], midline, disp_2d_landmarks)
            left_pts_2d, right_pts_2d = interpolate_scheme(aligned_pts, self.num_landmark_pairs,
                                                    top_pts_al, lowest_pt_al, method, self.enclosed,
                                                    display_opt)

            # project back to 3d space
            left_pts = reverse_project_2d_to_3d(left_pts_2d, shift_factor, R_rot, R_2d, com)
            right_pts = reverse_project_2d_to_3d(right_pts_2d, shift_factor, R_rot, R_2d, com)
        else:
            print('incorrect method type')
            sys.exit()

        # get length measures
        Ls = get_cylinder_heights_2D(left_pts_2d, right_pts_2d, lowest_pt_al, method)

        return list(left_pts), list(right_pts), Ls

    def compute_ground_truth_volume(self, disp=0):
        """
        Computes ground truth volume
        Surface filter is used to get the surfaces from the outisde of the volume
        Note that output of d3 is a tetrahedral mesh!
        vtk mass properties only works for triangular mesh which is why we need this surface filter!

        the clean poly data is essential!!!!!!!! for volume calculation especially!!!!
        """

        self.endoActor.GetProperty().SetOpacity(0.2)
        self.endoActor.GetProperty().SetColor(1, 0, 0)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(self.endo_poly)

        d3 = vtk.vtkDelaunay3D()
        d3.SetInputConnection(clean.GetOutputPort())
        d3.SetTolerance(0.01)
        d3.SetAlpha(0.0)
        d3.Update()

        surfaceFilter = vtk.vtkDataSetSurfaceFilter()  # output is triangular mesh
        surfaceFilter.SetInputConnection(d3.GetOutputPort())
        surfaceFilter.Update()

        Mass = vtk.vtkMassProperties()
        Mass.SetInputConnection(surfaceFilter.GetOutputPort())
        Mass.Update()

        self.ground_truth_vol = Mass.GetVolume()/1000.0

        if disp:

            m = vtk.vtkDataSetMapper()
            m.SetInputConnection(d3.GetOutputPort())

            a = vtk.vtkActor()
            a.SetMapper(m)

            # set mapper for epi for visualization
            m2 = vtk.vtkDataSetMapper()
            m2.SetInputData(self.epi_poly)

            epi_actor = vtk.vtkActor()
            epi_actor.SetMapper(m2)
            epi_actor.GetProperty().SetOpacity(0.3)
            epi_actor.GetProperty().SetColor(1,0,0)

            ren = vtk.vtkRenderer()
            ren.SetBackground(0.0, 0.0, 0.0)
            ren.AddActor(epi_actor)
            ren.AddActor(a)

            vtk_show(ren)


    ## volume specific function
    def compute_volume(self, display_opts):
        """
        Function wrapper for estimating Simpson's volume depending on typ value.
        typ = 0 : basal
        typ = 1 : vendor (GE)
        typ = 2 : bare

        Important: check if self.orig_cut_poly_array has both endo and epi.

        kwargs : polys = [a4c, a2c]
                planeActors = [a4c, a2c]

        Load landmarks if not saving them...
        """

        dict_path = self.landmark_path + '/{}_data_dict.pkl'.format(self.case_num)
        if self.opts.exclude_landmarks:
            assert os.path.exists(dict_path), "Must extract landmarks first (do not include --exclude_landmarks"
            data_dict = pickle.load(open(dict_path, 'rb'))
            self.setVar(data_dict)  # update attributes
        else:
            data_dict = self.__dict__.copy()

        self.display_landmarks_a2ch_a4ch(display_opts)

        # 3: compute euclidean distances for 2ch and 4ch
        horiz_2ch_dists, horiz_4ch_dists = compute_horizontal_distances(self.ideal_horiz_2ch_a, self.ideal_horiz_2ch_b,
                                                                        self.ideal_horiz_4ch_a, self.ideal_horiz_4ch_b)

        funcs_dict = {'0' : self.Simpson_basal,
                     '1' : self.Simpson_GE,
                     '2' : self.Simpson_bare}

        ideal_vol = funcs_dict[str(self.typ)](horiz_2ch_dists, horiz_4ch_dists, self.Ls_2ch, self.Ls_4ch)
        simp_perr = (ideal_vol - self.ground_truth_vol) / self.ground_truth_vol * 100.0

        # measure dL disparities
        dL_disparity = self.measure_dL_disparity()

        # measure slanting angles
        sla2, sla4 = self.get_slanting_angle(0)

        # make dictionary to update
        vol_dict = dict(dL_disparity=dL_disparity,
                        a4c_sla=sla4,
                        a2c_sla=sla2,
                        ideal_vol = ideal_vol,
                        simp_perr = simp_perr)

        # update the current dictionary to save again
        data_dict.update(vol_dict)
        self.setVar(data_dict)  # update attributes

        # for item in vars(self).items():
        #     print('attr =', item)
        # sys.exit()
        return simp_perr

    def Simpson_bare(self, horiz_2ch_dists, horiz_4ch_dists, Ls_2ch, Ls_4ch):
        """
        'Bare-rule' Simpson's rule as described in the literature.
        Uses no assumptions to approximate the basal and apical regions.

        i.e. does not use half-cylinders or cones for better approximation.
        """
        ####################################################################################################
        self.dL_4ch = Ls_4ch['dL']
        self.dL_2ch = Ls_2ch['dL']
        funcs_dict = {'min': np.min, 'max': np.max, 'mean': np.mean}
        # print('funcs_dict[self.dL_typ]=', funcs_dict[self.dL_typ])

        dL = funcs_dict[self.dL_typ]([self.dL_2ch, self.dL_4ch])
        # dL = np.mean([self.dL_2ch, self.dL_4ch])  # use mean for synthetic
        self.delta_L = dL
        ####################################################################################################

        summ = 0
        const = (np.pi / 4.0) * dL
        self.vols_per_disk = []

        for i in range(self.num_landmark_pairs): # iterate downwards towards apex
            summ += horiz_2ch_dists[i]*horiz_4ch_dists[i]
            self.vols_per_disk.append(horiz_2ch_dists[i]*horiz_4ch_dists[i]*const)


        vol = const*summ
        vol = np.squeeze(vol)/1000.0 #factor for m

        self.vols_per_disk = np.asarray(self.vols_per_disk)/1000.0
        return vol

    def Simpson_GE(self, horiz_2ch_dists, horiz_4ch_dists, Ls_2ch, Ls_4ch):
        """
        GE method
        Function computes LV volume using Simpson's modified formula.
        We are using GE's formulation which divides the volume of the
        the left ventricle into 3:
            a) cone (to approximate the apex)
            b) elliptical truncated cones (to approximate the middle part of the LV)
            c) half cylinder (to approximate the disk at the basal region)

        n.b. no need for pablo's SHIFT because we use truncated disks

        INPUTS:
            horiz_2ch_dits : the horizontal distances in the 2 chamber view
            horiz_4ch_dist : the horizontal distances in the 4 chamber view
            H_half_cyl_2ch : height of half cylinder for A2C
            H_half_cyl_4ch : height of half cylinder for A4C
            H_cone_2ch : height of cone for A2C
            H_cone_4ch : height of cone for A4C
            L : the length of the long axis (max of A2C and A4C)
                - H_cone and H_half_cyl

        The height of each disc is calculated as a fraction (usually one-twentieth)
        of the LV long axis based on the longer of the two lengths from the two and four-chamber views
        See Figure 3 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2064911/
        """
        ####################################################################################################
        dL_2ch = Ls_2ch['dL']
        dL_4ch = Ls_4ch['dL']
        H_cone_2ch = Ls_2ch['H_cone']
        H_cone_4ch = Ls_4ch['H_cone']
        H_half_cyl_2ch = Ls_2ch['H_half_cyl']
        H_half_cyl_4ch = Ls_4ch['H_half_cyl']

        funcs_dict = {'min': np.min, 'max': np.max, 'mean': np.mean}
        # print('funcs_dict[self.dL_typ]=', funcs_dict[self.dL_typ])

        dL = funcs_dict[self.dL_typ]([dL_2ch, dL_4ch])
        # dL = np.max([dL_2ch, dL_4ch]) # always MAX

        # H_half_cyl_2ch = Ls_2ch[0]
        # H_half_cyl_4ch = Ls_4ch[0]
        # H_cone_2ch = Ls_2ch[2]
        # H_cone_4ch = Ls_4ch[2]

        self.dL_4ch = dL_4ch
        self.dL_2ch = dL_2ch
        self.H_half_cyl_2ch = H_half_cyl_2ch
        self.H_half_cyl_4ch = H_half_cyl_4ch
        self.H_cone_2ch = H_cone_2ch
        self.H_cone_4ch = H_cone_4ch
        ####################################################################################################

        # 1: Estimate basal region with half cylinder to account for slanted base.
        self.vols_per_disk = []
        # half_cyl_vol = (np.pi/8.0) * (horiz_2ch_dists[1]*horiz_4ch_dists[1]) * np.max([H_half_cyl_2ch, H_half_cyl_4ch])
        half_cyl_vol = (np.pi/8.0) * (horiz_2ch_dists[0]*horiz_4ch_dists[0]) * np.max([H_half_cyl_2ch, H_half_cyl_4ch])

        """
        REMEMBER, diameters of cylinder have to be the second highest (i.e. perpendicular to the midline)
                    and NOT the top diameters!!!! Otherwise, you severely overestimate!
        """

        self.half_cyl_vol = half_cyl_vol
        self.vols_per_disk.append(half_cyl_vol)

        # dispall([self.endoActor, list(self.ideal_horiz_2ch_a[0]), list(self.ideal_horiz_2ch_b[0])],
        #         [(1,0,0), (0,1,0), (0,1,0)],
        #         [None, 0.05, 0.05])

        # 2: Estimate middle portion of LV with truncated / full disks.
        volume_discs = []

        for i in range(1, len(horiz_2ch_dists)-1): # skip first as these are for half cyl
            A_1 = (np.pi/4.0) * (horiz_2ch_dists[i]*horiz_4ch_dists[i])
            A_2 = (np.pi/4.0) * (horiz_2ch_dists[i+1]*horiz_4ch_dists[i+1])
            # Disc_volume = (1.0/3.0) * (L / float(self.num_disks)) * (A_1+A_2+(math.sqrt(A_1*A_2)))
            Disc_volume = (1.0/3.0) * dL * (A_1+A_2+(math.sqrt(A_1*A_2)))
            volume_discs.append(Disc_volume)
            self.vols_per_disk.append(Disc_volume)

        # 3: Estimate apex with cone.
        cone_vol = (np.pi/12.0) * np.max([H_cone_2ch,H_cone_4ch]) * (horiz_2ch_dists[-1]*horiz_4ch_dists[-1])

        # 4: Total volume = half_cylinder + disks + cone
        tot_vol = np.sum(volume_discs) + half_cyl_vol + cone_vol
        tot_vol =  np.squeeze(tot_vol)/1000.0

        self.vols_per_disk.append(cone_vol)
        self.vols_per_disk = np.asarray(self.vols_per_disk)/1000.0

        return tot_vol

    def Simpson_basal(self, horiz_2ch_dists, horiz_4ch_dists, Ls_2ch, Ls_4ch):
        """
        Function computes LV volume using Basal method
            - truncated elliptical cones (to approximate the middle part of the LV)
            - elliptical cone

        The height of each disc is calculated as a fraction (usually one-twentieth)
        of the LV long axis based on the longer of the two lengths from the two and four-chamber views
        See Figure 3 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2064911/

        n.b. no need for pablo's SHIFT because we use truncated disks

        INPUTS:
            horiz_2ch_dits : the diameters in the 2 chamber view
                        DOES INCLUDE THE SLANTED LANDMARKS
            horiz_4ch_dist : the diameters in the 4 chamber view
                        DOES INCLUDE THE SLANTED LANDMARKS
            H_cone_2ch : height of cone in A2C
            H_cone_4ch : height of cone in A4C
            L : the length of the long axis (max of A2C and A4C)
            dL_basals : the dL for basal for each disk]

            N.B. up to len(horiz_2ch_dists)-1 since we need two pairs of landmarks for each disk.
            one for i and another pair for i+1 th.
        """
        ############################################################
        # dL_2ch = Ls_2ch[0]
        # dL_4ch = Ls_4ch[0]
        dL_2ch = Ls_2ch['dL']
        dL_4ch = Ls_4ch['dL']
        H_cone_2ch = Ls_2ch['H_cone']
        H_cone_4ch = Ls_4ch['H_cone']

        funcs_dict = {'min': np.min, 'max': np.max, 'mean': np.mean}
        # print('funcs_dict[self.dL_typ]=', funcs_dict[self.dL_typ])
        dL = funcs_dict[self.dL_typ]([dL_2ch, dL_4ch])
        H_cone = np.max([H_cone_2ch, H_cone_4ch])


        self.dL_4ch = dL_4ch
        self.dL_2ch = dL_2ch
        self.H_cone_2ch = H_cone_2ch
        self.H_cone_4ch = H_cone_4ch
        self.dL_2ch = dL_2ch
        self.dL_4ch = dL_4ch
        self.delta_L = dL
        self.H_cone = H_cone
        ############################################################

        # 1: Estimate with truncated / full disks.
        volume_discs = []

        for i in range(0, len(horiz_2ch_dists)-1): # iterate downwards towards apex, skip the first diameter,
            A_1 = (np.pi/4.0) * (horiz_2ch_dists[i]*horiz_4ch_dists[i])
            A_2 = (np.pi/4.0) * (horiz_2ch_dists[i+1]*horiz_4ch_dists[i+1])
            # Disc_volume = (1.0/3.0) * (self.delta_L / float(self.num_disks)) * (A_1+A_2+(math.sqrt(A_1*A_2)))
            Disc_volume = (1.0/3.0) * dL * (A_1+A_2+(math.sqrt(A_1*A_2)))
            volume_discs.append(Disc_volume)

        # 3: Method 1: Estimate apex with cone.
        # cone_vol = (np.pi/12.0) * H_cone * (horiz_2ch_dists[-1] * horiz_4ch_dists[-1])

        # 4: Method 2: Estimae apex with elliptic paraboloid
        # H_cone = dL
        cone_vol = (np.pi / 8.0) * H_cone * (horiz_2ch_dists[-1]*horiz_4ch_dists[-1])

        tot_vol = np.sum(volume_discs) + cone_vol
        tot_vol =  np.squeeze(tot_vol)/1000.0

        self.vols_per_disk = volume_discs
        self.vols_per_disk.append(cone_vol)
        self.vols_per_disk = np.asarray(self.vols_per_disk)/1000.0

        return tot_vol

    def measure_dL_disparity(self):
        # measures dL disparity
        # defined as the distance between the second highest horizontal line in the A4C and
        # the second highest line in the a2c.
        # not exactly correct, if slanting is in a4c, then it should be distance between
        # second highest line in a4c and FIRST highest line in a2c.

        a2c_second_pts = np.array([self.ideal_horiz_2ch_a[1], self.ideal_horiz_2ch_b[1]])
        a4c_second_pts = np.array([self.ideal_horiz_4ch_a[1], self.ideal_horiz_4ch_b[1]])


        disp = 0
        if disp:
            a2c_a = include_points(list(a2c_second_pts), 2, 0.05, (0,1,0))
            a4c_a = include_points(list(a4c_second_pts), 2, 0.05, (0, 0, 1))

            ren = vtk.vtkRenderer()
            ren.AddActor(a2c_a)
            ren.AddActor(a4c_a)
            ren.AddActor(self.endoActor)
            vtk_show(ren)


        mean2 = np.mean(a2c_second_pts, axis=0)
        mean4 = np.mean(a4c_second_pts, axis=0)

        dL_disparity = np.linalg.norm(mean2 - mean4)

        return dL_disparity

    def get_slanting_angle(self, disp):
        """
        Returns slanting angles for 2ch and 4ch.
        This is the angle between the line perpendicular to the midline
        and the basal line.
        """
        # for 2ch
        top_center = ( self.ideal_top_pts_2ch[0] + self.ideal_top_pts_2ch[1] ) / 2.0
        midline = normalize(self.ideal_lowest_point_2ch - top_center)
        baseline_2ch = normalize(self.ideal_top_pts_2ch[0] - self.ideal_top_pts_2ch[1])
        normal = find_plane_eq(self.ideal_top_pts_2ch[0],
                                self.ideal_top_pts_2ch[1],
                                self.ideal_lowest_point_2ch)[:3]

        horizontal_vec_2ch = normalize(np.cross(normalize(normal), midline))

        if np.dot(baseline_2ch, horizontal_vec_2ch) < 0.0:
            baseline_2ch = -baseline_2ch

        slanting_angle_2ch = vg.angle(baseline_2ch, horizontal_vec_2ch, assume_normalized = True, look = normal, units = 'deg')

        # for 4ch
        top_center = ( self.ideal_top_pts_4ch[0] + self.ideal_top_pts_4ch[1] ) / 2.0
        midline = normalize(self.ideal_lowest_point_4ch - top_center)
        baseline_4ch = normalize(self.ideal_top_pts_4ch[0] - self.ideal_top_pts_4ch[1])
        normal = find_plane_eq(self.ideal_top_pts_4ch[0],
                                self.ideal_top_pts_4ch[1],
                                self.ideal_lowest_point_4ch)[:3]
        horizontal_vec_4ch = normalize(np.cross(normalize(normal), midline))

        if np.dot(baseline_4ch, horizontal_vec_4ch) < 0.0:
            baseline_4ch = -baseline_4ch

        slanting_angle_4ch = vg.angle(baseline_4ch, horizontal_vec_4ch, assume_normalized = True, look = normal, units = 'deg')

        if disp:
            norml = np.linalg.norm(self.ideal_top_pts_2ch[0] - self.ideal_top_pts_2ch[1])
            baseline_2ch_act = get_arrow_act(self.ideal_top_pts_2ch[0], self.ideal_top_pts_2ch[0] + norml*baseline_2ch, 0.05, 5)
            horizontal_vec_2ch_act = get_arrow_act(self.ideal_top_pts_2ch[0], self.ideal_top_pts_2ch[0] + norml*horizontal_vec_2ch, 0.05, 5)
            baseline_4ch_act = get_arrow_act(self.ideal_top_pts_4ch[0], self.ideal_top_pts_4ch[0] + norml*baseline_4ch, 0.05, 5)
            horizontal_vec_4ch_act = get_arrow_act(self.ideal_top_pts_4ch[0], self.ideal_top_pts_4ch[0] + norml*horizontal_vec_4ch, 0.05, 5)
            self.endoActor.GetProperty().SetOpacity(0.6)

            renderers = []
            viewPorts = split_window(1,2)
            ren = vtk.vtkRenderer()
            ren.SetViewport(viewPorts[0,:])
            ren.AddActor(self.endoActor)
            ren.AddActor(baseline_2ch_act)
            ren.AddActor(horizontal_vec_2ch_act)
            renderers.append(ren)

            ren = vtk.vtkRenderer()
            ren.SetViewport(viewPorts[1,:])
            ren.AddActor(self.endoActor)
            ren.AddActor(baseline_4ch_act)
            ren.AddActor(horizontal_vec_4ch_act)
            renderers.append(ren)

            vtk_multiple_renderers(renderers, 800, 800)


        self.a4c_sla = slanting_angle_4ch
        self.a2c_sla = slanting_angle_2ch

        return slanting_angle_2ch, slanting_angle_4ch



    ## metric specific functions
    def get_average_from_landmarks(self):
        a4c_a = self.ideal_horiz_4ch_a
        a4c_b = self.ideal_horiz_4ch_b
        a2c60_a = self.ideal_horiz_2ch_a
        a2c60_b = self.ideal_horiz_2ch_b

        av_pts = []
        for i in range(len(a4c_a)):
            av_pt = (a4c_a[i] + a4c_b[i] + a2c60_a[i] + a2c60_b[i]) / 4.0
            av_pts.append(av_pt)

        # ren = vtk.vtkRenderer()
        # p1a = include_points(a4c_a, len())

        return np.asarray(av_pts)

    def get_TBC_slices(self, poly, num_slices, endo_apex, C, av_pts, color_slice=None, inner_rim_poly=None, push_rv_dir_down=0.2, disp=0):
        """
        cuts poly by planes in plane_pts
        displays it by slice in multiple renderer windows

        plane_pts_arr : [ [pt1, pt2, pt3]_i,
                       [pt1, pt2, pt3]_i+1,
                       ...
                       ]

        enclosed_always: 1 --> makes sure that the loops are always closed, This is essential for function "get_singed_ang_diffs" otherwise cast ray wont intersect with non-closed ellipse perimeter
                        : 0 --> when you want to visualize the slices for each method better

        push_rv_dir_down = percentage of how much to move the rv dir com downwards for plotting rv line dir
        """
        M, N = get_factors(num_slices)
        renderers = []
        viewports = split_window(M, N)
        poly_act = get_actor_from_polydata(poly, (1, 0, 0))

        slice_polys = []
        slice_actors = []

        midline_vec = endo_apex - C
        mag_midline = np.linalg.norm(midline_vec)

        # if you want to orient the slices as SBR (bare):
        midline_uvec = normalize(midline_vec)
        pl = midline_uvec
        pl_orig = self.C
        acts = []
        inner_act = None

        for i in range(num_slices):
            if (inner_rim_poly != None) and (i == 0):
                inner_rim_com = get_com_from_pd(inner_rim_poly)
                pl_orig = inner_rim_com

                sorted_pts, _ = sort_points_by_circular(convert_poly_to_numpy(inner_rim_poly))
                new_inner_poly, inner_act = get_line_actor(sorted_pts, 2, color_slice, circular=1)
                inner_act.GetProperty().SetLineWidth(5)
                textActor = get_text_actor('slice ' + str(i), [10, 40], 18, (1, 1, 1))

                slice_polys.append(new_inner_poly)
                slice_actors.append(inner_act)

                ren = vtk.vtkRenderer()
                ren.SetViewport(viewports[i])
                ren.AddActor(textActor)
                ren.AddActor(poly_act)
                ren.AddActor(inner_act)
                renderers.append(ren)
                # vtk_show(ren)

                continue

            slice_poly, sliceActor = get_edges_strips(poly, pl, pl_orig, None, color_slice)

            if i > 0:  # for the second slice, it is important to check if the circle is closed or not

                # if closed, all points will appear exactly twice in line connectivity
                # if not closed, there will be 2 points or more that appear only once in line connectivity
                line_con = get_line_connectivity_poly(slice_poly)
                uniq, counts = np.unique(np.squeeze(line_con), return_counts=True)

                perc_transl = 0.005
                counter = 0
                max_iter = 20
                pl_orig2 = av_pts[i]

                # later addition:
                slice_poly, sliceActor = get_edges_strips(poly, pl, pl_orig2, None, color_slice)

                while (0 in counts) or (1 in counts) or (counter > max_iter):
                    # print('slice {} not closed. Translate slice down a bit..'.format(i))
                    tol_vec = perc_transl * mag_midline * midline_uvec  # move perc_transl % down midline
                    pl_orig2 = pl_orig2 + tol_vec
                    slice_poly, sliceActor = get_edges_strips(poly, pl, pl_orig2, None, color_slice)

                    line_con = get_line_connectivity_poly(slice_poly)
                    uniq, counts = np.unique(np.squeeze(line_con), return_counts=True)

                    counter = counter + 1
                    perc_transl = perc_transl + 0.005


                    # orig_Cact = include_points(pl_orig2, 1, 1, (0,1,1))
                    # ren = vtk.vtkRenderer()
                    # ren.AddActor(sliceActor)
                    # ren.AddActor(orig_Cact)
                    # ren.AddActor(poly_act)
                    # # vtk_show(ren)

            textActor = get_text_actor('slice ' + str(i), [10, 40], 18, (1, 1, 1))
            ren = vtk.vtkRenderer()
            ren.SetViewport(viewports[i])
            ren.AddActor(textActor)
            ren.AddActor(poly_act)
            ren.AddActor(sliceActor)
            renderers.append(ren)
            # vtk_show(ren)

            slice_polys.append(slice_poly)
            slice_actors.append(sliceActor)

            acts.append(sliceActor)
            if inner_act != None:
                acts.append(inner_act)

        # assembly = get_assembly_from_actors(acts)
        # ren = vtk.vtkRenderer()
        # ren.AddActor(poly_act)
        # ren.AddActor(assembly)
        # vtk_show(ren)

        # now remember, replace the first slice point by inner rim poly

        # if 0:
        #     vtk_multiple_renderers(renderers, 600, 600)

        if disp:
            assembly = get_assembly_from_actors(slice_actors)
            midline_vec = self.endo_apex_node - self.C
            rv_dir_act = get_line_act2(50.0 * self.rv_dir, self.C + push_rv_dir_down*midline_vec, (0, 0, 0))
            # rv_dir_act = get_line_act2(50.0 * self.rv_dir, self.com, (0, 0, 0))
            rv_dir_act.GetProperty().SetLineWidth(5)
            ren = vtk.vtkRenderer()
            ren.SetBackground(1, 1, 1)
            # try:
            #     ren.AddActor(self.orig_planeActor[0])
            #     ren.AddActor(self.orig_planeActor[1])
            # except:
            #     ren.AddActor(self.orig_planeActor[0][0])
            #     ren.AddActor(self.orig_planeActor[1][0])
            ren.AddActor(assembly)
            ren.AddActor(rv_dir_act)
            ren.AddActor(self.endoActor)
            vtk_show(ren)

        return renderers, slice_polys, slice_actors

    def get_BOD_slices(self, plane_pts_arr, poly, midline_vec, color_slice=None, inner_rim_poly=None, push_rv_dir_down=0.2, disp=0):
        """
        cuts poly by planes in plane_pts
        displays it by slice in multiple renderer windows

        plane_pts_arr : [ [pt1, pt2, pt3]_i,
                       [pt1, pt2, pt3]_i+1,
                       ...
                       ]
        """
        num_slices = len(plane_pts_arr)
        M, N = get_factors(num_slices)
        renderers = []
        viewports = split_window(M, N)
        poly_act = get_actor_from_polydata(poly, (1, 0, 0))

        slice_polys = []
        slice_actors = []

        for i in range(num_slices):
            if (inner_rim_poly != None) and (i == 0):
                sorted_pts, _ = sort_points_by_circular(convert_poly_to_numpy(inner_rim_poly))
                new_inner_poly, inner_act = get_line_actor(sorted_pts, 2, color_slice, circular=1)
                inner_act.GetProperty().SetLineWidth(5)

                textActor = get_text_actor('slice ' + str(i), [10, 40], 18, (1, 1, 1))

                slice_polys.append(new_inner_poly)
                slice_actors.append(inner_act)

                # ren = vtk.vtkRenderer()
                # ren.SetViewport(viewports[i])
                # ren.AddActor(textActor)
                # ren.AddActor(poly_act)
                # ren.AddActor(inner_act)
                # renderers.append(ren)
                # vtk_show(ren)

                continue

            ppts = plane_pts_arr[i]
            pl = get_plane_source(ppts[0], ppts[1], ppts[2])
            slice_poly, sliceActor = get_edges_strips(poly, pl, None, None, color_slice)

            if i > 0:  # for the second slice, it is important to check if the circle is closed or not
                # if closed, all points will appear exactly twice in line connectivity
                # if not closed, there will be 2 points or more that appear only once in line connectivity
                line_con = get_line_connectivity_poly(slice_poly)
                uniq, counts = np.unique(np.squeeze(line_con), return_counts=True)

                perc_transl = 0.005
                counter = 0
                max_iter = 20
                while (0 in counts) or (1 in counts) or (counter > max_iter):
                    print('Second circle slice not closed. Translate slice down a bit..')
                    midline_uvec = normalize(midline_vec)
                    mag_midline = np.linalg.norm(midline_vec)
                    tol_vec = perc_transl * mag_midline * midline_uvec  # move perc_transl % down midline

                    pl = get_plane_source(ppts[0] + tol_vec, ppts[1] + tol_vec, ppts[2] + tol_vec)
                    slice_poly, sliceActor = get_edges_strips(poly, pl, None, None, color_slice)
                    sliceActor.GetProperty().SetLineWidth(5)

                    line_con = get_line_connectivity_poly(slice_poly)
                    uniq, counts = np.unique(np.squeeze(line_con), return_counts=True)

                    perc_transl = perc_transl + 0.005
                    counter = counter + 1

                    # ren = vtk.vtkRenderer()
                    # ren.AddActor(sliceActor)
                    # ren.AddActor(poly_act)
                    # pptsact = include_points(ppts, 3, 1, (1, 0, 1))
                    # ren.AddActor(pptsact)
                    # vtk_show(ren)

            textActor = get_text_actor('slice ' + str(i), [10, 40], 18, (1, 1, 1))
            ren = vtk.vtkRenderer()
            ren.SetViewport(viewports[i])
            ren.AddActor(textActor)
            ren.AddActor(poly_act)
            ren.AddActor(sliceActor)
            renderers.append(ren)

            slice_polys.append(slice_poly)
            slice_actors.append(sliceActor)

        # now remember, replace the first slice point by inner rim poly

        if 0:  # view all in multiple windows
            vtk_multiple_renderers(renderers, 600, 600)

        if disp:
            assembly = get_assembly_from_actors(slice_actors)
            midline_vec = self.endo_apex_node - self.C
            rv_dir_act = get_line_act2(50.0 * self.rv_dir, self.C + push_rv_dir_down*midline_vec, (0, 0, 0))
            # rv_dir_act = get_line_act2(50.0 * self.rv_dir, self.com, (0, 0, 0))
            rv_dir_act.GetProperty().SetLineWidth(5)
            ren = vtk.vtkRenderer()
            ren.SetBackground(1, 1, 1)
            # try:
            #     ren.AddActor(self.orig_planeActor[0])
            #     ren.AddActor(self.orig_planeActor[1])
            # except:
            #     ren.AddActor(self.orig_planeActor[0][0])
            #     ren.AddActor(self.orig_planeActor[1][0])
            ren.AddActor(assembly)
            ren.AddActor(rv_dir_act)
            ren.AddActor(self.endoActor)
            vtk_show(ren)

        return renderers, slice_polys, slice_actors

    def get_SBR_slices(self, perc_transl, color_slice=None, disp=0):
        """
        Using the horizontal landmarks extracted, slice the endo poly
        with planes formed with each 4-pair of landmarks.

        perc_transl : how much should the top most landmarks be translated
        downwards to enable smooth slice acquisition.
                    : it is a scale of midline length magnitude.
            i.e. if perc_transl = 0.1, the points are translated downwards
                by 10% of the midline length.

        Note: this is used for measure_ellipticity
            and should only be ran after computing self.horiz_xch_a/b points
        Return:
            list of slice polys and actors
        """
        h2a = self.ideal_horiz_2ch_a
        h2b = self.ideal_horiz_2ch_b
        h4a = self.ideal_horiz_4ch_a
        h4b = self.ideal_horiz_4ch_b

        slices_polys = []
        slices_actors = []

        num_pts = len(h2a)

        if self.typ in [0,1]: # if either basal or ge method
            midline_uvec = normalize(self.endo_apex_node - self.C)
            mag_midline = np.linalg.norm(self.endo_apex_node - self.C)
            tol_vec = perc_transl*mag_midline*midline_uvec # move perc_transl % down midline

            sort_4_0, sort_2_0 = sort_landmark_pairs_along_midline([h4a[0], h4b[0]],
                                                                [h2a[0], h2b[0]],
                                                                self.C,
                                                                self.endo_apex_node)
            pp1 = np.asarray(sort_2_0[0]) + tol_vec
            pp2 = np.asarray(sort_2_0[-1]) + tol_vec
            pp3 = np.asarray(sort_4_0[0]) + tol_vec

            center_pts = np.mean([pp1, pp2, pp3], axis=0)
            pl = normalize(find_plane_eq(pp1, pp2, pp3))[:3]
            slice_poly_0, sliceActor_0 = get_edges_strips(self.endo_poly, pl, center_pts, None, color_slice)
            sliceActor_0, sliceActor_0 = close_polyline(slice_poly_0, sliceActor_0)
            sliceActor_0.GetProperty().SetLineWidth(5)
            sliceActor_0.GetProperty().SetColor(color_slice)

            slices_polys.append(slice_poly_0)
            slices_actors.append(sliceActor_0)

            starting_idx = 1

            disp_0 = 0
            if disp_0:
                p2a = include_points(list(h2a[0]), 1, self.needle, (0, 1, 0))
                p2b = include_points(list(h2b[0]), 1, self.needle, (0, 1, 0))
                p4a = include_points(list(h4a[0]), 1, self.needle, (0, 0, 1))
                p4b = include_points(list(h4b[0]), 1, self.needle, (0, 0, 1))

                ren = vtk.vtkRenderer()
                ren.AddActor(sliceActor_0)
                ren.AddActor(self.endoActor)
                ren.AddActor(p2a)
                ren.AddActor(p2b)
                ren.AddActor(p4a)
                ren.AddActor(p4b)
                vtk_show(ren)

        else:
            # if bare method, do first slice since it's truncated.
            starting_idx = 0

        ## Iterate through remaining layers.
        for i in range(starting_idx, num_pts): # start from 1 instead of 0
            if self.a2c_sla > self.a4c_sla:
                pp1 = h2a[i]
                pp2 = h2b[i]
                pp3 = pp1 + normalize(h4a[i] - h4b[i])
            else:
                pp1 = h4a[i]
                pp2 = h4b[i]
                pp3 = pp1 + normalize(h2a[i] - h2b[i])

            # sort_4i, sort_2i = sort_landmark_pairs_along_midline([h4a[i], h4b[i]],
            #                                                 [h2a[i], h2b[i]],
            #                                                 self.C,
            #                                                 self.endo_apex_node)
            # # 2. get 2 highest and 1 lowest
            # pp1 = np.asarray(sort_2i[0])
            # pp2 = np.asarray(sort_4i[-1])
            # pp3 = np.asarray(sort_4i[0])

            pl = get_plane_source(pp3, pp1, pp2)

            slice_poly, sliceActor = get_edges_strips(self.endo_poly, pl, None, None, None)
            slice_poly, sliceActor = close_polyline(slice_poly, sliceActor)
            sliceActor.GetProperty().SetColor(color_slice)
            sliceActor.GetProperty().SetLineWidth(5)

            slices_polys.append(slice_poly)
            slices_actors.append(sliceActor)

            disp_iter = 0
            if disp_iter:
                p2a = include_points(list(h2a[i]), 1, self.needle, (0,1,0))
                p2b = include_points(list(h2b[i]), 1, self.needle, (0,1,0))
                p4a = include_points(list(h4a[i]), 1, self.needle, (0,0,1))
                p4b = include_points(list(h4b[i]), 1, self.needle, (0,0,1))
                ren = vtk.vtkRenderer()
                ren.AddActor(sliceActor)
                ren.AddActor(self.endoActor)
                ren.AddActor(p2a)
                ren.AddActor(p2b)
                ren.AddActor(p4a)
                ren.AddActor(p4b)
                vtk_show(ren)


        if disp:
            assembly = get_assembly_from_actors(slices_actors)
            midline_vec = self.endo_apex_node - self.C
            ren = vtk.vtkRenderer()
            self.endoActor.GetProperty().SetColor(1,0,0)
            ren.SetBackground(1, 1, 1)
            push_rv_dir_down_factor = 0.4 # ADJUST THIS IF BLACK ARROW IS TILTED
            arrow_origin = self.C + push_rv_dir_down_factor*midline_vec
            arrow_end = 80.0 * self.rv_dir
            # arrow_end = 80.0 * self.rv_dir + 0.3*midline_vec # use this for UKBB
            arrow_act = draw_arrow(start=arrow_origin, end=arrow_end, scale=50,
                                   tip_length = 0.05, tip_radius=0.05, tip_resolution=1000,
                                   shaft_radius=0.01, shaft_resolution=1000)
            arrow_act.GetProperty().SetColor(0,0,0)
            # try:
            #     ren.AddActor(self.orig_planeActor[0])
            #     ren.AddActor(self.orig_planeActor[1])
            # except:
            #     ren.AddActor(self.orig_planeActor[0][0])
            #     ren.AddActor(self.orig_planeActor[1][0])
            # ren.AddActor(assembly)
            ren.AddActor(arrow_act)
            ren.AddActor(self.endoActor)
            vtk_show(ren)


        return slices_polys, slices_actors

    def get_Am_and_mu(self, inner_rim_poly = None, skip_first_slice = False, disp=0):
        """
        Computes angles from pca_major_axis to a4c.
        Computes angles from pca minor axis to a2c.
        Measure eccentricity governing the cross-sectional slice. Not by the A4C/A2C diameters.

        conf: how to define Am orientation angle
            : 'closest' : means define Am orientation angle as the angle of A4C/A2C with the closest axes.
            : 'A4C to major' : means define Am as the angle between A4C and the true major axis
            : 'A2C to minor' : define Am as the angle between A2C and the true minor axis.

        When choosing either A4C_a or a4c_b, choose the one closest to the RV dir.
        Seems likes a4c_a is the one closest rv dir always! (which makes it easier for us)
        """

        dict_path = self.landmark_path + '/{}_data_dict.pkl'.format(self.case_num)
        if self.opts.exclude_landmarks:
            assert os.path.exists(dict_path), "Must extract landmarks first (do not include --exclude_landmarks"
            data_dict = pickle.load(open(dict_path, 'rb'))
            self.setVar(data_dict)  # update attributes
        else:
            data_dict = self.__dict__.copy()

        a4c_a = self.ideal_horiz_4ch_a
        a4c_b = self.ideal_horiz_4ch_b
        a2c60_a = self.ideal_horiz_2ch_a
        a2c60_b = self.ideal_horiz_2ch_b
        av_pts = self.get_average_from_landmarks()
        # a2c60_c0 = get_points_in_line(a2c60_a[0], a2c60_b[0], weight=0.55)
        # plane_pts_arr = list(zip(a4c_a, a4c_b, np.vstack((a2c60_c0, a2c60_a[1:]))))

        rens, slice_polys, slice_actors = self.get_TBC_slices(self.endo_poly,
                                                            len(a4c_a),
                                                            self.endo_apex_node,
                                                            self.C, av_pts,
                                                            color_slice=(0, 0.5, 0.5),
                                                            inner_rim_poly = inner_rim_poly,
                                                            disp = disp)

        # slice_polys, slice_actors = self.get_landmark_slices(0.01, disp)
        slices_pts = convert_polys_to_pts_list(slice_polys)
        inner_rim_pts = convert_poly_to_numpy(self.inner_rim_poly)
        # for the first slice (inner rim points), make sure you project them all into a plane
        midline = list(normalize(self.endo_apex_node - self.C))
        inner_rim_pts_proj = project_3d_points_to_plane(inner_rim_pts,
                                                        a4c_a[0],
                                                        a4c_b[0],
                                                        a2c60_a[0], normal=midline)
        slices_pts[0] = inner_rim_pts_proj
        flat_inner_poly = vtk.vtkPolyData()
        flat_inner_poly.SetPoints(convert_numpy_to_points(inner_rim_pts_proj))
        flat_inner_poly.SetLines(slice_polys[0].GetLines())
        slice_polys[0] = flat_inner_poly

        angs_a4c_closest = []
        angs_a2c_closest = []
        angs_a4c_to_major = []
        angs_a2c_to_minor = []
        eccents = []

        for i, slice_poly in enumerate(slice_polys):
            if (skip_first_slice == True) and (i==0):
                continue

            evec1, evec2 = get_pcs(slice_poly)
            approx_rad = 1.5 * (np.linalg.norm(a4c_a[i] - a4c_b[i]) / 2.0)
            if self.typ == 2: # for bare, truncated basal disks mean we need larger radius
                approx_rad = 10.0*approx_rad

            major_pts, minor_pts = get_major_minor_axis_from_evecs(evec1, evec2,
                                                        approx_rad, slice_poly, tol=100,
                                                        disp=0)

            if np.linalg.norm(major_pts[0] - major_pts[1]) < np.linalg.norm(minor_pts[0] - minor_pts[1]):
                temp = major_pts
                major_pts = minor_pts
                minor_pts = temp

            ## Now calculate eccentricty = major diam / minor diam
            eccent = np.linalg.norm(major_pts[1] - major_pts[0]) / np.linalg.norm(minor_pts[1] - minor_pts[0])
            eccents.append(eccent)

            # determine normal to calculate angles using new angle function
            plane_pts = [slices_pts[i][0], slices_pts[i][1], slices_pts[i][2]]
            Vn = find_plane_eq(plane_pts[0], plane_pts[1], plane_pts[2])[:3]

            # make sure Vn points upwards!
            midline = normalize(self.C - self.endo_apex_node) # this vector goes UP!
            if np.dot(Vn, midline) < 0.0: Vn = -Vn
            Vn = normalize(Vn)

            ## now make sure that choose the correct a4c_a or a4c_b (the one that is closest to the RV dir!)
            # but before this, we need to project the points onto the plane
            com = np.mean(slices_pts[i], axis=0)
            a4c_a_proj = project_3d_points_to_plane(a4c_a[i], plane_pts[0], plane_pts[1], plane_pts[2])
            a4c_b_proj = project_3d_points_to_plane(a4c_b[i], plane_pts[0], plane_pts[1], plane_pts[2])
            a2c_a_proj = project_3d_points_to_plane(a2c60_a[i], plane_pts[0], plane_pts[1], plane_pts[2])
            a2c_b_proj = project_3d_points_to_plane(a2c60_b[i], plane_pts[0], plane_pts[1], plane_pts[2])
            half_horiz_dist = np.linalg.norm(a4c_a_proj - a4c_b_proj)/2.0
            rv_dir_proj = project_3d_points_to_plane(half_horiz_dist*self.rv_dir + com, plane_pts[0], plane_pts[1], plane_pts[2])

            # choose correct a4c now
            if np.linalg.norm(a4c_a_proj - rv_dir_proj) < np.linalg.norm(a4c_b_proj - rv_dir_proj): # if a4c_A is closer to rv dir
                a4c_rv =  a4c_a_proj
            else:
                a4c_rv = a4c_b_proj

            # dispall([a4c_rv[0], rv_dir_proj[0], a4c_a_proj[0], a4c_b_proj[0], a2c_a_proj[0], a2c_b_proj[0], self.endoActor],
            #         [(1,1,0), (1,1,1), (0,1,0), (0.5, 1, 0.5), (0,0,1), (0.5, 0.5, 1), (0.5, 0.5, 0.5)],
            #         [3, 2, 2, 2, 2, 2, None])

            a4c_uvec = np.squeeze(a4c_rv - com)
            a2c_uvec = np.squeeze(a2c_a_proj - com)

            # CONF = 'closest'
            #### compute which pca major points are closest to a4ca
            ds_a4c = [np.linalg.norm(a4c_rv - major_pts[0]),
                  np.linalg.norm(a4c_rv - major_pts[1]),
                  np.linalg.norm(a4c_rv - minor_pts[0]),
                  np.linalg.norm(a4c_rv - minor_pts[1])]
            argmin_a4c = np.argmin(ds_a4c)
            if (argmin_a4c==0) or (argmin_a4c==1): # if a4c closest to the major pts
                # print('MAJOR')
                pca_major_uvec = major_pts[argmin_a4c] - com
                ds_a2c = [np.linalg.norm(a2c60_a[i] - minor_pts[0]),
                          np.linalg.norm(a2c60_a[i] - minor_pts[1])]
                argmin_a2c = np.argmin(ds_a2c)
                pca_minor_uvec = minor_pts[argmin_a2c] - com
            elif (argmin_a4c==2) or (argmin_a4c==3): #if a4c closest to the minor pts,
                # print('minor')
                pca_major_uvec = minor_pts[argmin_a4c-2] - com # subtract 2 to make sure it is 0 or 1
                ds_a2c = [np.linalg.norm(a2c60_a[i] - major_pts[0]),
                          np.linalg.norm(a2c60_a[i] - major_pts[1])]
                argmin_a2c = np.argmin(ds_a2c)
                pca_minor_uvec = major_pts[argmin_a2c] - com
            else:
                print('argmin_a4c should be integer value of 0, 1, 2, or 3')

            # compute major vec
            ang_a4c_cl = math.degrees(get_signed_angle_3d(Vn, normalize(pca_major_uvec), normalize(a4c_uvec)))
            angs_a4c_closest.append(ang_a4c_cl)
            # print('ang_a4c =', ang_a4c)

            # compute minor vec
            ang_a2c_cl = math.degrees(get_signed_angle_3d(Vn, normalize(pca_minor_uvec), normalize(a2c_uvec)))
            angs_a2c_closest.append(ang_a2c_cl)
            # print('ang_a2c =', ang_a2c)

            # CONF = 'a4c to major'
            ds = [np.linalg.norm(a4c_rv - major_pts[0]),
                  np.linalg.norm(a4c_rv - major_pts[1])]
            argmin_a4c = np.argmin(ds)
            pca_major_uvec = major_pts[argmin_a4c] - com
            ang_a4c_major = math.degrees(get_signed_angle_3d(Vn, normalize(pca_major_uvec), normalize(a4c_uvec)))
            angs_a4c_to_major.append(ang_a4c_major)

            # CONF = 'a2c to minor'
            ds = [np.linalg.norm(a2c60_a[i] - minor_pts[0]),
                  np.linalg.norm(a2c60_a[i] - minor_pts[1])]
            argmin_a2c = np.argmin(ds)
            pca_minor_uvec = minor_pts[argmin_a2c] - com
            ang_a2c_minor = math.degrees(get_signed_angle_3d(Vn, normalize(pca_minor_uvec), normalize(a2c_uvec)))
            angs_a2c_to_minor.append(ang_a2c_minor)

            if disp:
                if i==0:
                    print('ang_a4c_major =', ang_a4c_major)
                    a4c_line = get_line_act2(com , a4c_rv[i], (0, 1, 0)) # green = a4c
                    a2c60_line = get_line_act2(com, a2c60_a[i], (0, 0, 1)) # blue = a2c60
                    pca_major_line = get_line_act2(com, pca_major_uvec + com, (1, 0, 0)) # red = pca major axis
                    pca_minorline = get_line_act2(com, pca_minor_uvec + com, (1, 1, 0)) # yellow = pca minor axis
                    pt_rv_dir_act = get_line_act2(com, 50.0*self.rv_dir + com, (1, 1, 1)) # white = rv dir

                    self.endoActor.GetProperty().SetOpacity(1.0)
                    ren = vtk.vtkRenderer()
                    ren.AddActor(self.endoActor)
                    ren.AddActor(a4c_line)
                    ren.AddActor(a2c60_line)
                    ren.AddActor(pca_major_line)
                    ren.AddActor(pca_minorline)
                    ren.AddActor(pt_rv_dir_act)
                    vtk_show(ren)


        # measure simpson's disks eccentricty
        simpson_ellipse_ecc = self.measure_eccentricity(method='TBC', skip_first_slice=skip_first_slice, disp=0)
        metric_dict = dict(angs_a4c_closest=angs_a4c_closest,
                        angs_a2c_closest=angs_a2c_closest,
                        angs_a4c_to_major=angs_a4c_to_major,
                        angs_a2c_to_minor = angs_a2c_to_minor,
                        anatomical_ecc = eccents,
                        simpson_ellipse_ecc = simpson_ellipse_ecc)

        # update the current dictionary to save again
        data_dict.update(metric_dict)
        self.setVar(data_dict)  # update attributes

        return angs_a4c_to_major

    def measure_eccentricity(self, method='TBC', skip_first_slice=False, disp=0):
        """
        Simple measure of eccentricity (A4C horiz distance vs A2C horiz distance)
        There are many ways that A4C and A2C lines are generated (BOD, TBC, SBR)

        Args:
            method: indicate which way you want A4C and A2C lines to be generated
                    If 'basal': the A4C and A2C lines will be slanted parallel to base
                    If 'GE': the A4C and A2C lines will be same as SBR (perpendicular to midline) but top most line will be like basal lines.
                    If 'bare': the A4C and A2C lines are perpendicular to midline
                Default : 'basal'

        Returns:
            eccents: list of eccentricities
                    : if eccent < 1.0, then A2C diameter is longer
                    : if eccent > 1.0, then A4C diameter is longer
        """
        root_path = os.path.dirname(os.path.dirname(self.landmark_path)) # go back two folders
        full_path = os.path.join(root_path, '{}/{}/{}_data_dict.pkl'.format(method, self.ang_sep, self.case_num))
        if not os.path.exists(full_path):
            print('Data dict path {} , does not exist..'.format(full_path))
            sys.exit()

        # print('Measuring eccentricity from data dict {} ..'.format(d))
        data_dict = pickle.load(open(full_path, 'rb'))
        horiz_2ch_a = data_dict['ideal_horiz_2ch_a']
        horiz_2ch_b = data_dict['ideal_horiz_2ch_b']
        horiz_4ch_a = data_dict['ideal_horiz_4ch_a']
        horiz_4ch_b = data_dict['ideal_horiz_4ch_b']

        # # sanity check that correct landmarks are being generated based on method type.
        if disp:
            dispall([horiz_2ch_a, horiz_2ch_b, horiz_4ch_a, horiz_4ch_b, self.endoActor], [(0,0,1), (0,0,1), (0,1,0), (0,1,0), (1,0,0)], [2, 2, 2, 2, None])


        # 2. Measure diameters
        horiz_2ch_dists, horiz_4ch_dists = compute_horizontal_distances(horiz_2ch_a, horiz_2ch_b,horiz_4ch_a, horiz_4ch_b)

        # 3. Measure eccents
        eccents = []

        for i in range(len(horiz_2ch_dists)):
            if (skip_first_slice) and (i==0):
                continue
            ecc = horiz_2ch_dists[i]/horiz_4ch_dists[i] # APLAX / A4C
            eccents.append(ecc)

        eccents = np.squeeze(eccents)
        return eccents



    ## eccentricity profile function
    def plot_ellipticity_profile_slices(self, disp=0, display_slices=0):
        """
        Based on: "Analysing error of fit functions for ellipses" by Paul L. Rosin

        https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/conic3-bmvc.pdf

        1. For each 4-pair landmarks, form plane and slice cut the mesh.

        2. Convert slice cut onto 2-D plane where x-axis is the major axis
            and the y-axis is the minor axis.

        When convert slice points and 4-pair lnmks to xy-plane, make sure
        that both of them are translated to the origin from the
        com of the 4-pair lnmks (and NOT the com of the slice points).

        Otherwise you get incorrect euclidean distances!

        N.B. Must run this AFTER
            1. assigning self.ideal_horiz_xch_a/b points!
            2. basal slanting angle estimation


        """

        dict_path = self.landmark_path + '/{}_data_dict.pkl'.format(self.case_num)
        if self.opts.exclude_landmarks:
            assert os.path.exists(dict_path), "Must extract landmarks first (do not include --exclude_landmarks"
            data_dict = pickle.load(open(dict_path, 'rb'))
            self.setVar(data_dict)  # update attributes
        else:
            data_dict = self.__dict__.copy()

        # get landmark slices
        a4c_a = self.ideal_horiz_4ch_a
        a4c_b = self.ideal_horiz_4ch_b
        a2c_a = self.ideal_horiz_2ch_a
        a2c_b = self.ideal_horiz_2ch_b
        a4c_to_major = self.angs_a4c_to_major

        av_pts = self.get_average_from_landmarks()

        av_pts = self.get_average_from_landmarks()
        plane_pts_arr = list(zip(a4c_a, a4c_b, np.vstack((a2c_a, a2c_b[1:]))))
        self.endoActor.GetProperty().SetOpacity(1)

        # if you want to visualize original slices (i.e. sometimes slices are not always enclosed ( like in bare rule basal slices)
        color_slice = (0.1,0.1,0.1)
        if self.typ == 0:
            _, slices_polys, slices_actors = self.get_BOD_slices(plane_pts_arr, self.endo_poly,
                                                                   self.endo_apex_node - self.C,
                                                                   color_slice = color_slice,
                                                                   inner_rim_poly=self.inner_rim_poly,
                                                                   disp=display_slices)  #self.inner_rim_poly
            # slices_polys, slices_actors = self.get_SBR_slices(0.02, color_slice=color_slice, disp=display_slices)
        elif self.typ == 1:
            _, slices_polys, slices_actors = self.get_TBC_slices(self.endo_poly, len(a4c_a), self.endo_apex_node, self.C, av_pts, color_slice = color_slice,
                                                                 inner_rim_poly=self.inner_rim_poly,
                                                                 disp=display_slices) #
        else: # for bare you can leave enclosed ellipses since that's inherent of the method
            slices_polys, slices_actors = self.get_SBR_slices(0.02, color_slice = color_slice,
                                                              disp=display_slices)

        eccents = []

        rows = 4
        cols = 5
        # fig = plt.figure()
        # gs1 = gridspec.GridSpec(rows, cols)
        # gs1.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.
        # ax.set_aspect('equal')
        # ax.axis("off")

        fig, ax = plt.subplots(1,1)
        ax.set_aspect('equal')
        ax.axis("off")

        plt.gca().invert_yaxis()

        # for some reason, first slice, the rv dir is not posisioned correctly

        for i in range(len(slices_polys)):
            lnmks = [self.ideal_horiz_4ch_a[i],
                    self.ideal_horiz_4ch_b[i],
                    self.ideal_horiz_2ch_a[i],
                    self.ideal_horiz_2ch_b[i]]
            com = np.mean(lnmks[:4], axis=0)
            av_horiz_radius = np.linalg.norm(lnmks[0] - lnmks[1])/2.0
            # rv_dir = (av_horiz_radius * self.rv_dir)
            rv_dir = lnmks[0] + 0.5*normalize(self.rv_dir)
            lnmks.append(rv_dir)

            numpy_slice_pts = numpy_support.vtk_to_numpy(slices_polys[i].GetPoints().GetData())
            all_pts = np.vstack((lnmks, numpy_slice_pts))
            all_pts_2d, _  = project_onto_xy_plane(all_pts, com)
            landmarks_new = all_pts_2d[0:5]
            slices_xy = all_pts_2d[5:-1]
            com_2d = np.mean(landmarks_new[:4], axis=0)

            rv_dir_ang = compute_angle_between_2d(landmarks_new[0,:2] - landmarks_new[1,:2], [-1,0]) # from a4c dir to x-axis
            a4c_rv_dir_ang = get_signed_angle(landmarks_new[4,:2]  - com_2d[:2], landmarks_new[0,:2] - com_2d[:2]) # from com --> rv dir vector to x axis
            R = np.array([[np.cos(rv_dir_ang), -np.sin(rv_dir_ang)],
                            [np.sin(rv_dir_ang), np.cos(rv_dir_ang)]])

            slices_rot = np.dot(R, slices_xy[:,:2].T).T
            landmarks_rot = np.dot(R,landmarks_new[:,:2].T).T
            #for rv dir point, just add it manually using 23 ang from a4c
            custom_rot = math.radians(-19)
            R2 = np.array([[np.cos(custom_rot), -np.sin(custom_rot)],
                            [np.sin(custom_rot), np.cos(custom_rot)]])

            landmarks_rot[-1] = landmarks_rot[0] + np.array([0,-2])
            landmarks_rot[-1] = np.dot(R2, landmarks_rot[-1]).T

            com_rot = np.mean(landmarks_rot, axis=0)[:2]

            # all_pts_2d = rotate_clockwise_align(all_pts_2d,
            #                        normalize(landmarks_new[0] - landmarks_new[1]),
            #                        np.array([1,0]))
            # landmarks_rot = all_pts_2d[0:5]
            # slices_rot = all_pts_2d[5:-1]
            # com_rot = np.mean(landmarks_new[:4], axis=0)

            # if (np.abs(a4c_rv_dir_ang) < 90.0) and (np.sign(a4c_rv_dir_ang) == -1.0): # if smaller than 90, and negative
            #     landmarks_rot[0] = invert_axis_by_values(landmarks_rot[0], x_flip=False, y_flip=True)
            #     landmarks_rot[1] = invert_axis_by_values(landmarks_rot[1], x_flip=False, y_flip=True)
            #     landmarks_rot[2] = invert_axis_by_values(landmarks_rot[2], x_flip=False, y_flip=True)
            #     landmarks_rot[3] = invert_axis_by_values(landmarks_rot[3], x_flip=False, y_flip=True)
            #     slices_rot = invert_axis_by_values(slices_rot, x_flip=False, y_flip=True)

            check = 0
            if check:
                fig, axs = plt.subplots(2,1)
                axs[0].scatter(slices_xy[:,0], slices_xy[:,1], s=0.5, c='k')
                axs[0].scatter(landmarks_new[:2,0], landmarks_new[:2,1], s=10, c='g', label='a4c')
                axs[0].scatter(landmarks_new[2:4,0], landmarks_new[2:4,1], s=10, c='b', label='a2c')
                axs[0].scatter(landmarks_new[4,0], landmarks_new[4,1], s=15, c='r', label='rv_dir')
                axs[1].scatter(slices_rot[:,0], slices_rot[:,1], s=0.5, c='k')
                axs[1].scatter(landmarks_rot[:2, 0], landmarks_rot[:2, 1], s=10, c='g', label='a4c')
                axs[1].scatter(landmarks_rot[2:4, 0], landmarks_rot[2:4, 1], s=10, c='b', label='a2c')
                axs[1].scatter(landmarks_rot[4,0], landmarks_rot[4,1], s=15, c='r', label='rv_dir')
                axs[0].set_aspect('equal', adjustable="datalim")
                axs[1].set_aspect('equal', adjustable="datalim")
                axs[0].legend()
                axs[1].legend()
                # if invert_needed:
                #     axs[1].invert_yaxis()
                plt.show()

            if self.typ in [0,1]:
                if i > 0:
                    k = i-1
                    plot_many_ecc_slices(k, ax, slices_rot, landmarks_rot, True, disp)
            else: # if bare rule
                if i < 20:
                    k = i
                    plot_many_ecc_slices(k, ax, slices_rot, landmarks_rot, True, disp)

            # define ellipse equations based on ratio of APLAX to A2C diameters
            A4C_diam = np.linalg.norm(landmarks_rot[0] - landmarks_rot[1]) / 2.0 #
            APLAX_diam = np.linalg.norm(landmarks_rot[2] - landmarks_rot[3]) / 2.0

            eccent = APLAX_diam / A4C_diam
            eccents.append(eccent)

        if disp:
            plt.show()

        plot_eccentricity_profile(eccents, a4c_to_major)





    ## saving specific functions
    def get_pickable_dict(self):
        # Return only the pickable attributes of our Heart class
        state = self.__dict__.copy()
        del state['mesh']
        del state['meshActor']
        del state['mesh_poly']
        del state['triangles']
        del state['points']
        del state['endo_poly']
        del state['epi_poly']
        del state['inner_rim_poly']
        del state['outer_rim_poly']
        del state['endoActor']
        del state['epiActor']
        del state['irActor']
        del state['orActor']
        del state['orig_slice_ren']
        del state['orig_cut_poly_array']
        del state['orig_planeActor']
        del state['lines_act']

        # for key, val in state.items():
        #     print('key {} , val {} '.format(key, val))
        # sys.exit()
        return state

    def save_pickle(self):
        # 4: Save landmarks
        if self.opts.export_pkl:
            d = self.landmark_path
            # print('saving in {}..'.format(d))
            if not os.path.exists(d): # check folder exists
                os.makedirs(d) # else, create new folder

            # print(self.__dict__)
            pickable_dict = self.get_pickable_dict()
            pickle.dump(pickable_dict, open(d + '/{}_data_dict.pkl'.format(self.case_num), 'wb'))

