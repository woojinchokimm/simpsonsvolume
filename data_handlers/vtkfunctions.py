import sys
import vtk as vtk
import numpy as np
import vg
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from vtkmodules.util import numpy_support
import math

"""
Collection of vtk functions commonly used in the heart.py class.
"""

def clean_data(data):
    # data can be reader or polydata
    cpd = vtk.vtkCleanPolyData()
    try:
        cpd.SetInputData(data)
    except:
        cpd.SetInputConnection(data.GetOutputPort())
    cpd.Update()
    return cpd

def vtk_show(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    # create renderer window
    window = vtk.vtkRenderWindow()
    window.PointSmoothingOn() # makes sure points appear as circle not rectangular
    window.AddRenderer(renderer)
    window.SetSize(600, 600)

    # create render window interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()

    # start the display
    window.Render()
    interactor.Start()

def get_actor_from_polydata(pd, color, dataset=False):
    """
    Converts polydata to actor with the specified color.
    color : tuple e.g.:
            (1,0,0) for red.
            0 for no color
    """
    if dataset == False:
        mapper = vtk.vtkPolyDataMapper()
    else:
        mapper = vtk.vtkDataSetMapper()

    mapper.SetInputData(pd)
    mapper.ScalarVisibilityOff() #IF COLOR IS NOT  CHANGING FOR SOME REASOn

    act = vtk.vtkActor()
    act.SetMapper(mapper)

    if color != None:
        act.GetProperty().SetColor(color)

    return act

def get_com_from_pd(pd):
    """
    Gets com from pd
    """
    vtkcom = vtk.vtkCenterOfMass()
    vtkcom.SetInputData(pd)
    vtkcom.Update()
    com = np.asarray(vtkcom.GetCenter())

    return com

def get_axes_actor(scales, translates):

    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Scale(scales[0],scales[1],scales[2])
    transform.Translate(translates[0], translates[1], translates[2])
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(transform)

    return axes

def include_points(points, num_points_to_add, pointsize, color):
    """ function to include list of points on actor for display
    Inputs:
        points (list [] : points to be added)
        num_points
    """

    if num_points_to_add > 1:
        assembly = vtk.vtkAssembly()

        for i in range(num_points_to_add):
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(pointsize)
            if type(points) == vtk.vtkPoints:
                sphere.SetCenter(points.GetPoint(i))
            else:
                sphere.SetCenter(points[i])

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            act = vtk.vtkActor()
            act.SetMapper(mapper)
            if color != None:
                act.GetProperty().SetColor(color)

            assembly.AddPart(act)

        return assembly

    else: # if only adding 1 point
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(pointsize)
        sphere.SetCenter(points)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        act = vtk.vtkActor()
        act.SetMapper(mapper)
        act.GetProperty().SetColor(color)

        return act

def vtk_multiple_renderers(renderers, width, height):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """

    # create renderer window
    window = vtk.vtkRenderWindow()
    for i in range(len(renderers)):
        window.AddRenderer(renderers[i])

    window.SetSize(width, height)

    # create render window interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()

    # start the display
    window.Render()
    interactor.Start()

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def my_rodriguez_rotation(P, k, theta):
    """
    P are the points to be rotated
    k is the axis about which the points are rotated
    theta is the degree in radians

    IMPORTAAAANT: axis of rotation k MUST BE NORMALIZEEEEEEEEEEEEEED AT ALL COSTS
    """

    P_rot = np.zeros((len(P), 3))

    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k, P[i])*np.sin(theta) + \
            k*np.dot(k, P[i])*(1.0-np.cos(theta))

    return P_rot

def find_plane_eq(p1, p2, p3):
    """
    From three points (p1, p2, p3) in a plane, it returns the plane equation
    ax1 + bx2 + cx3 = d

    Note: normal of the plane is given by n = (a,b,c)
    """

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    plane_eq = np.array([a, b, c, d])

    return plane_eq

def get_edges_strips(poly, plane, origin, text, color):
    """
    Computes vtkCutter and poly data from plane equation cutting.

    Note: if the input is a planeSource, we need to change it to vtkPlane
    as our cutter only allows input of vtkPlane.
    """

    if (type(plane) == vtk.vtkPlane):
        VTKplane = plane
    elif (type(plane) == vtk.vtkPlaneSource):
        #convert vtkplaneSource to vtkPlane
        VTKplane = vtk.vtkPlane()
        VTKplane.SetOrigin(plane.GetOrigin())
        VTKplane.SetNormal(plane.GetNormal())
    else:
        # define plane
        a = plane[0]
        b = plane[1]
        c = plane[2]

        # create vtk plane object
        VTKplane = vtk.vtkPlane()
        VTKplane.SetNormal(a, b, c)
        VTKplane.SetOrigin(origin)

    # create cutter
    cutEdges = vtk.vtkCutter()
    cutEdges.SetInputData(poly)
    cutEdges.SetCutFunction(VTKplane)
    cutEdges.GenerateCutScalarsOn()
    cutEdges.GenerateTrianglesOn()
    cutEdges.SetValue(0, 0) # ALWAYS 0,0 as input parameters

    # create strips # just for output purposes
    cutStrips = vtk.vtkStripper()
    # cutStrips.JoinContiguousSegmentsOn()
    cutStrips.SetInputConnection(cutEdges.GetOutputPort())
    cutStrips.Update()

    # # get polydata from strips (just for output purposes)
    # cutPoly = vtk.vtkPolyData()
    # cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
    # cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

    #### works better than above because getoutput gets all the polys and lines
    cutPoly = cutEdges.GetOutput()

    # get planeActor
    planeActor = find_planeActor(cutEdges, text, color, 0)

    return cutPoly, planeActor

def find_planeActor(cutEdges, text, color, display_opt):
    cutterMapper = vtk.vtkPolyDataMapper()
    cutterMapper.SetInputConnection(cutEdges.GetOutputPort())
    cutterMapper.ScalarVisibilityOff()

    # create plane actor ..
    planeActor = vtk.vtkActor()
    planeActor.SetMapper(cutterMapper)
    if color != None:
        planeActor.GetProperty().SetColor(color)
    planeActor.GetProperty().SetLineWidth(6)

    if display_opt:
        ren = vtk.vtkRenderer()

        # create text box to display the angles ..
        if text != None:
            textActor = vtk.vtkTextActor()
            textActor.SetInput(text)
            textActor.SetPosition2(10, 40)
            textActor.GetTextProperty().SetFontSize(24)
            textActor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
            ren.AddActor(textActor)

        ren.AddActor(planeActor)

        vtk_show(ren)

    return planeActor

def get_line_act2(startpt, endpt, color):
    # gets line act but using the linesource object which is better
    linesource = vtk.vtkLineSource()
    linesource.SetPoint1(startpt)
    linesource.SetPoint2(endpt)
    linesource.Update()

    lineact = get_actor_from_polydata(linesource.GetOutput(), color)

    return lineact

def display_landmarks(left_pts, right_pts, low_pt, top_left, top_right, ptsz, display_opt):
    """
    Displays the landmarks, the horizontal lines
        as well as the midline.

    Inputs:
        - all_act : actor for the cut_endo_poly
        - left_pts : left landmarks
        - right_pts : left side landmarks
        - ptsz : point sizes
    """


    # 1. get actors for points
    top_center = (top_left + top_right)/2.0
    top_center_act = include_points(list(top_center), 1, ptsz, (1,0,1))

    left_act = include_points(left_pts, len(left_pts), ptsz, (0,1,0))
    right_act = include_points(right_pts, len(right_pts), ptsz, (0,0,1))
    if low_pt is not None:
        low_pt_act = include_points(list(low_pt), 1, ptsz, (1,0,1))

    # 2.a now add horizontal lines
    VTK_horiz_all = vtk.vtkPoints()

    for i in range(len(left_pts)):
        VTK_horiz_all.InsertNextPoint(left_pts[i])
        VTK_horiz_all.InsertNextPoint(right_pts[i])

    lineArray = vtk.vtkCellArray()

    for i in range(len(left_pts)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i + i) # 0, 2, 4, 6 ,..
        line.GetPointIds().SetId(1, i + i + 1) # 1, 3, 5, 7,...
        lineArray.InsertNextCell(line)

    # 2.b create polydata
    polyLine = vtk.vtkPolyData()

    # 2.c add points and lines to polydata container
    polyLine.SetPoints(VTK_horiz_all)
    polyLine.SetLines(lineArray)

    lineMapper = vtk.vtkPolyDataMapper()
    lineMapper.SetInputData(polyLine)

    lineActor = vtk.vtkActor()
    lineActor.SetMapper(lineMapper)
    lineActor.GetProperty().SetColor(0, 0, 1)
    lineActor.GetProperty().SetLineWidth(2)

    # 3.a also add one more (line to represent vertical direction, to show perpendicular)
    if low_pt is not None:
        long_axis_array = vtk.vtkCellArray()
        long_axis = vtk.vtkLine()
        long_axis.GetPointIds().SetId(0, 0)
        long_axis.GetPointIds().SetId(1, 1)
        long_axis_array.InsertNextCell(long_axis)

        long_axis_pts = vtk.vtkPoints()
        long_axis_pts.InsertNextPoint(top_center)
        long_axis_pts.InsertNextPoint(low_pt)

        # 3.b create actor for long axis line
        long_axis_polydata = vtk.vtkPolyData()

        # 2.c add points and lines to polydata container
        long_axis_polydata.SetPoints(long_axis_pts)
        long_axis_polydata.SetLines(long_axis_array)

        la_mapper = vtk.vtkPolyDataMapper()
        la_mapper.SetInputData(long_axis_polydata)

        la_act = vtk.vtkActor()
        la_act.SetMapper(la_mapper)
        la_act.GetProperty().SetColor(0, 0, 1)
        la_act.GetProperty().SetLineWidth(2)

    # all_act.GetProperty().SetColor(1,0,0)
    ren = vtk.vtkRenderer()
    ren.SetBackground(1,1,1)
    # ren.AddActor(all_act)
    ren.AddActor(right_act)
    ren.AddActor(left_act)
    ren.AddActor(lineActor)
    ren.AddActor(top_center_act)

    if low_pt is not None:
        ren.AddActor(la_act)
        ren.AddActor(low_pt_act)

    assembly = vtk.vtkAssembly()
    # assembly.AddPart(all_act)
    assembly.AddPart(right_act)
    assembly.AddPart(left_act)
    assembly.AddPart(lineActor)

    if display_opt:
        vtk_show(ren)

    return ren, assembly

def change_assembly_properties(assembly, opacity, color, ptsz=None):
    """
    change opacity of all the actors within assembly

    !!! opacity should never be 1.0 if you want to have multiple actors with different opacities!!!

    always set opacity to 0.99 rather than 1.0!
    """
    #
    collection = vtk.vtkPropCollection()
    assembly.GetActors(collection)
    collection.InitTraversal()
    for k in range(collection.GetNumberOfItems()):
        act = vtk.vtkActor.SafeDownCast(collection.GetNextProp())
        if opacity != None:
            act.GetProperty().SetOpacity(opacity)
        if color != None:
            act.GetProperty().SetColor(color)
        if ptsz != None:
            act.GetProperty().SetPointSize(ptsz)

def find_top_points_optimize(pts, C, lowest_pt, vw, hw, tw, show_top_pts = 0):
    """
    pts = 3d points!
    Get the top points
    top point 1 is guessed as the point closest to C.
    For top point 2..
    objective function:
        1. maximize vertical distance 'vd' between lowest point and query point
        2. maximize horizontal distance 'hd' between top point 1 and query point

    vw : vertical weighting for optimization function
    hw : horizontal weighting for optimization function

    tp2 = argmax ( weight1*vd + weight2* hd)

    weights give importance to vertical or horizontal distance

    N.B. For tp1, if we have the case where the view is enclosed and there
    is no basal slanting, then we have to choose the tp1 as the one
    with the smallest projection but ALSO the one furthest away from C.
    """

    ####
    # to perfect this, we might want to put it in a loop
    # so we keep guessing until tp1 and tp2 distances are maximal or something?
    ####

    # find top point 1 by projecting to midline
    num_pts = pts.shape[0]
    midline = normalize(lowest_pt - C)
    org = C - midline
    projs = np.dot(pts - np.tile(org, (num_pts, 1)), midline)
    tp1_idx = np.argsort(projs)[0]
    tp1 = pts[tp1_idx]

    # maximize horizontal and vertical distance from our guessed tp1
    vds = np.linalg.norm(pts - np.tile(lowest_pt, (num_pts, 1)), axis=1)
    hds = np.linalg.norm(pts - np.tile(tp1, (num_pts, 1)), axis=1)

    # minimize (or -ve maximize) the distance of tp2 to C
    dist_to_Cs = np.linalg.norm(pts - np.tile(C, (num_pts, 1)), axis=1)

    # set objective function
    obj = vw*vds + hw*hds - tw*dist_to_Cs

    # guess tp2
    tp2_idx = np.argmax(obj)
    tp2 = pts[tp2_idx]

    # disp = 1
    if show_top_pts:
        acts = [list(pts), list(tp1), list(tp2)]
        colors = [(1,0,0), (0,1,0), (0,0,1)] # tp1 is green, tp2 is blue
        sizes = [0.1, 0.5, 0.5]
        dispall(acts, colors ,sizes)

    top_pts = [tp1, tp2]

    return top_pts, tp1_idx, tp2_idx

def dispall(ls, colors, pointsizes):
    """
    Displays all the objects in ls into renderer.
    e.g. ls = [vtk.vtkPolyData, np array of points, vtkpoints]
    Colors - list of colors for all objects in ls
        len(colors) == len(ls)
    pointsizes - list of pointsizes only for the points in ls (not polydatas)
        len(pointsizes) == number of points objects in ls

        Any error in 'no index for array size' for pointsizes
        means that above condition is not met.
    """
    if len(colors) != len(ls):
        print('ls and colors should have the same number of elements')

    ren = vtk.vtkRenderer()
    for i, obj in enumerate(ls):
        if type(obj) == vtk.vtkPolyData:
            act = get_actor_from_polydata(obj, colors[i])
        elif type(obj) == vtk.vtkPoints:
            act = include_vtk_points(obj, pointsizes[i], colors[i])
        elif (type(obj) == list) or (type(obj) == np.ndarray) or (type(obj) == tuple):
            try:
                n_pts = np.asarray(obj).squeeze().shape[0]
                act = include_points(obj, n_pts, pointsizes[i], colors[i])
            except:
                act = include_points(obj, 1, pointsizes[i], colors[i])
        elif type(obj) == vtk.vtkOpenGLActor:
            act = obj
            if colors[i] != None:
                act.GetProperty().SetColor(colors[i])
        elif type(obj) == vtk.vtkAssembly:
            act = obj
            if colors[i] != None:
                collection = vtk.vtkPropCollection()
                act.GetActors(collection)
                collection.InitTraversal()
                for k in range(collection.GetNumberOfItems()):
                    collection.GetNextProp().GetProperty().SetColor(colors[i])
        elif type(obj) == vtk.vtkAxesActor:
             act = obj
        else:
            print('dispall only allows polydata, vtkpoints, vtkAssemby, or list of points')
            sys.exit()

        ren.AddActor(act)

    vtk_show(ren)

def include_vtk_points(vtkpoints, pointsize, color):
    """ returns actor for vtk points for display"""
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtkpoints)

    vertexFilter = vtk.vtkVertexGlyphFilter()
    vertexFilter.SetInputData(pd)

    vertexM = vtk.vtkPolyDataMapper()
    vertexM.SetInputConnection(vertexFilter.GetOutputPort())

    act = vtk.vtkActor()
    act.SetMapper(vertexM)
    act.GetProperty().SetPointSize(pointsize)
    if color!= None:
        act.GetProperty().SetColor(color)

    return act

def project_onto_xy_plane(points3d, center_point, tol=1e-12):
    """
    Correct version:

        This function returns 3d points projected onto xy plane
        whilst maintaing distance and scale parameters between points.
        https://math.stackexchange.com/questions/375995/rotate-3d-plane

        INPUT:
            points3d : 3d points of the view
            center_point : typically com of points or any other point
            note: the points are ordered from top_point_1 to top_point_2

        OUTPUT:
            points2d : points3d rotated and translated to xy plane
                        such that z component is 0
            R : the final transformation matrix used for the projection onto the 3d plane.
            post_n : normal of the final plane

    N.B. Doesn't work when crxAB starts with [0,0,some_val]

    If error occurs, its because the points are xy, yz, or xz plane where one of the axis has 0 component.
    Solution: simply convert crxAB zero values to 1e-12 instead!

    Need to do both for A4C and A2C cos you dont want to lose ang_sep between the two after random rotation.
    """

    points3d = np.asarray(points3d, dtype=np.float)
    # points3d[np.abs(points3d)<tol] = tol # check in description of function why we do this.
    num_pts = points3d.shape[0]

    # translate all points so that COM is at the origin
    # com = np.mean(points3d, axis=0)
    c = -1.0*center_point # vector pointing towards origin
    pts_trans = points3d + np.tile(c, (num_pts, 1))

    # compute normal of plane defined by top_pt1, top_pt2 and low_pt
    crssAxB = np.cross(pts_trans[0], pts_trans[-1]) # originally pts_trans[-1] second cross input
    crssAxB[np.abs(crssAxB)<tol] = tol

    n = crssAxB / np.linalg.norm(crssAxB)
    nx = n[0]
    ny = n[1]
    nz = n[2]

    # construct Rz ..
    nx2ny2 = np.sqrt(nx**2 + ny**2)
    Rz = np.array([[nx / nx2ny2, ny / nx2ny2, 0],
                [-ny / nx2ny2, nx / nx2ny2, 0],
                [0, 0, 1]])

    # construct Ry ..
    n2 = np.dot(Rz, n)

    Ry = np.array([[n2[2], 0, -n2[0]],
                [0, 1, 0],
                [n2[0], 0, n2[2]]])


    # apply Rz and Ry ..
    R = np.dot(Ry, Rz)
    coords_xy = np.dot(R, pts_trans.T).T
    return coords_xy[:,:2], R

def get_cylinder_heights_2D(left_pts, right_pts, lowest_pt, method):
    """
    left_pts and right_pts are in 2D!!!!!!

    they are aligned such that lowest point is at 0,0
    and midline is on the y-axis.

    Gets cylinder heights as the perpendicualr distances
    between the parallel lines.


    N.B. distances should be the same between the parallel lines
    throughout top to bottom.

    Choose 2 to 3 indices for horizontal pts
    as usually associated with the half cylinders.
    sclowest : last pair of landmarks (closest to lowest_pt)
    """
    # get main cylinder heights
    v = normalize(left_pts[2] - right_pts[2])
    p1 = right_pts[3]
    p2 = ( right_pts[2] + left_pts[2] ) / 2.0
    dL = np.linalg.norm(np.cross(v, p2-p1))

    # H cone for basal and GE
    mid_lowest = (left_pts[-1] + right_pts[-1]) / 2.0
    H_cone = np.linalg.norm(mid_lowest - lowest_pt)

    # H_half_cyl for GE only
    top_left = left_pts[0]
    top_right = right_pts[0]
    H_half_cyl = np.max([top_left[1], top_right[1]]) - np.min([top_left[1], top_right[1]])
    # print('H_half_cyl before', H_half_cyl)
    # top_center = (top_left + top_right) / 2.0
    # second_highest_center = (left_pts[1] + right_pts[1] ) / 2.0
    # H_half_cyl = np.linalg.norm(top_center - second_highest_center)
    # print('H_half_cyl after', H_half_cyl)

    # plt.figure()
    # plt.scatter(left_pts[:,0], left_pts[:,1])
    # plt.scatter(right_pts[:,0], right_pts[:,1])
    # plt.scatter(second_highest_center[0],  second_highest_center[1], c='g')
    # plt.scatter(top_center[0],  top_center[1], c='r')
    # plt.show()

    # for basal we need cone
    if method == 'BOD':

        # return [dL, H_cone]
        return dict(dL = dL, H_cone=H_cone)

    elif (method == 'TBC'):

        # return [H_half_cyl, dL, H_cone]
        return dict(H_half_cyl = H_half_cyl, dL=dL, H_cone=H_cone)

    elif method == 'SBR':
        # for bare take the middle horizontal lines
        #       as the first 5 or so will be truncated and will affect dL calculation
        v = normalize(left_pts[10] - right_pts[10])
        p1 = right_pts[11]
        p2 = (right_pts[10] + left_pts[10]) / 2.0
        dL = np.linalg.norm(np.cross(v, p2 - p1))
        # return dL
        return dict(dL = dL)


    else:
        print('invalid method entered in get_cylinder_heights_2D')
        sys.exit()

def align_2d_pts(pts, lowest_pt, top_pts, line, disp):
    """
        Rotates and translates the 2d landmarks (which lie in the xy plane) and rotates
        them such that y is the lv midline (see below)

        1. Rotate to align points to VERTICAL AXES (not necesarily the y-axis)
        2. Translate so that line is now aligned to the y-axis.
        3. Find the top points

        Inputs:
            line: line to which y-axis is aligned to

        Should be ran after project_onto_xy_plane since that function
        only brings the 3d points to xy plane (doesnt align them once
        it is on the xy plane).

        midline : lowest_point to center of top points
        top_pts : [tp1, tp2]#
        top_LR_idxs: [index of top_pts for top_left,
                    index of top_pts for top_right]
    """
    num_pts = pts.shape[0]
    a,b = line
    x,y = np.array([0,1])
    theta = np.arctan2(-b*x + a*y , a*x + b*y) # signed angle

    # compute rotation matrix
    R_rot = np.array([[np.cos(theta) , -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])

    # apply rotation matrix
    rotated_pts = np.dot(R_rot, pts.T).T

    # rotate and translate these special points too for display!!!
    lowest_pt = np.dot(R_rot, lowest_pt)
    tp1 = np.dot(R_rot, top_pts[0])
    tp2 = np.dot(R_rot, top_pts[1])
    top_center = (tp1 + tp2)/2.0

    if tp1[0] < tp2[0]: # if tp1 is to the left of tp2
        top_LR_idxs = ['left', 'right'] # gives
    else:
        top_LR_idxs = ['right', 'left']

    """
    PART 2. Translate all points in x direction
    Because we want to set the pts such that the lowest point is at x=0.
    """
    # now translate in x direction
    shift_factor = np.array([lowest_pt[0], 0])
    pts_trans = rotated_pts - np.tile(shift_factor, (num_pts, 1))
    lowest_pt = lowest_pt - shift_factor     # translate aswell
    tp1 = tp1 - shift_factor     # translate aswell
    tp2 = tp2 - shift_factor     # translate aswell
    top_center = top_center - shift_factor     # translate aswell

    top_pts_al = [tp1, tp2]
    if disp:
        plt.figure()
        plt.scatter(pts_trans[:,0], pts_trans[:,1], c='k')
        plt.scatter(lowest_pt[0], lowest_pt[1], c='r')
        plt.scatter(top_center[0], top_center[1], c='g')
        plt.scatter(tp1[0], tp1[1], c='b')
        plt.scatter(tp2[0], tp2[1], c='b')
        plt.axvline(x=0.0)
        plt.show()

    return pts_trans, top_pts_al, lowest_pt , top_LR_idxs, R_rot, shift_factor

def interpolate_scheme(aligned_pts, num_lnmk_pairs, top_pts, lowest_pt, method, enclosed, disp):
    """
    Returns 2d interpolation points for basal method.
    aligned_pts : 2d points aligned in specific way depending on method
    If method is basal:
        line perpendicular to basal line is y-axis.
    If method is GE or bare:
        midline is y-axis.

    enclosed : 1 for basal line pts included
            : 0 for open surface contour pts

    Top left and right are found easily since it is aligned in y-axis.
    We find points with highest y-values but opposite signs in x.

    Output:
        left_pts : [0] is top_left
        right_pts : [1] is top_right
    """
    if top_pts[0][0] < top_pts[1][0]:
        top_left = top_pts[0]
        top_right = top_pts[1]
    else:
        top_left = top_pts[1]
        top_right = top_pts[0]

    top_center = (top_left  + top_right) / 2.0
    midline_vec = lowest_pt - top_center

    # set up piecewise interpolator
    sort_idxs = np.argsort(aligned_pts[:,1]) #must be monotically increasing ys
    sorted_xs = aligned_pts[sort_idxs,0]
    sorted_ys = aligned_pts[sort_idxs,1]

    num_cast_ray_pts = 150

    # generate range
    if method=='BOD':
        left_pts = np.zeros((num_lnmk_pairs, 2)) # -1 because the top left and top right are found afterwards
        right_pts = np.zeros((num_lnmk_pairs, 2))

        left_pts[0] = top_left
        right_pts[0] = top_right

        # find vector perpendicular to baseline
        horizontal_mag = 1.5*np.linalg.norm(top_left - top_right)
        baseline_vec = normalize(top_left - top_right)
        perp_bv = np.array([baseline_vec[1], -baseline_vec[0]])

        if np.dot(perp_bv, midline_vec) < 0:
            perp_bv = -perp_bv

        angle = vg.angle(np.array([perp_bv[0], perp_bv[1], 0]),
                            np.array([midline_vec[0], midline_vec[1], 0]),
                            assume_normalized = False,
                            look = np.array([0,0,1]),
                            units = 'rad')

        perp_bv = np.linalg.norm(midline_vec)*np.cos(angle)*perp_bv

        x_dir_mag = np.max(aligned_pts[:,0]) - np.min(aligned_pts[:,0])
        p1_start = np.array([ np.min(aligned_pts[:,0]), top_left[1]])
        p2_start = p1_start - x_dir_mag*baseline_vec  #-ve to make sure baseline is from left to right

        weights = np.linspace(0, 1, num_lnmk_pairs, endpoint = False)
        weights = weights[1:] # since we already added top points!

        # x_sorted = np.sort(aligned_pts[:,0])
        # tol = np.linalg.norm(x_sorted[0] - x_sorted[1])
        tol = np.linalg.norm( top_center - top_right )

        pts1 = cast_ray(p1_start, p2_start, weights, perp_bv, aligned_pts, tol, num_cast_ray_pts, 0)
        pts2 = cast_ray(p2_start, p1_start, weights, perp_bv, aligned_pts, tol, num_cast_ray_pts, 0)

        if pts1[0][0] < pts2[0][0]:
            left_pts[1:] = pts1
            right_pts[1:] = pts2
    elif (method=='TBC'):

        left_pts = np.zeros((num_lnmk_pairs, 2)) # -1 because half cylinder counts as 1 disk
        right_pts = np.zeros((num_lnmk_pairs, 2))

        left_pts[0] = top_left
        right_pts[0] = top_right

        # get maximum distance of tp1 and tp2 and add a bit extra
        horizontal_mag = 1.5*np.linalg.norm(top_left - top_right)
        horizontal_vec = normalize(np.array([midline_vec[1], -midline_vec[0]]))

        # create line points
        if top_left[1] < top_right[1]:
            lowest_side = top_left
            highest_side = top_right
            left_pts[1] = lowest_side   # add the second lowest point as it corresponds to the same fix point
        else:
            lowest_side = top_right
            highest_side = top_left
            right_pts[1] = lowest_side


        if np.dot((highest_side - lowest_side), horizontal_vec) < 0.0:
            horizontal_vec = -horizontal_vec

        weights = np.linspace(0, 1, num_lnmk_pairs-1, endpoint = False)
        # we do -1 since we already added top points but want to start from 0
        # because our ray starts from the lowest of the two top points

        x_dir_mag = np.max(aligned_pts[:,0]) - np.min(aligned_pts[:,0])
        p1_start = lowest_side -0.2*x_dir_mag*horizontal_vec
        p2_start = p1_start + 1.5*x_dir_mag*horizontal_vec
        # p1_start = np.array([ np.min(aligned_pts[:,0]), lowest_side[1]])


        tol = np.linalg.norm( top_center - top_right )
        # tol = 0.1

        # plot_flow(aligned_pts, 0.2, 0.2)
        midline_dir = lowest_pt - ((p1_start + p2_start) / 2.0)

        # add the second top most points manually
        #p2_start+0.000001*midline_dir add a tiny bit down cos its necessary to detect pt
        pt = cast_ray(p2_start+0.000001*midline_dir, p1_start, [weights[0]], midline_dir, aligned_pts, tol, num_cast_ray_pts, 0)[0]

        if pt[0] < 0.0: #if on left side
            left_pts[1] = pt
        else: # if on right side
            right_pts[1] = pt

        # plt.figure()
        # plt.scatter(lowest_side[0], lowest_side[1], c='b', s=50)
        # plt.scatter(highest_side[0], highest_side[1], c='g', s=50)
        #
        # plt.scatter(p1_start[0], p1_start[1], c='r')
        # plt.scatter(aligned_pts[:,0], aligned_pts[:,1], c='k', s=1)
        # plt.scatter(p2_start[0], p2_start[1])
        # plt.scatter(pt[0], pt[1], c=(1,1,0), s=70)
        #
        # plt.show()

        # do for the rest of the pints
        pts1 = cast_ray(p1_start, p2_start, weights[1:], midline_dir, aligned_pts, tol, num_cast_ray_pts, 0)
        pts2 = cast_ray(p2_start, p1_start, weights[1:], midline_dir, aligned_pts, tol, num_cast_ray_pts, 0)

        if pts1[0][0] < pts2[0][0]:
            left_pts[2:,:] = pts1
            right_pts[2:, :] = pts2
        else:
            left_pts[2:,:] = pts2
            right_pts[2:, :] = pts1
    elif (method=='SBR'):
        # same as GE except the p1 and p2 start higher
        # do not include top left and top right to left_pts and right_pts
        # this is because with the shifted landmarks version, top points are different

        left_pts = np.zeros((num_lnmk_pairs, 2)) # -1 because half cylinder counts as 1 disk
        right_pts = np.zeros((num_lnmk_pairs, 2))

        # create line points
        if top_left[1] < top_right[1]:
            lowest_side = top_left
            highest_side = top_right
        else:
            lowest_side = top_right
            highest_side = top_left

        # check if the contour points have basal points as well or open contours
        if enclosed == 0: #i.e. open area
            basal_line_pts = getEquidistantPoints(highest_side, lowest_side, int(0.2*aligned_pts.shape[0]), 1, 1)
            aligned_pts = np.vstack((aligned_pts, basal_line_pts))


        # get maximum distance of toppt1 and tp2 and add a bit extra
        horizontal_vec = normalize(np.array([midline_vec[1], -midline_vec[0]]))
        if np.dot((highest_side - lowest_side), horizontal_vec) < 0.0:
            horizontal_vec = -horizontal_vec

        x_dir_mag = np.max(aligned_pts[:,0]) - np.min(aligned_pts[:,0])
        p1_start = highest_side + 0.2*x_dir_mag*horizontal_vec
        p2_start = p1_start - 1.5*x_dir_mag*horizontal_vec

        midline_dir = lowest_pt - ((p1_start + p2_start) / 2.0)

        tol = 0.001*np.linalg.norm( top_center - top_right )
        midline_dir = np.array([highest_side[0], lowest_pt[1]]) - highest_side
        weights = np.linspace(0, 1, num_lnmk_pairs, endpoint = False)  #  no need numSamples - 1 cos we're not adding the top mst points
        shifted_weights = weights + ((weights[1] - weights[0])/2.0)

        # plt.figure()
        # plt.scatter(lowest_side[0], lowest_side[1], c='b', s=50)
        # plt.scatter(highest_side[0], highest_side[1], c='g', s=50)
        # plt.scatter(p1_start[0], p1_start[1], c='r')
        # plt.scatter(aligned_pts[:,0], aligned_pts[:,1], c='k', s=1)
        # plt.scatter(p2_start[0], p2_start[1])
        # plt.show()

        pts1 = cast_ray(p1_start, p2_start, shifted_weights, midline_dir, aligned_pts, tol, num_cast_ray_pts, 0)
        pts2 = cast_ray(p2_start, p1_start, shifted_weights, midline_dir, aligned_pts, tol, num_cast_ray_pts, 0)

        # need to shift cos landmarks must be at the center of disk sides (Pablo method)
        # pts1 = cast_ray(p1_start, p12_inside, shifted_weights, diagonal_dir, aligned_pts, tol,  150, 1)
        # pts2 = cast_ray(p2_start, p12_inside, shifted_weights, diagonal_dir, aligned_pts, tol,  150, 1)

        if pts1[0][0] < pts2[0][0]:
            left_pts[:] = pts1
            right_pts[:] = pts2
        else:
            left_pts[:] = pts2
            right_pts[:] = pts1

    else:
        print('select available methods: basal, GE or bare')

    ##### display purposes

    if disp:
        fig = plt.figure()
        plt.scatter(sorted_xs, sorted_ys, c='k')
        for i in range(left_pts.shape[0]):
            plt.plot([left_pts[i,0], right_pts[i,0]], [left_pts[i,1], right_pts[i,1]], c='b')
            # plt.scatter(right_pts[:,0], right_pts[:,1], c='b')
        plt.show()

    return left_pts, right_pts

def getEquidistantPoints(p1, p2, n, first_inc, last_inc):
    """
    Generates n points between p1 and p2.
    first_inc : 1 to include first point
    last_inc : 1 to include last point
    """
    if len(p1) == 3:
        pts = np.column_stack((np.linspace(p1[0], p2[0], n+1),
                            np.linspace(p1[1], p2[1], n+1),
                            np.linspace(p1[2], p2[2], n+1)))
    else:
        pts = np.column_stack((np.linspace(p1[0], p2[0], n+1),
                            np.linspace(p1[1], p2[1], n+1)
                            ))

    if first_inc == 0:
        pts = pts[1:] # delete the first one
    if last_inc == 0:
        pts = pts[:-1] # delete the last one

    return pts

def reverse_project_2d_to_3d(pts_2d, shift_factor, R_rot, R_2d, com, disp=0):
    """
    Applies transformations on pts_2d.

    1. shift_factor : 2d vector (translation)
    2. R_rot : 2d rotation
    3. R_2d : backproject 2d pts to 3d (rotation)
    4. com : 3d point (translation)
    """
    num_pts = len(pts_2d)
    pts_2d = pts_2d + np.tile(shift_factor, (num_pts, 1))

    # plt.figure()
    # plt.plot([0, pts_2d[0][0]], [0, pts_2d[0][1]], label='pt1')
    # plt.plot([0, pts_2d[1][0]], [0, pts_2d[1][1]], label='pt2')
    # plt.plot([0, pts_2d[2][0]], [0, pts_2d[2][1]], label='a')
    # plt.plot([0, pts_2d[3][0]], [0, pts_2d[3][1]], label='b')
    # plt.legend()
    # plt.show()

    pts_2d = np.dot(R_rot.T, pts_2d.T).T

    # plt.figure()
    # plt.plot([0, pts_2d[0][0]], [0, pts_2d[0][1]], label='pt1')
    # plt.plot([0, pts_2d[1][0]], [0, pts_2d[1][1]], label='pt2')
    # plt.plot([0, pts_2d[2][0]], [0, pts_2d[2][1]], label='a')
    # plt.plot([0, pts_2d[3][0]], [0, pts_2d[3][1]], label='b')
    # plt.legend()
    # plt.show()

    pts_3d = np.column_stack((pts_2d, np.zeros((num_pts,1)))) # add third z component (zeros)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot([0, pts_3d[0][0]], [0, pts_3d[0][1]], [0, pts_3d[0][2]], label='pt1')
    # ax.plot([0, pts_3d[1][0]], [0, pts_3d[1][1]], [0, pts_3d[1][2]], label='pt2')
    # ax.plot([0, pts_3d[2][0]], [0, pts_3d[2][1]], [0, pts_3d[2][2]], label='a')
    # ax.plot([0, pts_3d[3][0]], [0, pts_3d[3][1]], [0, pts_3d[3][2]], label='b')
    # plt.legend()
    # plt.show()

    pts_3d = np.dot(R_2d.T, pts_3d.T).T # inverse of rotation matrix is simply the transpose.

    if disp:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([0, pts_3d[0][0]], [0, pts_3d[0][1]], [0, pts_3d[0][2]], label='pt1')
        ax.plot([0, pts_3d[1][0]], [0, pts_3d[1][1]], [0, pts_3d[1][2]], label='pt2')
        ax.plot([0, pts_3d[2][0]], [0, pts_3d[2][1]], [0, pts_3d[2][2]], label='a')
        ax.plot([0, pts_3d[3][0]], [0, pts_3d[3][1]], [0, pts_3d[3][2]], label='b')
        plt.legend()
        plt.show()

    pts_3d = pts_3d + np.tile(com, (num_pts, 1))

    if disp:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([com[0], pts_3d[0][0]], [com[1], pts_3d[0][1]], [com[2], pts_3d[0][2]], label='pt1')
        ax.plot([com[0], pts_3d[1][0]], [com[1], pts_3d[1][1]], [com[2], pts_3d[1][2]], label='pt2')
        ax.plot([com[0], pts_3d[2][0]], [com[1], pts_3d[2][1]], [com[2], pts_3d[2][2]], label='a')
        ax.plot([com[0], pts_3d[3][0]], [com[1], pts_3d[3][1]], [com[2], pts_3d[3][2]], label='b')
        plt.legend()
        plt.show()

    return pts_3d

def get_arrow_act(startPoint, endPoint, r, lw):
    """
    Returns vtkactor for a given vector defined by startPoint --> endPoint.
    r - arrow tip radius
    lw - line width of the actor
    """
    arrowSource = vtk.vtkArrowSource()
    arrowSource.SetTipRadius(r)

    normalizedX = endPoint - startPoint
    length = np.linalg.norm(normalizedX)
    normalizedX = normalize(normalizedX)

    arbitrary = np.random.rand(3,1).squeeze()
    normalizedZ = np.cross(normalizedX, arbitrary)
    normalizedZ = normalize(normalizedZ)

    normalizedY = np.cross(normalizedZ, normalizedX)

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for i in range(3):
        matrix.SetElement(i,0,normalizedX[i])
        matrix.SetElement(i,1,normalizedY[i])
        matrix.SetElement(i,2,normalizedZ[i])

    transform = vtk.vtkTransform()
    transform.Translate(startPoint)
    transform.Concatenate(matrix)
    transform.Scale(length, length, length)

    transformPD = vtk.vtkTransformPolyDataFilter()
    transformPD.SetTransform(transform)
    transformPD.SetInputConnection(arrowSource.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort())
    act = vtk.vtkActor()
    act.SetMapper(mapper)
    act.GetProperty().SetLineWidth(lw)
    return act

def split_window(num_x, num_y):
    """
    num_x = number of boxes in the x direction

    (xmin, ymin, xmax, ymax) --> locations of left and right corner positions starting from top of square
    """

    viewPorts = np.zeros((num_x*num_y, 4))
    counter = 0
    for i in range(num_x):
        for j in range(num_y):
            viewPorts[num_x*j + i, :] = [i*(1.0/float(num_x)),
                                         1.0 - (j+1.0)*(1.0/float(num_y)),
                                         (i+1.0)*1.0/float(num_x),
                                         1.0 - j*(1.0/float(num_y))
                                         ]  # [num_y*i + j,:] for vertical fill first

    return viewPorts

def cast_ray(p1_start, p2_start, weights, midline_dir, pts, tol, num_pts, disp):

    """
    Casts ray from p1 to p2.
    P1 and P2 move along midline_vec with increments of weights[i].
    Intersects those rays with pts.


    N.B. Is specifically designed to return the first intersected
    points rather than all of them.

    To make use of them, call the function twice:
        one from p1 to center of p12_center
        and one from p2 to center p12_center.

    This way you remove the intersections at the middle
    portion.

    IMPORTANT: When plotting iteratively, when the points are not in the radius
    of the line_pt, then the 2 selected points will be the first and second
    points in pts array.

    num_pts = number of points in the line
    """
    intersected_points = np.zeros((len(weights), 2))
    num_p = len(weights)
    scale = 2
    tol = scale*tol

    for i in range(num_p):
        p1 = p1_start + weights[i]*midline_dir
        p2 = p2_start + weights[i]*midline_dir

        linepts = getEquidistantPoints(p1, p2, num_pts, 1, 1)

        prev_2_dists = [99999,99999]
        diffs = [0,0] # to track if min dists is getting smaller or bigger

        for j, line_pt in enumerate(linepts):
            dists = cdist(pts, line_pt.reshape(1,2)).flatten()

            # get the first two minimum distances and points
            min_idxs = np.argsort(dists)

            ps1 = pts[min_idxs[0]]
            ps1_sign = side_of_line(p1, p2, ps1).squeeze()
            ps2_sign = ps1_sign
            k = 0
            while (ps1_sign == ps2_sign) and (k!=len(min_idxs)-1):
                k += 1
                ps2 = pts[min_idxs[k]]
                ps2_sign = side_of_line(p1, p2, ps2).squeeze()


            if ps1_sign == ps2_sign:
                continue


            diffs[0] = dists[min_idxs[0]] - prev_2_dists[0]
            diffs[1] = dists[min_idxs[k]] - prev_2_dists[1]

            if (diffs[0] < 0.0) and (diffs[1] < 0.0):
                prev_2_dists[0] = dists[min_idxs[0]]
                prev_2_dists[1] = dists[min_idxs[k]]

            if disp:
                plt.figure()
                plt.scatter(pts[:,0], pts[:,1], s= 0.5)
                bar_left = np.array([line_pt[0] - tol, line_pt[1]])
                bar_right = np.array([line_pt[0] + tol, line_pt[1]])

                plt.plot([bar_left[0], line_pt[0]], [bar_left[1], line_pt[1]], c='m')
                plt.plot([bar_right[0], line_pt[0]], [bar_right[1], line_pt[1]], c='m')
                plt.scatter(ps1[0], ps1[1], s=50, c='g')
                plt.scatter(ps2[0], ps2[1], s=50, c='g')
                plt.scatter(linepts[:,0], linepts[:,1], s=0.5, c='b')
                plt.scatter(line_pt[0], line_pt[1], s=100, c='b')
                plt.scatter(p1[0], p1[1], s=150, c='r')
                plt.scatter(p2[0], p2[1], s=100, c='k')
                # plt.scatter(intersected_points[i][0], intersected_points[i][1], s=20, c='r')
                plt.show()

            if (diffs[0] > 0.0) or (diffs[1] > 0.0):
                line_1 = line( ps1,  ps2)
                horiz_line = line(p1, p2)
                intersected_points[i] = intersection(line_1, horiz_line)
                break;


    return intersected_points

def side_of_line(A, B, pt):
    """
    For 2D, determines which side of line pt is with
    respect to the line defined by A-B
    A,B and pt : ndarray
    returns sign
    -1 if above
    +1 if below
    0 if on line
    """
    x,y = pt
    x1,y1 = A
    x2,y2 = B

    return np.sign((x-x1)*(y2-y1) - (y-y1)*(x2-x1))

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def compute_horizontal_distances(horiz_2ch_a, horiz_2ch_b, horiz_4ch_a, horiz_4ch_b):
    """
    Compute horizontal points using the left and right points for the 2ch and 4ch views

    Returns:
        - 2ch horizontal distances
        - 4ch horizontal distances
        - d_2ch : perpendicular distance between horizontal lines in 2ch
        - d_4ch : perpendicular distance between horizontal lines in 4ch
    """
    horiz_2ch_a = np.asarray(horiz_2ch_a)
    horiz_2ch_b = np.asarray(horiz_2ch_b)
    horiz_4ch_a = np.asarray(horiz_4ch_a)
    horiz_4ch_b = np.asarray(horiz_4ch_b)

    horiz_2ch_dists = np.zeros((len(horiz_2ch_a), 1), dtype=float)
    horiz_4ch_dists = np.zeros((len(horiz_2ch_a), 1), dtype=float)

    for k in range(len(horiz_2ch_a)):
        horiz_2ch_dists[k] = np.linalg.norm(horiz_2ch_a[k] - horiz_2ch_b[k])
        horiz_4ch_dists[k] = np.linalg.norm(horiz_4ch_a[k] - horiz_4ch_b[k])

    return horiz_2ch_dists, horiz_4ch_dists

def get_points_in_line(pts_left, pts_right, weight=0.5):
    """
    For each pair of points on the left and right,
    we form a line between them and pick a point on that line based on weight.
    weight = {0 to 1}
    If weight is close to 0: gives point closer to left side
    If weight is close to 1 : gives point closer to right side
    Default weight is 0.5 : which gives halfway of left and right point (i.e. average)
    """
    pts = []
    for i in range(len(pts_left)):
        pt_L = pts_left[i]
        pt_R = pts_right[i]
        vec = pt_R - pt_L
        uvec = normalize(vec)
        mag = np.linalg.norm(vec)
        pt = pt_L + weight * mag * uvec
        pts.append(pt)

    return pts

def convert_polys_to_pts_list(polys):
    """
    Converts list of vtkpolydatas to a list of numpy array of points
    polys : list of polys
    """
    list_pts = []
    for poly in polys:
        list_pts.append(convert_poly_to_numpy(poly))

    return list_pts

def convert_poly_to_numpy(poly):
    #converts points in poly to numpy
    pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    return pts

def project_3d_points_to_plane(points, p1, p2, p3, normal=None):
    """
    Projects the points in 'array' onto a 3d plane defined by the points
    p1, p2 and p3.

    Either give p1, p2, p3 to define plane
    Or give normal to define plane
    NOT BOTH.

    Inputs:
        array : ndarray (n_pts x 3)
        p1, p2, p3: ndarray (3 x 1)
        numpoints : number of points
    Returns: returns 3d points!
        projected : ndarray (3 x 1)
    """

    points = np.asarray(points)
    try:
        dim = points.shape[1]
    except:
        points = [points]

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)

    # get vectors in plane
    v1 = p3 - p1
    v2 = p2 - p1

    # compute cross product
    cp = np.cross(v1, v2)
    a, b, c = cp  # normal to plane is ax + by + cz

    # # evaluate d, not needed
    # d = np.dot(cp, p3)

    # thus, normal is given by
    plane = vtk.vtkPlane()
    origin = p1
    if not isinstance(normal, list):
        normal = normalize(np.array([a, b, c]))
    plane.SetOrigin(p1)
    plane.SetNormal(normal)

    projected_pts = []
    for pt in points:
        proj = [0, 0, 0]
        plane.ProjectPoint(pt, origin, normal, proj)
        projected_pts.append(proj)

    return np.asarray(projected_pts)

def convert_numpy_to_points(arr):
    """
    Converts numpy array of points to vtkPoints() object.
    Uses numpy_support library.
    """
    vtk_float_arr = numpy_support.numpy_to_vtk(num_array=arr, deep=True, array_type=vtk.VTK_FLOAT)
    vtkpts = vtk.vtkPoints()
    vtkpts.SetData(vtk_float_arr)

    return vtkpts

def get_pcs(poly, get_evec3 = False):
    # gets principal components from polydata using vtk modules.
    # for now 3d data polys
    xarr = vtk.vtkDoubleArray()
    xarr.SetNumberOfComponents(1)
    xarr.SetName("x")

    yarr = vtk.vtkDoubleArray()
    yarr.SetNumberOfComponents(1)
    yarr.SetName("y")

    zarr = vtk.vtkDoubleArray()
    zarr.SetNumberOfComponents(1)
    zarr.SetName("z")

    for i in range(poly.GetNumberOfPoints()):
        pt = poly.GetPoints().GetPoint(i)
        xarr.InsertNextValue(pt[0])
        yarr.InsertNextValue(pt[1])
        zarr.InsertNextValue(pt[2])

    datasetTable = vtk.vtkTable()
    datasetTable.AddColumn(xarr)
    datasetTable.AddColumn(yarr)
    datasetTable.AddColumn(zarr)

    pcastats = vtk.vtkPCAStatistics()
    pcastats.SetInputData(datasetTable)
    pcastats.SetColumnStatus("x", 1)
    pcastats.SetColumnStatus("y", 1)
    pcastats.SetColumnStatus("z", 1)
    pcastats.RequestSelectedColumns()
    pcastats.SetDeriveOption(True)
    pcastats.Update()

    evec1 = vtk.vtkDoubleArray() # pc1
    pcastats.GetEigenvector(0, evec1)

    evec2 = vtk.vtkDoubleArray() # pc2
    pcastats.GetEigenvector(1, evec2)

    evec3 = vtk.vtkDoubleArray()  # pc2
    pcastats.GetEigenvector(2, evec3)

    evec1 = np.squeeze(numpy_support.vtk_to_numpy(evec1))
    evec2 = np.squeeze(numpy_support.vtk_to_numpy(evec2))
    evec3 = np.squeeze(numpy_support.vtk_to_numpy(evec3))

    if get_evec3:
        return evec1, evec2, evec3
    else:
        return evec1, evec2

def get_major_minor_axis_from_evecs(evec1, evec2, approx_rad, slice_poly, tol=0.01, disp=0):
    """
    Cast ray to find intersection between pcs and the slice_poly.
    """
    # since pca translates whole thing to origin, we need to move it back to com
    com = get_com_from_pd(slice_poly)

    major_far_1 = com + 1.7 * approx_rad * normalize(evec1)
    major_far_2 = com - 1.7 * approx_rad * normalize(evec1)
    minor_far_1 = com + 1.7 * approx_rad * normalize(evec2)
    minor_far_2 = com - 1.7 * approx_rad * normalize(evec2)

    pSources = [com, com, com, com]
    pTargets = [major_far_1, major_far_2, minor_far_1, minor_far_2]

    if disp:
        ren = vtk.vtkRenderer()
        slice_act = get_actor_from_polydata(slice_poly, (0,1,0))
        sourceact = include_points(pSources, 4, 2, (1,0,0))
        targetact = include_points(pTargets, 4, 2, (1,1,0))
        ren.AddActor(slice_act)
        ren.AddActor(sourceact)
        ren.AddActor(targetact)
        vtk_show(ren)

    inters_pts = cast_ray_3d(slice_poly, pSources, pTargets, tol=tol)

    major_pt1 = inters_pts[0]
    major_pt2 = inters_pts[1]
    minor_pt1 = inters_pts[2]
    minor_pt2 = inters_pts[3]

    if disp:
        ren = vtk.vtkRenderer()
        slice_act = get_actor_from_polydata(slice_poly, (0,1,0))
        major_pt1act = include_points(major_pt1, 1, 2, (1,0,0))
        major_pt2act = include_points(major_pt2, 1, 2, (1,0,0.5))
        minor_pt1act = include_points(minor_pt1, 1, 2, (1,1,0))
        minor_pt2act = include_points(minor_pt2, 1, 2, (0,1,0))
        ren.AddActor(slice_act)
        ren.AddActor(major_pt1act)
        ren.AddActor(major_pt2act)
        ren.AddActor(minor_pt1act)
        ren.AddActor(minor_pt2act)
        vtk_show(ren)

    return [major_pt1, major_pt2], [minor_pt1, minor_pt2]

def cast_ray_3d(poly, pSources, pTargets, tol=0.01):
    """
    Casts ray using vtk modules.
    In 3d!
    poly : vtkPolyData object
    pSources: list of 3d points
    pTargets : list of 3d points
    """
    try:
        bspTree = vtk.vtkModifiedBSPTree() # sometimes this works
        bspTree.SetDataSet(poly)
        bspTree.BuildLocator()

        if len(pSources) != len(pTargets):
            raise ValueError('There must be same number of source points as target points.')

        inters_pts = []
        for i in range(len(pSources)):
            subId = vtk.mutable(0)
            pcoords = [0, 0, 0]
            t = vtk.mutable(0)
            inters = [0, 0, 0]
            pointid = bspTree.IntersectWithLine(pSources[i], pTargets[i], tol, t, inters, pcoords, subId)

            if pointid == 0:
                raise ValueError('casting ray did not intersect with anything.')

            inters_pts.append(inters)

    except: # sometimes works with this one
        bspTree = vtk.vtkOBBTree() # sometimes this one works
        bspTree.SetDataSet(poly)
        bspTree.BuildLocator()

        if len(pSources) != len(pTargets):
            raise ValueError('There must be same number of source points as target points.')

        inters_pts = []
        for i in range(len(pSources)):
            subId = vtk.mutable(0)
            pcoords = [0, 0, 0]
            t = vtk.mutable(0)
            inters = [0, 0, 0]
            pointid = bspTree.IntersectWithLine(pSources[i], pTargets[i], tol, t, inters, pcoords, subId)

            if pointid == 0:
                raise ValueError('casting ray did not intersect with anything.')

            inters_pts.append(inters)

    return np.squeeze(inters_pts)

def get_signed_angle_3d(Vn, Va, Vb):
    """
    Newest version that includes normal of plane (Vn) (normalized)
    Gets angle FROM Va to Vb
    """
    crx = np.cross(Va, Vb)
    dotcrxn = np.dot(crx, Vn)
    adotb = np.dot(Va, Vb)
    signed_angle = np.arctan2( dotcrxn, adotb)
    return signed_angle

def get_factors(val):
    """
    Get two factors of val that are as close as possible
    """
    N = np.sqrt(val)
    N = np.floor(N)
    M = val/N

    while (val % N != 0):
        N = N-1
        M = val/N

    return int(M), int(N)

def sort_points_by_circular(points):
    """
    sorts points circularly by calculating angles from com
    """
    points = np.asarray(points)
    num_bo_pts = points.shape[0]

    points2d = points

    # if points are 3d, convert to 2d:
    if len(points[0]) > 2:
        points2d, _ = project_onto_xy_plane(points, np.mean(points, axis=0))

    # select reference point and reference line
    ref_pt = points2d[0]
    com = np.mean(points2d, axis=0)
    a,b = normalize(ref_pt - com)

    # compute angles
    signed_angles = np.zeros((num_bo_pts, ), dtype=float)
    for i, pt in enumerate(points2d):
        x, y = normalize(pt - com)
        theta = np.arctan2(-b*x + a*y , a*x + b*y)
        signed_angles[i] = theta

    # sort angles
    sort_idxs = np.argsort(signed_angles)
    sorted_pts = points[sort_idxs]

    return sorted_pts, sort_idxs

def get_line_actor(pts, lw, color, circular=0):
    """
    Returns vtk line actor from a numpy array pts.
    pts : (n_points x dim)
    lw : linewidth

    """
    if isinstance(pts, np.ndarray) or isinstance(pts, list):
        vtk_points = vtk.vtkPoints()
        num_points = int(len(pts))

        for pt in pts:
            vtk_points.InsertNextPoint(pt)
    else:
        vtk_points = pts
        num_points = vtk_points.GetNumberOfPoints()

    lineArray = vtk.vtkCellArray()

    for i in range(num_points-1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i )
        line.GetPointIds().SetId(1, i + 1)
        lineArray.InsertNextCell(line)

    if circular:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, num_points-1)
        line.GetPointIds().SetId(1, 0)
        lineArray.InsertNextCell(line)

    # create polydata
    polyLine = vtk.vtkPolyData()

    # add points and lines to polydata container
    polyLine.SetPoints(vtk_points)
    polyLine.SetLines(lineArray)

    # create pipeline for display*
    lineMapper = vtk.vtkPolyDataMapper()
    lineMapper.SetInputData(polyLine)

    lineActor = vtk.vtkActor()
    lineActor.SetMapper(lineMapper)
    lineActor.GetProperty().SetColor(color)
    lineActor.GetProperty().SetLineWidth(lw)

    return polyLine, lineActor

def get_text_actor(text, pos, fontsize, color):
    textActor = vtk.vtkTextActor()
    textActor.SetInput(text)
    textActor.SetPosition2(pos[0], pos[1])
    textActor.GetTextProperty().SetFontSize(fontsize)
    if color != None:
        textActor.GetTextProperty().SetColor(color)
    return textActor

def get_line_connectivity_poly(polydata):
    # gets line connectivity (i.e. vertex to vertex line connectivity array)
    line_con = [] # line connectivity
    lines_poly = polydata.GetLines()
    lines_poly.InitTraversal()
    idList = vtk.vtkIdList()
    while lines_poly.GetNextCell(idList):
        inner_con = []
        for pointId in range(idList.GetNumberOfIds()):
            inner_con.append(idList.GetId(pointId))
        line_con.append(inner_con)

    return np.asarray(line_con)

def get_assembly_from_actors(actors_list):

    assembly = vtk.vtkAssembly()
    for i in range(len(actors_list)):
        assembly.AddPart(actors_list[i])

    return assembly

def plot_eccentricity_profile(eccs, a4c_to_major):
    a4c_to_major = np.array(a4c_to_major)
    a4c_to_major[a4c_to_major < 0.0] = a4c_to_major[a4c_to_major < 0.0] + 180

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Slice #', fontsize=20)
    ax1.set_ylabel(r'Eccentricity  $\mu$', fontsize=18)
    ax1.axhline(y=1.0, linestyle='--', color=color, alpha=0.7)
    # print('min ecc =', np.min(eccs))
    # print('max ecc =', np.max(eccs))
    # print(ax1.get_ylim())

    ax1.plot(np.arange(len(eccs)), eccs, marker='o', color=color)
    ax1.set_ylim(0.2, 1.5) # for YHC and UKBB plots
    # ax1.set_ylim(0.8, 1.05)

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(r'Orientation Angle  $A_m \, (^{\circ})$', fontsize=18)
    ax2.plot(np.arange(len(eccs)), a4c_to_major, marker='x', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
    ax2.set_ylim(0, 180)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


    # plt.figure()
    # plt.plot(np.arange(len(eccs)), eccs, marker='o')
    # plt.axhline(y=1.0, linestyle='--', color='r', alpha=0.7)
    # plt.xticks(np.arange(len(eccs)), fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel('Slice #', fontsize=20)
    # plt.ylabel(r'Eccentricity  $\mu$', fontsize=20)
    # plt.show()

def compute_angle_between_2d(a, b):

    """
    GIVES CLOCKWISE ANGLE between a and b.
    Returns the angle in RADIANS between vectors 'a' and 'b'::
    Only works with 2d
    """
    a = np.asarray(a)
    b = np.asarray(b)

    ax, ay = a
    bx, by = b

    dot = np.dot(a,b)
    crx = ax*by - ay*bx

    angle = np.arctan2(np.abs(crx), dot)

    if crx < 0.0:
        angle = 2.0*np.pi - angle

    return angle

def get_signed_angle(v,w):
    # degrees angle from v to w
    return math.degrees(np.arctan2(w[1], w[0]) - np.arctan2(v[1], v[0]))

def plot_many_ecc_slices(k, ax, slices_rot, landmarks_rot, plot_fit, disp):
    font = {'family': 'Georgia',
            'color': 'black',
            'weight': 'bold',
            'size': 21,
            }

    ax.set_aspect('equal')
    offset_x_plot = np.mod(k, 5) * 75
    offset_y_plot = np.floor(k / 5) * 75

    offset_x_text = (np.mod(k, 5) * 72) + 6 * np.mod(k, 5)  # offset between rows + offset within the column
    offset_y_text = (np.floor(k / 5) * 90) + 5 * np.floor(k / 5)  # offset between cols + offset within the column

    ax.text(0.11 + offset_x_text / 400.0, 0.84 - offset_y_text / 400.0, str(k), fontdict=font,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    fit_best_ellipse(slices_rot, ax, offset=[offset_x_plot, offset_y_plot], label=None,
                                               plot_fit=plot_fit, disp=disp)

    # also add the a4c and a2c lines
    ax.plot(landmarks_rot[:2, 0] + offset_x_plot, landmarks_rot[:2, 1] + offset_y_plot, alpha=0.8, color='g')
    ax.plot(landmarks_rot[2:4, 0] + offset_x_plot, landmarks_rot[2:4, 1] + offset_y_plot, alpha=0.8, color='b')
    ax.scatter(landmarks_rot[4, 0] + offset_x_plot,
               landmarks_rot[4, 1] + offset_y_plot, alpha=0.8, color='r')

def fit_best_ellipse(points_2d, ax, offset=None, label=None, plot_fit=True, disp = 0):
    """
    https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
    offset = if you want to plot all the cross-sections very close together without any white space in between subplots,
            then you need to plot into 1 axes (and put an offset between contour points for each slice)
    """

    X = np.array([[val] for val in points_2d[:,0]])
    Y = np.array([[val] for val in points_2d[:,1]])

    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()

    # # Print the equation of the ellipse in standard form
    # print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1], x[2], x[3],
    #                                                                                           x[4]))

    x_coord = np.linspace(np.min(X.flatten()) - 20, np.max(X.flatten()) + 20, 300)
    y_coord = np.linspace(np.min(Y.flatten()) - 20, np.max(Y.flatten()) + 20, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord

    if offset==None:
        ax.scatter(X, Y, color='azure')  # Plot the original data
        if plot_fit:
            CS = ax.contour(X_coord, Y_coord, Z_coord, levels=[1], color='crimson', linewidths=2) # Plot the least squares ellipse (the fitted ellipse)
            contour_pts = CS.allsegs[0][0]
    else:
        ax.scatter(X + offset[0], Y + offset[1], color='tab:blue', s=10, zorder=1)  # Plot the original data
        if plot_fit:
            CS = ax.contour(X_coord + offset[0], Y_coord + offset[1], Z_coord, levels=[1], colors='tab:red', linewidths=2, zorder=2)  # Plot the least squares ellipse (the fitted ellipse)
            contour_pts = CS.allsegs[0][0]

    # set the title inside the axes
    # if label != None:
    #     ax.text(.5, .9, label, horizontalalignment='center', transform=ax.transAxes)

    # compute difference between fitted ellipse line and the data points
    # l2norms = measure_diff_fittedellipse_data(points_2d, contour_pts)

def get_plane_source(origin, p1, p2):

    pl = vtk.vtkPlaneSource()
    pl.SetOrigin(origin)
    pl.SetPoint1(p1)
    pl.SetPoint2(p2)
    pl.Update()

    return pl

def sort_landmark_pairs_along_midline(pts_4ch, pts_2ch, C, endo_node):
    """
    Sorts the pairs of landmarks (2 points per view) along midline
    midline : full vector (not normalized) from self.C to endo apex node

    i.e. - Lowest point corresponds to the one that is closest to apex node
         - Highest point is closest to self.C

    sorted_pts : highest to lowest
        sorted_pts[0] : highest pt
        sorted_pts[-1] : lowest pt
    note: higher projection value means closer to apex.
    """
    pts_4ch = np.asarray(pts_4ch)
    pts_2ch = np.asarray(pts_2ch)

    mag = np.linalg.norm(endo_node - C)
    norm_vec = normalize(endo_node - C)
    fix_pt = C - mag*norm_vec

    # for 2ch first
    projs = []
    for pt in pts_2ch:
        proj = np.dot(pt - fix_pt, norm_vec)
        projs.append(proj)

    sort_idxs = np.argsort(projs) #ascending order
    sorted_pts_2ch = pts_2ch[sort_idxs]

    # for 4ch
    projs = []
    for pt in pts_4ch:
        proj = np.dot(pt - fix_pt, norm_vec)
        projs.append(proj)

    sort_idxs = np.argsort(projs) #ascending order
    sorted_pts_4ch = pts_4ch[sort_idxs]

    return sorted_pts_4ch, sorted_pts_2ch

def close_polyline(poly, polyact):
    # given a polyline, it checks if it is open, and closes it.
    lines_sorted, opened_poly = sort_line_con(poly)
    if opened_poly==1:  # opened polyline
        # insert first idx of point in sorted line to last position
        lines_sorted.append(lines_sorted[0])
        sorted_pts = get_pts_from_sorted_idxs(poly, lines_sorted)
        # poly, polyact = convert_numpy_to_poly(sorted_pts)
        poly, polyact = get_line_actor(sorted_pts, 1, (0, 1, 0), circular=1)
        return poly, polyact
    else:
        return poly, polyact

def sort_line_con(poly):
    """
    Given an open polyline (slice view), gets the line_con and sorts it
    so that it indexes forms a line.
    e.g. [0,1], [1,2], [2,3], ... --> [0,1,2,3,...]
    """

    line_con = get_line_connectivity_poly(poly)
    d = dict(list(line_con))
    sorted_lines = []

    uniq, counts = np.unique(np.squeeze(line_con), return_counts=True)
    opened_poly = 0
    if (0 in counts) or (1 in counts): # open polyline
        opened_poly = 1
        start = next(iter(d.keys() - d.values()))
        while start in d:
            end = d[start]
            sorted_lines.append((start, end))
            start = end
    else: # if closed
        _, _, sort_idxs = sort_circular_pts(poly.GetPoints())
        sorted_lines = [(sort_idxs[i], sort_idxs[i + 1]) for i in range(len(sort_idxs) - 1)]
        sorted_lines.append((sort_idxs[-1], sort_idxs[0]))

    sorted_lines = [e for l in sorted_lines for e in l]
    sorted_lines = list(dict.fromkeys(sorted_lines))

    return sorted_lines, opened_poly

def sort_circular_pts(vtk_points):
    """
        Sorts points in circle by calculating angle between reference line
        and angle lines.
        Returns numpy array of sorted points
            and the sorted (signed) angles.

        URL:
            https://gamedev.stackexchange.com/questions/69475/how-do-i-use-the-dot-product-to-get-an-angle-between-two-vectors
    """

    # convert vtk points to numpy
    cpd_numpy = numpy_support.vtk_to_numpy(vtk_points.GetData())
    circle_pts_2D, _ = project_onto_xy_plane(cpd_numpy, np.mean(cpd_numpy, axis=0))
    num_bo_pts = circle_pts_2D.shape[0]

    # select reference point and reference line
    ref_pt = circle_pts_2D[0]
    com = np.mean(circle_pts_2D, axis=0)
    a,b = normalize(ref_pt - com)

    # compute angles
    signed_angles = np.zeros((num_bo_pts, ), dtype=float)
    for i, pt in enumerate(circle_pts_2D):
        x, y = normalize(pt - com)
        theta = np.arctan2(-b*x + a*y , a*x + b*y)
        signed_angles[i] = theta

    # sort angles
    sort_idxs = np.argsort(signed_angles)
    sorted_pts = cpd_numpy[sort_idxs]

    return sorted_pts, np.sort(signed_angles), sort_idxs

def get_pts_from_sorted_idxs(poly, lines_sorted):
    # given a polydata and a list of point indices [0,1,2,3,...., 122]
    # that represent a polyline,
    # we return the  points sorted in the order of 'lines_sorted'
    np_pts = convert_poly_to_numpy(poly)
    sorted_pts = []
    for i in range(len(lines_sorted)):
        sorted_pts.append(np_pts[lines_sorted[i]])

    return np.asarray(sorted_pts)

def draw_arrow(start=np.array([0,0,0]), end=np.array([1,0,0]), scale=80, tip_length = 20, tip_radius=5,  tip_resolution=1000, shaft_radius=2, shaft_resolution=1000):
    # origin of the arrow
    # end of the arrow
    # length of the arrow
    arrow = vtk.vtkArrowSource()
    arrow.SetTipLength(tip_length)
    arrow.SetTipRadius(tip_radius)
    arrow.SetTipResolution(tip_resolution)
    arrow.SetShaftRadius(shaft_radius)
    arrow.SetShaftResolution(shaft_resolution)

    normalizedX = end - start
    normalizedX = normalize(normalizedX)

    rng = vtk.vtkMinimalStandardRandomSequence()
    rng.SetSeed(12323)
    arbitrary = [0, 0, 0]
    for i in range(3):
        rng.Next()
        arbitrary[i] = rng.GetRangeValue(-10, 10)
    normalizedZ = np.cross(normalizedX, arbitrary)
    normalizedZ = normalize(normalizedZ)

    normalizedY = np.cross(normalizedZ, normalizedX)
    normalizedY = normalize(normalizedY)

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for i in range(3):
        matrix.SetElement(i, 0, normalizedX[i])
        matrix.SetElement(i, 1, normalizedY[i])
        matrix.SetElement(i, 2, normalizedZ[i])

    transform = vtk.vtkTransform()
    transform.Translate(start)
    transform.Concatenate(matrix)
    transform.Scale(scale, scale, scale)

    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetTransform(transform)
    tpd.SetInputConnection(arrow.GetOutputPort())

    mapp = vtk.vtkDataSetMapper()
    mapp.SetInputConnection(tpd.GetOutputPort())

    act = vtk.vtkActor()
    act.SetMapper(mapp)

    return act
