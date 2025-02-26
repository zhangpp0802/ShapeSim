import os
import itertools
from itertools import product
import numpy as np
import math
import xml.etree.ElementTree as ET
import star_polygon
import triangle as tr
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import collections  as mc
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
import matplotlib.tri as tri
import matplotlib.cm as cm 
import random
import pathlib
from tqdm import tqdm
from datetime import datetime
from skgeom import *
import statistics
import geometry
import operator

class Environment():
    """An environment class.

    This class stores data related to a physical or virtual environment, which is defined in an
    XML file. In practice, the environment is defined as a closed polygon with holes. The boundary of 
    this polygon represents the walls enclosing the environment, and holes in the polygon represent
    obstacles in the environment.

    Attributes:
        file: The name of the file that defines the environment layout.
        name: The name of the environment.
        border: The polygon that defines the boundary of the environment.
        center: The center of the environment. This is not always the centroid.
        width: The maximum width of the environment.
        height: The maximum height of the environment.
        area: The area of the walkable space in the environment.
        obstacles: A list of obstacles present in the environment. Does not include the boundary.
        triangulation_verts: The vertices defining the triangulation of the walkable space in 
                              the environment.
        triangulation_segs: The line segments defining the triangulation of the walkable space in
                             the environment.
        triangulation_holes: The holes in the environment, for which we should not triangulate.
                              These holes are used to identify the obstacles of the environment, so
                              they must always be inside an obstacle. The holes are represented as
                              2D points (that lie inside the obstacle).
        triangle_areas: The areas of the triangles in the triangulation of the environment. This is
                         used when uniformly sampling points in the environment.
        triangulated_env: The triangulated environment (from the triangle library).
        triangulation_visibility_polygons: The visibility polygons whose kernels are the points of the
                                           triangluation of the environment free space.
        random_sampled_visibility_polygons: The visibility polygons whose kernels are the points that are
                                            randomly sampled from the environment free space.
        visibility_arrangement: Auxilliary structure required for visibility computation.
        tri_expansion: Auxilliary structure required for visibility computation.
    """

    def __init__(self, env_file, triangle_area=None):
        self.file = env_file
        self.name = None
        self.border = None # Polygon object
        self.center = None
        self.width = None
        self.height = None
        self.area = None
        self.obstacles = [] # List of polygon objects
        self.triangulation_verts = None
        self.triangulation_segs = None
        self.triangulation_holes = None
        self.triangle_areas = []
        self.triangulated_env = None
        self.triangulation_visibility_polygons = []
        self.random_sampled_visibility_polygons = []
        self.visibility_arrangement = arrangement.Arrangement()
        self.tri_expansion = None

        self.parse_env_file(env_file)
        self.compute_area()
        self.compute_extents()
        if triangle_area == None:
            self.triangulate(area=(self.area / 500))
        else:
            self.triangulate(area=triangle_area)
        self.build_visibility_polygons()

    def compute_area(self):
        """Compute the area of the free space of the environment.
        """
        obs_areas = [o.area for o in self.obstacles]
        obs_area = sum(obs_areas)
        self.area = self.border.area - obs_area

    def compute_extents(self):
        """Compute the bounding box of the environment.
        """
        min_x = min(p[0] for p in self.border.pts)
        max_x = max(p[0] for p in self.border.pts)
        min_y = min(p[1] for p in self.border.pts)
        max_y = max(p[1] for p in self.border.pts)
        for o in self.obstacles:
            min_x = min(min_x, min([p[0] for p in o.pts]))
            max_x = max(max_x, max([p[0] for p in o.pts]))
            min_y = min(min_y, min([p[1] for p in o.pts]))
            max_y = max(max_y, max([p[1] for p in o.pts]))
        self.extents_x = (max_x - min_x) / 2
        self.extents_y = (max_y - min_y) / 2
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        sum_x = sum([p[0] for p in self.border.pts])
        sum_x /= len(self.border.pts)
        sum_y = sum([p[1] for p in self.border.pts])
        sum_y /= len(self.border.pts)
        self.centroid = np.array([sum_x, sum_y])

    def parse_env_file(self, env_file):
        """Parse an environment description file to create an environment object.
        
        Args:
            env_file: The filename of the environment description file to be parsed.
        """
        tree = ET.parse(env_file)
        root = tree.getroot()
        self.name = root.attrib['name']
        print('*' * len('* Parsing environment: {} *'.format(self.name)))
        print('* Parsing environment: {} *'.format(self.name))
        print('*' * len('* Parsing environment: {} *'.format(self.name)))
        
        for child in root:
            if child.tag == 'border':
                self.parse_border(child)

            elif child.tag == 'obstacle':
                self.parse_obs(child)
        
        # Convert environmnet to segments, for visibility computations
        for i in range(len(self.border.pts)):
            p1 = self.border.pts[i]
            p2 = self.border.pts[(i+1) % len(self.border.pts)]
            self.visibility_arrangement.insert(Segment2(Point2(p1[0], p1[1]), Point2(p2[0], p2[1])))
        for o in self.obstacles:
            for i in range(len(o.pts)):
                p1 = o.pts[i]
                p2 = o.pts[(i+1) % len(o.pts)]
                self.visibility_arrangement.insert(Segment2(Point2(p1[0], p1[1]), Point2(p2[0], p2[1])))

        self.width = abs(min([p[0] for p in self.border.pts]) - max([p[0] for p in self.border.pts]))
        self.height = abs(min([p[1] for p in self.border.pts]) - max([p[1] for p in self.border.pts]))
        print('--- Finished environment parsing ---')

    def parse_border(self, elem):
        """Parse out the border (bounding polygon) of the environment from the environment file.

        Args:
            elem: XML element containing the vertices data for the environment boundary.
        """
        print('Parsing {}...'.format(elem.attrib['name']))
        center = None
        verts = []

        for child in elem:
            # Border of the environment
            if len(child.attrib) == 0 and child.tag == 'vertices':
                for vert in child:
                    v = np.array([float(vert.text.split(',')[0]), float(vert.text.split(',')[1])])
                    verts.append(v)
            # Center of the environment (usually [0.0, 0.0])
            elif child.attrib['type'] == 'center':
                center = np.array([float(child.text.split(',')[0]), float(child.text.split(',')[1])])

        # Add a random Steiner point to break the first wall into 2 segments. This is done to 
        # ensure that we compute different Delaunay triangulations if we are given a pair of 
        # identical environments.
        p1 = verts[0]
        p2 = verts[1]
        t = random.uniform(0, 1)
        steiner_pt = (t * p1) + ((1-t) * p2)
        verts.insert(1, steiner_pt)
        self.border = star_polygon.StarPolygon(verts, compute_area=True)
        self.center = center

    def parse_obs(self, elem):
        """Parse out an obstacle (polygon) of the environment from the environment file.

        Args:
            elem: XML element containing the vertices data for the obstacle.
        """
        print('Parsing {}...'.format(elem.attrib['name']))
        verts = []

        for child in elem:
            for vert in child:
                v = np.array([float(vert.text.split(',')[0]), float(vert.text.split(',')[1])])
                verts.append(v)

        self.obstacles.append(star_polygon.StarPolygon(verts, compute_area=True))

    def triangulate(self, area=0.1):
        """Compute the constrained conforming Delaunay triangulation of the free space of the 
        environment.

        Args:
            area: Maximum area constraint on the size of triangles in the triangulation.
        """
        print('--- Triangulating environment ---')
        print('CDT area:', area)
        # Extract the segment data from the environment border and obstacles
        walls = [[v[0], v[1]] for v in self.border.pts]
        obs = []
        for o in self.obstacles:
            temp = [[v[0], v[1]] for v in o.pts]
            obs.append(temp)

        # Format the data so it can be used by the triangle library
        self.triangulation_verts, self.triangulation_segs, self.triangulation_holes = self.get_triangulation_data(walls, obs)

        # Let there be triangles!
        if len(self.triangulation_holes) == 0:
            A = dict(vertices=self.triangulation_verts, segments=self.triangulation_segs)
        else:
            A = dict(vertices=self.triangulation_verts, segments=self.triangulation_segs, holes=self.triangulation_holes)
        self.triangulated_env = tr.triangulate(A, 'pa{}'.format(area))

        # Compute the areas of the triangles. This is used to unformly sample the triangulation.
        for tri in self.triangulated_env['triangles']:
            p1 = self.triangulated_env['vertices'][tri[0]]
            p2 = self.triangulated_env['vertices'][tri[1]]
            p3 = self.triangulated_env['vertices'][tri[2]]
            self.triangle_areas.append(self.triangle_area(p1, p2, p3))
        self.area = sum(self.triangle_areas)
        print('--- Finished triangulation ---')

    def triangle_area(self, p1, p2, p3):
        """Compute the area of a triangle.
        
        Args:
            p1: The first vertex of the triangle.
            p2: The second vertex of the triangle.
            p3: The third vertex of the triangle.

        Returns:
            The area of the triangle defined by (p1, p2, p3).

        Return type:
            float
        """
        return abs(((p1[0]*(p2[1] - p3[1])) + (p2[0]*(p3[1] - p1[1])) + (p3[0]*(p1[1] - p2[1]))) * 0.5)

    def get_triangulation_data(self, walls, obs):
        """Extract some useful information from the triangulation structure, including the triangulation
        vertices and segments, and the locations of holes in the environment polygon (where holes represent
        obstacles).

        Args:
            walls: The walls of the outer border of the environment (the edges of the environment boundary).
            obs: A list of obstacles in the environment.

        Returns:
            A tuple of the vertices, segments and holes in the triangulation of the environment.

        Return type:
            A tuple of three lists ([], [], [])
        """
        verts = []
        segments = []
        holes = []

        idx = 0
        verts_offset = 0
        for w in walls:
            verts.append(w)
        for w in walls:
            segments.append((idx, (idx+1) % len(walls)))
            idx += 1

        verts_offset = len(verts)
        for obstacle in obs:
            for o in obstacle:
                verts.append(o)
        for obstacle in obs:
            idx = 0
            for o in obstacle:
                segments.append((idx+verts_offset, ((idx+1) % len(obstacle))+verts_offset))
                idx += 1
            verts_offset += len(obstacle)
            holes.append(self.get_point_inside(obstacle))

        return verts, segments, holes

    def build_visibility_polygons(self):
        """Compute visibility polygons at all of the sampled locations in the environment
        free space.
        """
        # One-time precomputation to aid in visibility queries
        self.tri_expansion = TriangularExpansionVisibility(self.visibility_arrangement) 
        tri_x = [v[0] for v in self.triangulated_env['vertices']]
        tri_y = [v[1] for v in self.triangulated_env['vertices']]
        print('Building visibility polygons for triangulation vertices.')
        for i in tqdm(range(len(tri_x))):
            vis_poly = self.build_vis_poly_helper(tri_x[i], tri_y[i])
            if vis_poly:
                self.triangulation_visibility_polygons.append(vis_poly)

        self.random_sampled_visibility_polygons = []

        print('num DT vis polys:', len(self.triangulation_visibility_polygons))
        print('num random sample vis polys:', len(self.random_sampled_visibility_polygons))

    def get_poly(self,pt):
        """Get the polygon of a certain rotation
        rot: the degree of rotation
        """
        return self.build_vis_poly_helper(pt[0],pt[1])


    def build_vis_poly_helper(self, tri_x, tri_y):
        """Compute the visibility polygon at a given (legal) point in the environment.

        Args:
            tri_x: Position of the query point in the x-dimension.
            tri_y: Position of the query point in the y-dimension.

        Returns:
            The visibility polygon at (tri_x, tri_y).

        Return type:
            StarPolygon object
        """
        p = [tri_x, tri_y]
        if not self.point_is_legal(p):
            return None
        else:
            query_point = Point2(tri_x, tri_y)
            face = self.visibility_arrangement.find(query_point)
            vis = self.tri_expansion.compute_visibility(query_point, face)
            vis = [np.array([float(v.point().x()), float(v.point().y())]) for v in vis.vertices]
            return star_polygon.StarPolygon(vis, np.array([tri_x, tri_y]))
            
    def get_point_inside(self, verts):
        """Compute a random point inside a polygon.

        Args:
            verts: The vertices of the polygon we wish to sample.

        Returns:
            A random point inside the given polygon.

        Return type:
            A list of floats [x, y]
        """
        segs = [(i, (i+1) % len(verts)) for i in range(len(verts))]
        A = dict(vertices=verts, segments=segs)
        B = tr.triangulate(A, 'p')
        idx_p1 = B['triangles'].tolist()[0][0]
        idx_p2 = B['triangles'].tolist()[0][1]
        idx_p3 = B['triangles'].tolist()[0][2]
        p1 = B['vertices'].tolist()[idx_p1]
        p2 = B['vertices'].tolist()[idx_p2]
        p3 = B['vertices'].tolist()[idx_p3]

        return [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3]

    def point_is_legal(self, p):
        """Determine if a given point is inside the free space of the environment. For this to be true,
        the point must be inside the border of the environment and outside of all obstacles in the 
        environment. The point cannot be on the boundary of either the environment or any obstacles 
        inside the environment.

        Args:
            p: The point whose legality we wish to verify.

        Returns:
            True if the query point is inside the free space of the environment, false otherwise.

        Return type:
            boolean
        """
        # Point is not inside the border of the environment
        if self.border.point_inside(p) != 1:
            return False

        # Check if point is inside any obstacle
        for o in self.obstacles:
            if o.point_inside(p) != -1:
                return False

        return True

    def get_triangulation_vis_poly(self, kernel):
        """Find the pre-computed visibility polygon that is associated with the given kernel.
        In practice, we just find the visibility polygon whose kernel is the shortest distance from
        the given kernel. This means the user must pass in a point that we know to be a kernel of one
        of the pre-computed visibility polygons. If they do not do this, the returned vis. polygon
        most likely will not be the correct polygon for the given kernel point.

        Args:
            kernel: The point in the environment for which we wish to find the visibilty polygon.

        Returns:
            The visibilty polygon that was computed at the kernel point.

        Return type:
            StarPolygon object
        """
        listcopy = list.copy(self.triangulation_visibility_polygons)
        listcopy.sort(key=lambda x: np.linalg.norm(x.kernel - kernel))
        return listcopy[0]
    
    def draw(self, paths=[]):
        """Create a diagram of the environment and, optionally, paths through that environment.

        Args:
            paths: A list of paths in the environment that we wish to visualize.
        """
        fig, ax = plt.subplots(1)

        # Environment borders
        num_pts = len(self.border.pts)
        border = [(self.border.pts[i], self.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        lc = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax.add_collection(lc)
        
        # Environment obstacles
        for o in self.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            plt.fill(x, y, 'black')

        # Optional paths passed as an argument
        for p in paths:
            x = [pt.position[0] for pt in p.path]
            y = [pt.position[1] for pt in p.path]
            plt.plot(x, y)

        ax.set_aspect('equal')
        plt.title(self.name)
        plt.show()

    def draw_triangulation(self):
        """Create a diagram of the environment, including the triangulation of its free space.
        """
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Environment borders
        num_pts = len(self.border.pts)
        border = [(self.border.pts[i], self.border.pts[(i+1) % num_pts]) for i in range(num_pts)]
        lc1 = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        lc2 = mc.LineCollection(border, colors=np.array([(0, 0, 0, 1)]), linewidths=1)
        ax1.add_collection(lc1)
        ax2.add_collection(lc2)
        
        # Environment obstacles
        for o in self.obstacles:
            x = [pt[0] for pt in o.pts]
            y = [pt[1] for pt in o.pts]
            # plt.fill(x, y, 'black')
            ax1.fill(x, y, 'black')
            ax2.fill(x, y, 'black')

        # Get all triangle lines and draw them 
        verts = np.array(self.triangulated_env['vertices'])
        verts = [v for v in verts if self.point_is_legal(v)]
        verts_x = [v[0] for v in verts]
        verts_y = [v[1] for v in verts]
        verts = np.array(self.triangulated_env['vertices'])

        import matplotlib.font_manager
        fonts = [i for i in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf') if 'times' in i.lower()]

        ax1.triplot(verts[:, 0], verts[:, 1], self.triangulated_env['triangles'], 'k-', linewidth=1.25, color='green')
        ax1.set_axis_off()
        ax1.set_aspect('equal')
        ax2.scatter(verts_x, verts_y, c='green', s=8)
        ax2.set_axis_off()
        ax2.set_aspect('equal')
        plt.show()
        save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'paper materials', 'triangulation.pdf')

class Trajectory():
    def __init__(self, traj_file):
        self.traj_file = traj_file
        self.border = None # Polygon object
        self.center = None
        self.width = None
        self.height = None
        self.trajectory = []
        self.directions = []

        self.parse_traj_file(self.traj_file)
        self.calculate_directions()

    def parse_traj_file(self, env_file):
        """Parse an environment description file to create an environment object.
        
        Args:
            env_file: The filename of the environment description file to be parsed.
        """
        tree = ET.parse(env_file)
        root = tree.getroot()
        self.name = root.attrib['name']
        print('*' * len('* Parsing Trajectory: {} *'.format(self.name)))
        print('* Parsing trajectory: {} *'.format(self.name))
        print('*' * len('* Parsing trajectory: {} *'.format(self.name)))
        
        for child in root:
            if child.tag == 'border':
                self.parse_border(child)

            elif child.tag == 'traj':
                self.parse_traj(child)
        print('--- Finished environment parsing ---')

    def parse_border(self, elem):
        """Parse out the border (bounding polygon) of the environment from the environment file.

        Args:
            elem: XML element containing the vertices data for the environment boundary.
        """
        print('Parsing {}...'.format(elem.attrib['name']))
        center = None
        verts = []

        for child in elem:
            # Border of the environment
            if len(child.attrib) == 0 and child.tag == 'vertices':
                for vert in child:
                    v = np.array([float(vert.text.split(',')[0]), float(vert.text.split(',')[1])])
                    verts.append(v)
            # Center of the environment (usually [0.0, 0.0])
            elif child.attrib['type'] == 'center':
                center = np.array([float(child.text.split(',')[0]), float(child.text.split(',')[1])])

        # Add a random Steiner point to break the first wall into 2 segments. This is done to 
        # ensure that we compute different Delaunay triangulations if we are given a pair of 
        # identical environments.
        p1 = verts[0]
        p2 = verts[1]
        t = random.uniform(0, 1)
        steiner_pt = (t * p1) + ((1-t) * p2)
        verts.insert(1, steiner_pt)
        self.border = star_polygon.StarPolygon(verts, compute_area=True)
        self.center = center
    
    def parse_traj(self, elem):
        """Parse out an obstacle (polygon) of the environment from the environment file.

        Args:
            elem: XML element containing the vertices data for the obstacle.
        """
        print('Parsing {}...'.format(elem.attrib['name']))
        verts = []

        for child in elem:
            verts = []
            for vert in child:
                v = np.array([float(vert.text.split(',')[0]), float(vert.text.split(',')[1])])
                self.trajectory.append(v)
    
    def draw_traj(self):
        for pt in self.trajectory:
            x = pt[0]
            y = pt[1]
            plt.fill(x, y, 'black')
            ax.set_aspect('equal')
            plt.title(self.name)
            plt.show()
    def calculate_directions(self):
        for i in range(len(self.trajectory)):
            if i!=len(self.trajectory)-1: 
                # ignore first and last
                x1 = self.trajectory[i][0]
                y1 = self.trajectory[i][1]
                x2 = self.trajectory[i+1][0]
                y2 = self.trajectory[i+1][1]
                m = (y2-y1)/(x2-x1)
                self.directions.append(math.atan(m))







def compare_envs(env1, env2, pred_traj, phy_loc, speed_per_traj, with_rotations=True, title_data=''):
    """Compute the ENI metric for two given environments. This implements the main optimization algorithm 
    described in Section 3 of the paper.

    For each evenly spaced position (i.e. triangulation vertex) in env1, find the best matching position 
    in env2 (from random samples of the environment). In practice, env1 is the virtual environment and env2 
    is the physical environment.

    Args:
        env1: a Environment(). The virtual environment.
        env2: a Environment(). The physical environment.
        pred_traj: a Trajectory(). the predicted trajectory (in next X second) in the virtual environment env1.
        with_rotations: Whether or not to consider rotations of the visibility polygons when computing the
                        most compatible physical and virtual configurations.
        title_data: Optional additional information to include in the title of the generated plots. This is
                    useful for keeping track of what a particular ENI score was computed from.
    """
    import time
    start_time = time.time()
    if with_rotations:
        title_data += ' | WITH ROTATIONS'

    print('Comparing {} and {}...'.format(env1.name, env2.name))
    matches = [] # Format is [(match value, env1 position, env2 position), ...]

    # env1_positions_x = [v[0] for v in env1.triangulated_env['vertices']]
    # env1_positions_y = [v[1] for v in env1.triangulated_env['vertices']]
    # env1_polys = env1.random_sampled_visibility_polygons
    # env2_polys = env2.random_sampled_visibility_polygons
    # env2_polys = env2.triangulation_visibility_polygons
    env2_poly = env2.get_poly(phy_loc)

    traj_positions_x = [v[0] for v in pred_traj.trajectory]
    traj_positions_y = [v[1] for v in pred_traj.trajectory]
    rot_threshold = [20,50]
    total_scores = 0
    redirect_path = []
    for i in range(len(traj_positions_x)):
        if i != len(traj_positions_x)-1:
            traj_position_x = traj_positions_x[i]
            traj_position_y = traj_positions_y[i]

            virt_poly = env1.get_poly([traj_position_x,traj_position_y])
            print("pred_traj.directions")
            print(pred_traj.directions)
            direction = pred_traj.directions[i]
            rot_threshold = [-20,20]
            num_rotations = 10
            env2_polys_rotated = []
            rot_per_count = (rot_threshold[1]-rot_threshold[0])/num_rotations
            rots = [direction+rot_threshold[0]+i*rot_per_count for i in range(num_rotations)]
            print("ROT LIST")
            print(rots)
            env2_polys_rotated = []
            best_score = 10000000
            for rot in rots:
                phys_rotated = env2_poly.rotate(rot)
                temp_matches = [virt_poly.minus(phys_rotated),rot]
                if temp_matches[0]<best_score:
                    env2_polys_rotated = temp_matches
                    best_score = temp_matches[0]
            # update env2_poly phy_loc
            angle_radians = math.radians(env2_polys_rotated[1])
            new_x = phy_loc[0] + speed_per_traj * math.cos(angle_radians)
            new_y = phy_loc[1] + speed_per_traj * math.sin(angle_radians)
            phy_loc = [new_x,new_y]
            env2_poly = env2.get_poly(phy_loc)

            redirect_path.append(env2_polys_rotated[1])
            total_scores+=best_score
            print("env2_polys_rotated")
            print(env2_polys_rotated)
            print("redirect_path")
            print(redirect_path)
            print("score")
            print(total_scores)
    avg_score = total_scores/(len(traj_positions_x)-1)
    print("avg_score")
    print(avg_score)
            




    

    # env2_polys_rotated = []
    # rotation_amount = geometry.TWO_PI / num_rotations
    # for i in range(num_rotations):
    #     rotated_by_i = [env2_polygon.rotate(i * rotation_amount) for env2_polygon in env2_polys]
    #     env2_polys_rotated.extend(rotated_by_i)

    # for i in tqdm(range(len(env1_positions_x))):
    #     p = [env1_positions_x[i], env1_positions_y[i]]
    #     if not env1.point_is_legal(p):
    #         pass
    #     else:
    #         env1_vis_poly = env1.get_triangulation_vis_poly(np.array(p))
    #         best_match_val = float('inf')
    #         best_match_tuple = None
    #         if with_rotations:
    #             temp_matches = [(env1_vis_poly.minus(rotated_env2_poly), env1_vis_poly.kernel, rotated_env2_poly.kernel) for rotated_env2_poly in env2_polys_rotated]
    #             temp_matches.sort(key=lambda x: x[0])
    #             if temp_matches[0][0] < best_match_val:
    #                 best_match_val = temp_matches[0][0]
    #                 best_match_tuple = temp_matches[0]
    #             assert best_match_tuple != None
    #             matches.append(best_match_tuple)
    #         else:
    #             temp_matches = [(env1_vis_poly.minus(env2_vis_poly), env1_vis_poly.kernel, env2_vis_poly.kernel) for env2_vis_poly in env2_polys]
    #             temp_matches.sort(key=lambda x: x[0])
    #             matches.append(temp_matches[0])

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # match_vals = [x[0] for x in matches]
    # matches_mean = statistics.mean(match_vals)
    # matches_st_dev = statistics.stdev(match_vals)
    # title_data += ' | Mean score: {} | Std. Dev.: {} | Num CDT sampled points virtual: {} | Num CDT sampled points virtual: {} | Elapsed time: {}'.format(matches_mean, matches_st_dev, len(env1.triangulation_visibility_polygons), len(env2.triangulation_visibility_polygons), elapsed_time)

    # draw_bokeh_plots(matches, env1, env2, pred_traj, title_data)

def draw_bokeh_plots(matches, env1, env2, trajs, title_data):
    """Draw plots to visualize the ENI score of two environments. This method uses the bokeh library
    to create this visualization. This method is very ugly and you probably don't care about it.

    Args:
        env1: The virtual environment.
        env2: The physical environment.
        title_data: Optional additional information to include in the title of the generated plots. This is
                    useful for keeping track of what a particular ENI score was computed from.
    """
    from bokeh.io import curdoc, show
    from bokeh.layouts import row
    from bokeh.models import CustomJS, ColumnDataSource, LinearColorMapper, Grid, LinearAxis, MultiPolygons, Plot, ColorBar, LogColorMapper
    from bokeh.plotting import figure, output_file, save
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
    from bokeh.models import HoverTool
    import matplotlib as mpl
    from bokeh.layouts import gridplot
    from bokeh.models import Range1d

    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")
    # Correspondence plot
    # All credit goes to https://stackoverflow.com/questions/34164587/get-selected-data-contained-within-box-select-tool-in-bokeh
    # Data
    env1_pts_x = []
    env1_pts_y = []
    env2_pts_x = []
    env2_pts_y = []
    traj_pts_x = []
    traj_pts_y = []
    pts_vals = []
    for match in matches:
        env1_pos = match[1]
        if env1.point_is_legal(env1_pos):
            env2_pos = match[2]
            env1_pts_x.append(env1_pos[0])
            env1_pts_y.append(env1_pos[1])
            env2_pts_x.append(env2_pos[0])
            env2_pts_y.append(env2_pos[1])
            pts_vals.append(match[0])
    pts_vals_normalized = np.array([val / max(pts_vals)+1e-10 for val in pts_vals])
    colors = [
        '#%02x%02x%02x' % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(pts_vals_normalized))
    ]
    max_extent = max(abs(env1.min_x), abs(env1.max_x), abs(env1.min_y), abs(env1.max_y)) + 5

    # Environment 1
    s1 = ColumnDataSource(data=dict(x=env1_pts_x, y=env1_pts_y, color=colors))
    correspondence_env1_plot = figure(match_aspect=True, tools=['lasso_select', 'wheel_zoom', 'reset', 'save'], title=env1.name)
    correspondence_env1_plot.patch([p[0] for p in env1.border.pts], [p[1] for p in env1.border.pts], alpha=1, line_color='black', fill_color='white', line_width=2)
    for obs in env1.obstacles:
        correspondence_env1_plot.patch([p[0] for p in obs.pts], [p[1] for p in obs.pts], alpha=1, line_color='black', fill_color='black', line_width=2)
    # draw predicted trajectory on virtual map
    correspondence_env1_plot.patch([p[0] for p in trajs.trajectory], [p[1] for p in trajs.trajectory], alpha=1, line_color='black', fill_color='white', line_width=2)
    
    correspondence_env1_plot.xgrid.grid_line_color = None
    correspondence_env1_plot.ygrid.grid_line_color = None
    correspondence_env1_plot.xaxis.major_tick_line_color = None
    correspondence_env1_plot.xaxis.minor_tick_line_color = None
    correspondence_env1_plot.yaxis.major_tick_line_color = None
    correspondence_env1_plot.yaxis.minor_tick_line_color = None
    correspondence_env1_plot.x_range = Range1d(-max_extent, max_extent)
    correspondence_env1_plot.y_range = Range1d(-max_extent, max_extent)
    print("MAX VALUE IN COLOR BAR: " + str(max(pts_vals)))
    color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=max(pts_vals))
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
    correspondence_env1_plot.add_layout(color_bar, 'left')
    correspondence_env1_plot.circle('x', 'y', source=s1, radius=0.1, fill_color='color', line_color='color')

    # Environment 2
    s2 = ColumnDataSource(data=dict(x=env2_pts_x, y=env2_pts_y))
    s3 = ColumnDataSource(data=dict(x=[], y=[]))
    correspondence_env2_plot = figure(match_aspect=True, tools=['wheel_zoom', 'box_zoom', 'reset', 'save'], title=env2.name)
    correspondence_env2_plot.toolbar.active_drag = None
    correspondence_env2_plot.toolbar.active_scroll = "auto"
    correspondence_env2_plot.patch([p[0] for p in env2.border.pts], [p[1] for p in env2.border.pts], alpha=1, line_color='black', fill_color='white', line_width=2)
    for obs in env2.obstacles:
        correspondence_env2_plot.patch([p[0] for p in obs.pts], [p[1] for p in obs.pts], alpha=1, line_color='black', fill_color='black', line_width=2)
    correspondence_env2_plot.xgrid.grid_line_color = None
    correspondence_env2_plot.ygrid.grid_line_color = None
    correspondence_env2_plot.xaxis.major_tick_line_color = None
    correspondence_env2_plot.xaxis.minor_tick_line_color = None
    correspondence_env2_plot.yaxis.major_tick_line_color = None
    correspondence_env2_plot.yaxis.minor_tick_line_color = None
    correspondence_env2_plot.x_range = Range1d(-max_extent, max_extent)
    correspondence_env2_plot.y_range = Range1d(-max_extent, max_extent)
    correspondence_env2_plot.circle("x", "y", source=s3, radius=0.1, color="firebrick")
    # Fancy javascript to link subplots
    # js pushes selected points into ColumnDataSource of 2nd plot
    # Inspiration for this from a few sources:
    # credit: https://stackoverflow.com/users/1097752/iolsmit via: https://stackoverflow.com/questions/48982260/bokeh-lasso-select-to-table-update
    # credit: https://stackoverflow.com/users/8412027/joris via: https://stackoverflow.com/questions/34164587/get-selected-data-contained-within-box-select-tool-in-bokeh
    s1.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(s1=s1, s2=s2, s3=s3),
            code="""
            var inds = cb_obj.indices;
            var d1 = s1.data;
            var d2 = s2.data;
            var d3 = s3.data;
            d3['x'] = []
            d3['y'] = []
            for (var i = 0; i < inds.length; i++) {
                d3['x'].push(d2['x'][inds[i]])
                d3['y'].push(d2['y'][inds[i]])
            }
            s3.change.emit();
        """,
        ),
    )

    # Histogram
    match_values = [val[0] for val in matches]
    env1_positions = [val[1] for val in matches]
    env2_positions = [val[2] for val in matches]
    match_values_normalized = [val / max(match_values) for val in match_values]
    hist, edges = np.histogram(match_values, density=False, bins=50)
    hist_link_dataset = {}
    for i in range(len(edges)-1):
        bucket_lower = edges[i]
        bucket_upper = edges[i+1]
        if i < len(edges)-2:
            vals = [match for match in matches if match[0] >= bucket_lower and match[0] < bucket_upper]
        else:
            vals = [match for match in matches if match[0] >= bucket_lower and match[0] <= bucket_upper]
        formatted_vals = [[[v[1][0], v[1][1]], [v[2][0], v[2][1]]] for v in vals]
        hist_link_dataset[i] = formatted_vals

    hist_data = {}
    hist_data['y'] = hist
    hist_data['left'] = edges[:-1]
    hist_data['right'] = edges[1:]

    # hist_data['matches'] = hist_link_dataset
    histogram_plot = figure(title='Histogram of Inaccessible Virtual Space', x_axis_label='Area of inaccessible virtual space', y_axis_label='Count')
    # Quad glyphs to create a histogram
    hist_data = ColumnDataSource(hist_data)
    hist_rect = histogram_plot.quad(source=hist_data, bottom=0, top='y', left='left', right='right', color='#2ca25f', fill_alpha=0.7, hover_fill_color='#18703e', hover_fill_alpha=1.0, line_color='black')
    
    connection_circles_env1_data = ColumnDataSource(data=dict(x=[], y=[]))
    connection_circles_env1 = correspondence_env1_plot.circle('x', 'y', source=connection_circles_env1_data, radius=0.1, line_color="red", fill_color='#ff8400')

    connection_circles_env2_data = ColumnDataSource(data=dict(x=[], y=[]))
    connection_circles_env2 = correspondence_env2_plot.circle('x', 'y', source=connection_circles_env2_data, radius=0.1, line_color='blue', fill_color='#ff8400')
    hover_code = """
    const link_data = %s  
    const my_index = cb_data.index.indices;

    var d1 = connection_circles_env1_data.data;
    var d2 = connection_circles_env2_data.data;
    if (my_index.length > 0){
        var index = my_index[0];
        var key_list = Object.keys(link_data)
        var data_to_plot = link_data[index];

        d1['x'] = []
        d1['y'] = []
        d2['x'] = []
        d2['y'] = []
        for (var i = 0; i < data_to_plot.length; i++) {
            var env1_pos = data_to_plot[i][0];
            var env2_pos = data_to_plot[i][1];
            d1['x'].push(env1_pos[0])
            d1['y'].push(env1_pos[1])
            d2['x'].push(env2_pos[0])
            d2['y'].push(env2_pos[1])
        }
    }
    else{
        d1['x'] = []
        d1['y'] = []
        d2['x'] = []
        d2['y'] = []
    }
    connection_circles_env1_data.change.emit();
    connection_circles_env2_data.change.emit();
    """ % hist_link_dataset
    callback = CustomJS(args={'hist_rect': hist_rect.data_source, 'connection_circles_env1_data': connection_circles_env1_data, 'connection_circles_env2_data': connection_circles_env2_data}, code=hover_code)
    # Hover tool with vline mode
    hover = HoverTool(callback=callback,
                      mode='vline',
                      tooltips=[('Count', '@y')])
    histogram_plot.add_tools(hover)

    directory = '/home/yiranpp/Desktop/researchTryout/ENI/img'
    filename = os.path.join(directory,'{}_and_{}_interaction_{}.html'.format(env1.name, env2.name, current_time))
    output_file(filename)

    from bokeh.models import Div
    div = Div(text=title_data, width=200, height=100)
    save(gridplot([correspondence_env1_plot, correspondence_env2_plot, histogram_plot, div], ncols=3))

    print('Finished bokeh plot generation!')

if __name__ == '__main__':
    random.seed(13443)
    cur_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'envs')
    
    virt_grand_demo = Environment(os.path.join(cur_path, 'virt', 'big_square.xml'))
    phys_grand_demo = Environment(os.path.join(cur_path, 'phys', 'big_square.xml'))
    pred_traj = Trajectory('/home/yiranpp/Desktop/researchTryout/ENI/pred_trajs/pred_traj')
    phy_loc = [2,2]
    speed_per_traj = 1
    compare_envs(virt_grand_demo, phys_grand_demo, pred_traj, phy_loc, speed_per_traj, True, 'Best phys for each virt pos.\nArea of virt. nonoverlap')
    
    print('Finished ok')