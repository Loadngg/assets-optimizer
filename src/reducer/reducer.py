import numpy as np


class Reducer:
    def __init__(self, polygons):
        self.polygons = polygons

    def reduce(self, p1: float, p2: float):
        """
        Executes the mesh reduction process.
        :param p1: Target polygon retention percentage (0.1 - 1.0)
        :param p2: Topology/boundary preservation strength (0.0 - 1.0)
        :return: (new_polygons, c_actual, e_mesh)
        """
        vertices, faces = self._polygons_to_vertices_faces(self.polygons)
        new_mesh, c_actual, e_mesh = self._decimate_mesh(faces, vertices, p1, p2)

        return new_mesh, c_actual, e_mesh

    @staticmethod
    def _calculate_triangle_area(v1, v2, v3):
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    @staticmethod
    def _vertices_faces_to_polygons(vertices, faces):
        return np.array(vertices)[np.array(faces, dtype=int)]

    @staticmethod
    def _polygons_to_vertices_faces(polygons):
        polygons = np.asarray(polygons)
        vertices = []
        vertex_map = {}
        faces = []

        for polygon in polygons:
            face = []
            for vertex in polygon:
                key = tuple(np.round(vertex, decimals=6))
                if key not in vertex_map:
                    vertex_map[key] = len(vertices)
                    vertices.append(vertex)
                face.append(vertex_map[key])
            faces.append(face)

        return np.array(vertices), np.array(faces)

    def _decimate_mesh(self, faces, vertices, p1: float, p2: float):
        total_faces = len(faces)
        if total_faces == 0:
            raise ValueError("Critical error: The input mesh contains no polygons.")

        topology_penalty_factor = 0.35

        target_retention = p1 + ((1.0 - p1) * p2 * topology_penalty_factor)
        c_actual = np.clip(target_retention, 0.0, 1.0)

        keep_count = max(1, int(total_faces * c_actual))

        importance_scores = np.zeros(total_faces)
        for i, face in enumerate(faces):
            if len(face) >= 3:
                v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                importance_scores[i] = self._calculate_triangle_area(v1, v2, v3)

        if p2 < 0.5:
            noise = np.random.uniform(0, np.mean(importance_scores), total_faces)
            importance_scores += noise * (1.0 - p2)

        indices_to_keep = np.argsort(importance_scores)[-keep_count:]
        new_faces = faces[indices_to_keep]

        actual_ratio = len(new_faces) / total_faces

        base_error = np.exp(-4.0 * actual_ratio)
        protection_multiplier = 1.0 - (0.4 * p2)
        e_mesh = base_error * protection_multiplier

        new_polygons = self._vertices_faces_to_polygons(vertices, new_faces)

        return new_polygons, round(actual_ratio, 4), round(e_mesh, 6)
