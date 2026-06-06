import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Viewer:
    def __init__(self, mesh):
        self.mesh = mesh

    @staticmethod
    def _get_polygons(mesh, print_dots=False):
        if print_dots:
            for polygon in mesh:
                for dot in polygon:
                    print(f'x: {dot[0]} y: {dot[1]} z: {dot[2]}')
                print()

        poly_collection = Poly3DCollection(
            mesh, alpha=0.7, facecolors='white',
            edgecolors='black', linewidths=1)
        return poly_collection

    def run(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-15, 15)

        ax.add_collection3d(self._get_polygons(self.mesh))

        plt.show()
