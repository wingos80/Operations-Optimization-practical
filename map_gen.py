import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
parser.add_argument('--sides', type=str, default=None,
                    help='The number of sides for the polygon')

args = parser.parse_args()


# Calculate position of vertices for a regular polygon given the number of sides and that each side has length 1 unit, and the bottom edge is horizontal
def regular_polygon_vertices(n):
    vertices = np.zeros((n, 2))

    for i in range(1,n):
        vertices[i,0] = vertices[i-1,0] + np.cos(2*np.pi*i/n)*10
        vertices[i,1] = vertices[i-1,1] + np.sin(2*np.pi*i/n)*10
        # vertices[i, 0] = np.cos(2 * np.pi * i / n)
        # vertices[i, 1] = np.sin(2 * np.pi * i / n)
    return vertices

sides = int(args.sides)
vertices = regular_polygon_vertices(sides)
plt.scatter(vertices[:,0], vertices[:,1])
plt.axis('equal')
print(vertices)
plt.show()

