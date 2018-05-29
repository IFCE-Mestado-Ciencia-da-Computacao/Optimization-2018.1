import numpy as np
import pylab as plt


def plot(gamma, points=np.array([]), title="Title", limits=(-10, 10), size=(10, 10), contours=False):
    row = np.linspace(*limits, 50)
    X, Y = np.meshgrid(row, row)
    Z = 3 / 2 * (gamma * X ** 2 + Y ** 2)

    plt.figure(figsize=size)

    CS = plt.contour(X, Y, Z, 15, colors='k')
    if contours:
        plt.clabel(CS, fontsize=9, inline=1)
    plt.title(title)

    if (len(points) > 0):
        pointsx = points[:, 0]
        pointsy = points[:, 1]
        plt.scatter(pointsx, pointsy, color='white', linewidth='1', edgecolor='black')

    for i in range(1, len(points)):
        annotation = ''
        if i < 4:
            annotation = "$x_" + str(i - 1) + "$"

        plt.annotate(annotation, xy=points[i], xytext=points[i - 1],
                     arrowprops={'arrowstyle': '->', 'lw': 1},
                     va='center', ha='center')

    return plt

