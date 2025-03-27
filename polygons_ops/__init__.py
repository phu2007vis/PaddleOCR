import math
def clockwise_sort(points):
    points = points.squeeze()
    # Compute centroid (average of all points)
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)

    # Sort points by angle with respect to the centroid
    points = sorted(points,key=lambda p: math.atan2(p[1] - cy, p[0] - cx), reverse=True)
    return points