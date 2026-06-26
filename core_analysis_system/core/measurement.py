import numpy as np


class Measurement:

    @staticmethod
    def fracture_porosity(fracture_area, total_area):
        return fracture_area / total_area if total_area > 0 else 0

    @staticmethod
    def fracture_density(total_length, area):
        return total_length / area if area > 0 else 0

    @staticmethod
    def fracture_linear_density(count, length):
        return count / length if length > 0 else 0

    @staticmethod
    def pore_area_ratio(pore_area, total_area):
        return pore_area / total_area if total_area > 0 else 0

    @staticmethod
    def avg_pore_diameter(diameters):
        return np.mean(diameters) if len(diameters) > 0 else 0
