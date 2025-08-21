from pyproj import Proj
import numpy as np

class CoordinateTransformer:
    __utm_points = np.array(
        [
            [355608.06, 2769698.6],
            [355375.06, 2770108.6],
            [355367.06, 2770328.6],
            [355323.06, 2769555.6],
            [355646.06, 2769966.6],
        ]
    )

    __sim_points = np.array(
        [
            [1000008.0, 999709.0],
            [999775.0, 1000119.0],
            [999767.0, 1000339.0],
            [999723.0, 999566.0],
            [1000046.0, 999977.0],
        ]
    )

    def __init__(
        self,
        utm_points=__utm_points,
        sim_points=__sim_points,
        zone_number=51,
        zone_letter="N",
    ):
        """Create a coordinate transformer between UTM and simulation coordinates.

        Args:
            utm_points (np.ndarray): UTM reference points array, shape (N, 2)
            sim_points (np.ndarray): Corresponding simulation points array, shape (N, 2)
            zone_number (int): UTM zone number
            zone_letter (str): UTM zone letter
        """
        self.zone_number = zone_number
        self.zone_letter = zone_letter
        self.scale, self.rotation, self.translation = self._calculate_transformation(
            utm_points, sim_points
        )

    def _calculate_transformation(self, utm_points, sim_points):
        """Calculate the transformation parameters between UTM and simulation coordinates."""
        utm_center = np.mean(utm_points, axis=0)
        sim_center = np.mean(sim_points, axis=0)
        utm_centered = utm_points - utm_center
        sim_centered = sim_points - sim_center

        scale = np.linalg.norm(sim_centered) / np.linalg.norm(utm_centered)
        H = np.dot(utm_centered.T, sim_centered) / len(utm_points)
        U, _, Vt = np.linalg.svd(H)
        rotation = np.dot(Vt.T, U.T)
        translation = sim_center - scale * np.dot(utm_center, rotation.T)

        return scale, rotation, translation

    def to_lonlat(self, x, y, z=0):
        """Convert simulation coordinates (x, y, z) to geographic coordinates (lon, lat, alt).

        Args:
            x (float): Simulation coordinate x
            y (float): Simulation coordinate y
            z (float): Simulation coordinate z

        Returns:
            np.ndarray: (lon, lat, alt)
        """
        sim_point = np.array([x, y])
        utm_point = np.dot((sim_point - self.translation) / self.scale, self.rotation)

        is_northern = self.zone_letter.upper() == "N"
        proj_utm = Proj(
            proj="utm", zone=self.zone_number, ellps="WGS84", north=is_northern
        )

        lon, lat = proj_utm(utm_point[0], utm_point[1], inverse=True)
        return np.array([lon, lat, z])

    def to_xyz(self, lon, lat, alt=0):
        """Convert geographic coordinates (lon, lat, alt) to simulation coordinates (x, y, z).

        Args:
            lon (float): Longitude
            lat (float): Latitude
            alt (float): Altitude

        Returns:
            np.ndarray: (x, y, z)
        """
        proj_utm = Proj(
            proj="utm", zone=self.zone_number, ellps="WGS84", north=(lat >= 0)
        )
        easting, northing = proj_utm(lon, lat)

        utm_point = np.array([easting, northing])
        sim_point = self.scale * np.dot(utm_point, self.rotation.T) + self.translation
        return np.array([sim_point[0], sim_point[1], alt])
