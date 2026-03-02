# pip install python-geohash  (package name can vary by distro)
import geohash
import math

# encode
lat, lon = 40.741895, -73.989308
g = geohash.encode(lat, lon, precision=7)
print(g)

lat, lon = geohash.decode(g)
print(lat, lon)

# decode
def tile_bbox(z: int, x: int, y: int):
    # Web Mercator tile bbox in lat/lon
    n = 2 ** z

    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0

    def lat_from_y(y_):
        # inverse of web mercator
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_ / n)))
        return math.degrees(lat_rad)

    lat_max = lat_from_y(y)
    lat_min = lat_from_y(y + 1)
    return (lat_min, lon_min, lat_max, lon_max)

z, x, y = 10, 301, 385
print(tile_bbox(z, x, y))