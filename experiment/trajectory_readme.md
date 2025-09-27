# Trajectory Transformation Tool

A Python tool for transforming drone trajectories from simulation coordinates to motion capture system coordinates suitable for laboratory use.

## Overview

This tool reads drone trajectory data from GeoJSON files, transforms them from geographical coordinates to a target coordinate system, and outputs waypoints that can be used with motion capture systems in laboratory environments.

## Features

- Reads trajectory data from GeoJSON files
- Transforms coordinates from geographical (WGS84) to UTM projection
- Projects trajectories to fit within specified bounding boxes
- Converts LineString geometries to discrete waypoints
- Outputs waypoints in (x, y, z) format suitable for motion capture systems

## Requirements

```bash
pip install geopandas shapely
```

## File Structure

```
├── read_waypoints.py          # Main transformation script
├── plot/
│   ├── selected_missions.geojson  # Trajectory data
│   └── kaust_bbox.geojson         # Bounding area definition
└── README.md
```

## Input Files

### selected_missions.geojson
Contains the drone trajectory data as LineString geometries. Each feature represents a complete mission trajectory for one drone.

### kaust_bbox.geojson
Defines the bounding area used in simulation. This is used as the source coordinate system for transformation.

## Usage

Simply run the main script:

```bash
python read_waypoints.py
```

The script will:
1. Load trajectory data from `plot/selected_missions.geojson`
2. Load bounding box from `plot/kaust_bbox.geojson`
3. Transform coordinates from WGS84 to UTM Zone 37N (EPSG:32637)
4. Project trajectories to fit within a 100x100 unit target area
5. Convert each trajectory to waypoints and print them

## Output Format

The script outputs waypoints for each drone trajectory in the following format:

```
[Number of waypoints]
x1, y1, z1
x2, y2, z2
...
xn, yn, zn

[Next drone trajectory]
...
```

Where:
- `x, y` are the horizontal coordinates in the target coordinate system
- `z` is the altitude (defaults to 0 if not specified in the source data)

## Coordinate Transformation

The transformation process involves:

1. **CRS Conversion**: Trajectories are converted from WGS84 (EPSG:4326) to UTM Zone 37N (EPSG:32637)
2. **Bounding Box Projection**: Coordinates are scaled and translated to fit within the target bounding box (0, 0, 100, 100)
3. **Affine Transformation**: Uses a 2D affine transformation matrix to map source coordinates to target coordinates

## Visualization

You can visualize the input trajectory and bounding area files using [geojson.io](https://geojson.io):

1. Open [geojson.io](https://geojson.io) in your browser
2. Drag and drop your `.geojson` files onto the map
3. View and inspect your trajectories and bounding areas

## Customization

### Changing Target Bounding Box

Modify the `tgt_bbox` variable in the `main()` function:

```python
tgt_bbox = (0, 0, 100, 100)  # (minx, miny, maxx, maxy)
```

### Changing File Paths

Update the file paths in the `main()` function:

```python
filepath = "path/to/your/selected_missions.geojson"
# and
filepath = "path/to/your/kaust_bbox.geojson"
```

### Changing Coordinate Reference System

Modify the EPSG code in the `read_geojson()` function:

```python
transformed = gdf.to_crs(epsg=YOUR_EPSG_CODE).geometry
```

## Functions

### `main()`
Main execution function that orchestrates the entire transformation process.

### `read_geojson(filepath)`
Reads a GeoJSON file and transforms it from WGS84 to UTM coordinates.

### `linestring_to_waypoints(linestring)`
Converts a Shapely LineString geometry to a list of (x, y, z) waypoints.

### `project_to_bbox(geoms, src_bbox, tgt_bbox)`
Projects geometries from a source bounding box to a target bounding box using affine transformation.

## Notes

- The script assumes trajectories are stored as LineString geometries in the GeoJSON files
- Z-coordinates default to 0 if not present in the source data
- The transformation preserves the relative spatial relationships between trajectories
- UTM Zone 37N is used as it's appropriate for the KAUST region in Saudi Arabia

## Troubleshooting

**FileNotFoundError**: Ensure the GeoJSON files are in the correct `plot/` directory relative to the script.

**CRS Issues**: If you encounter coordinate reference system errors, verify that your GeoJSON files contain valid geographical coordinates.

**Empty Output**: Check that your GeoJSON files contain LineString geometries with coordinate data.