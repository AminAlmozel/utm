# import requests
# import json
import geojson
import time
import overpass


i = 0
tags = [
    "amenity=hospital",
    "aeroway=aerodrome",
    "Tag:aeroway=aerodrome",
    "landuse=forest"
]
area = "name=Toulouse"


# api = overpass.API()
api = overpass.API(timeout=600)

tic = time.time()

request = """

(
    area[{area}]->.boundaryarea;
    node[{tag}](area.boundaryarea);
    way[{tag}](area.boundaryarea);
    relation[{tag}](area.boundaryarea);
);
out body;
>;
out skel qt;

out;
""".format(area=area, tag=tags[i])

request ="""
// gather results
(

  // query part for: “aeroway=aerodrome and icao~LOWW”
  area["name"="Toulouse"]->.boundaryarea;
  node[amenity=hospital](area.boundaryarea);
  way[amenity=hospital](area.boundaryarea);
  relation[amenity=hospital](area.boundaryarea);
);
// print results
out geom;
"""
res = api.get(request)

toc = time.time()
print("Time for request: %.3f" % (toc - tic))

# dump as file, if you want to save it in file
with open("test.geojson",mode="w") as f:
  geojson.dump(res,f)

# api.get already returns a FeatureCollection, a GeoJSON type
# res = api.get("""
#     area(bbox:43.179651,5.332591,43.571692,5.779597);
#     area[poly:"50.7 7.1 50.7 7.2 50.75 7.15"];
#     // recurse down to get the nodes, required for the geometry
#     (._;>;);
# """)