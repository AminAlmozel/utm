import overpy
import geojson
import time

api = overpy.Overpass()
print("Sending request")
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

tic = time.time()
res = api.query(request)
toc = time.time()
print(toc - tic)
print("Recieved response")

# dump as file, if you want to save it in file
with open("./test.geo.json",mode="w") as f:
  geojson.dump(res,f)