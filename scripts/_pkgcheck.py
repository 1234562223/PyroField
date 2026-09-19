import importlib
for m in ["shapely","pyproj","geopandas","requests","skimage","scipy","cv2","matplotlib"]:
    try:
        mod=importlib.import_module(m)
        print("%-12s OK   %s"%(m,getattr(mod,"__version__","?")))
    except Exception as e:
        print("%-12s MISSING"%m)
