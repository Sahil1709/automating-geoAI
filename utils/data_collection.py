from pystac_client import Client

STAC_API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

def get_sentinel_data(lat, lon, start, end):
    """
    Get Sentinel-2 data for a given location and time range
    """
    # Search the catalogue
    catalog = Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()

    # Reduce to one per date (there might be some duplicates
    # based on the location)
    items = []
    dates = []
    assetsList = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

        assets = item.get_assets()
        for key, asset in assets.items():
            print(asset.media_type)
            if "image/tiff" in asset.media_type:
                print(f"Found asset: {key}")
                assetsList.append(asset)
    

    print(f"Found {len(items)} items")
    print(f"Found {len(assetsList)} Assets")

    return assetsList