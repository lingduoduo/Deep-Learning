import json
import os
import urllib.request


def retrieval_url(url, filename):
    if not os.path.exists(filename) and not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} exists! Skip download.")


def get_json_from_file(filename):
    with open(filename, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def print_dict(json_dict, items=5):
    print({x: json_dict[x] for (i, x) in enumerate(json_dict) if i < items})


if __name__== "__main__":
    url = "https://www.ncdc.noaa.gov/cag/national/time-series/110-pcp-12-12-1895-2016.json"
    data_path = "../data"
    filename = f"{data_path}/climate.json"
    retrieval_url(url, filename)
    json_data = get_json_from_file(filename)
    print(f"Json keys: {json_data.keys()}")
    print(print_dict(json_data['data']))

    with open(filename, 'r') as json_file:
        raw_json = json_file.read()
    json_from_string = json.loads(raw_json)
    print(f'Json keys: {json_from_string.keys()}')
    print(print_dict(json_from_string["data"]))

    precip_raw = json_data["data"]
    precipitation = [(x, float(precip_raw[x]['value'])) for x in precip_raw]
    print(precipitation[:5])