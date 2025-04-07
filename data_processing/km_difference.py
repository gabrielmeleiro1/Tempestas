#This was used to make sure I didn't erronously selected turbine clusters that were too close, affecting our overall model performance;
#Since their selection was done manually and our weather data source does not have very grained data (it can go from 1k to 15k of the distance of the data sample), so to make sure I didn't include turbine clusters that were too close.

import math

def calculate_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

locations = [
    {"name": "Flevoland Windplan Groen", "lat": 52.56, "lon": 5.76},
    {"name": "Windpark Noordoostpolder", "lat": 52.721, "lon": 5.583},
    {"name": "Windpark Zeewolde Princess", "lat": 52.85647, "lon": 4.93378},
    {"name": "Alexia Windpark", "lat": 52.8222, "lon": 4.932895},
    {"name": "Groningen Westereems", "lat": 53.4474, "lon": 6.842},
    {"name": "Delfzijl-Noord", "lat": 53.3215, "lon": 6.9701},
    {"name": "Windpark Oostpolder Geefsweer", "lat": 53.321651, "lon": 6.969826},
    {"name": "Borssele I & II", "lat": 51.70278, "lon": 3.07611},
    {"name": "Gemini Wind Farm", "lat": 54.03611, "lon": 5.96306},
    {"name": "Hollandse Kust (Zuid) I & II", "lat": 52.36667, "lon": 4.11667},
    {"name": "Hollandse Kust (Zuid) III & IV Hollandse Kust Noord", "lat": 52.71511, "lon": 4.251},
    {"name": "Eneco Luchterduinen", "lat": 52.40481, "lon": 4.161821},
    {"name": "Fryslan Wind Farm", "lat": 52.99435, "lon": 5.267503},
    {"name": "Anna Mariapolder", "lat": 51.384694, "lon": 4.259278},
    {"name": "Aruba", "lat": 12.474028, "lon": -69.89075},
    {"name": "Drentse Monden en Oostermoer", "lat": 52.9553216, "lon": 6.9352082},
    {"name": "Egmond aan Zee", "lat": 52.6, "lon": 4.416667},
    {"name": "Koegorspolder", "lat": 51.283333, "lon": 3.85},
    {"name": "Kreekraksluis", "lat": 51.416667, "lon": 4.233333},
    {"name": "Kroningswind", "lat": 51.79058, "lon": 4.069921},
    {"name": "Maasvlakte 2", "lat": 51.93607, "lon": 3.999385},
    {"name": "N33", "lat": 53.16511, "lon": 6.894004},
    {"name": "Prinses Amalia", "lat": 52.583333, "lon": 4.2},
    {"name": "Windplan Blauw", "lat": 52.566879, "lon": 5.586317},
    {"name": "Wieringermeer cluster Nuon", "lat": 52.84179, "lon": 5.026437},
    {"name": "Westermeerdijk buitendijks", "lat": 52.70857, "lon": 5.613599},
    {"name": "Westereems", "lat": 53.44617, "lon": 6.790496},
    {"name": "Zeewolde", "lat": 52.34678, "lon": 5.499688}
]

all_distances = [] # List to store tuples: (distance, name1, name2)
num_locations = len(locations)

for i in range(num_locations):
    for j in range(i + 1, num_locations): # Start j from i+1 to avoid self-comparison and duplicates
        loc1 = locations[i]
        loc2 = locations[j]

        # Calculate distance between loc1 and loc2
        distance = calculate_distance(loc1['lat'], loc1['lon'], loc2['lat'], loc2['lon'])

        all_distances.append((distance, loc1['name'], loc2['name']))

all_distances.sort()

top_10_pairs = all_distances[:10]

print("The 10 closest pairs of wind turbine clusters are:")
print("-" * 60) # Separator line
for rank, (dist, name1, name2) in enumerate(top_10_pairs, 1):
    print(f"{rank}. {name1} and {name2}")
    print(f"   Distance: {dist:.2f} km")
    print("-" * 60) # Separator line