import os
import json
import sys

def convert_json_to_txt(json_directory):
    # List all json files in the directory
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_directory, json_file)
        txt_path = os.path.join(json_directory, os.path.splitext(json_file)[0] + '.txt')

        # Read the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Write to the TXT file
        with open(txt_path, 'w') as f:
            for obj in data:
                x = obj['psr']['position']['x']
                y = obj['psr']['position']['y']
                z = obj['psr']['position']['z']
                dx = obj['psr']['scale']['x']
                dy = obj['psr']['scale']['y']
                dz = obj['psr']['scale']['z']
                yaw = obj['psr']['rotation']['z']
                category_name = obj['obj_type']

                line = f"{x} {y} {z} {dx} {dy} {dz} {yaw} {category_name}\n"
                f.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_json_to_txt.py <json_directory>")
        sys.exit(1)

    json_directory = sys.argv[1]

    if not os.path.isdir(json_directory):
        print(f"Error: {json_directory} is not a valid directory.")
        sys.exit(1)

    convert_json_to_txt(json_directory)
