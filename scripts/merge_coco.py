import json

def merge_coco(files, output_file):
    merged = {
        "info": {"description": "Merged COCO dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    ann_id = 1
    img_id_map = {}
    img_counter = 1
    seen_files = set()

    for f in files:
        print(f"Reading {f}...")
        data = json.load(open(f, "r"))

        # Copy categories once (all should be identical)
        if not merged["categories"] and "categories" in data:
            merged["categories"] = data["categories"]

        # Process images
        for img in data["images"]:
            fname = img["file_name"]
            if fname in seen_files:
                continue  # skip duplicates
            seen_files.add(fname)

            old_id = img["id"]
            new_id = img_counter
            img_counter += 1

            img_id_map[(f, old_id)] = new_id
            img["id"] = new_id
            merged["images"].append(img)

        # Process annotations
        for ann in data["annotations"]:
            old_img_id = ann["image_id"]
            if (f, old_img_id) not in img_id_map:
                continue
            ann["id"] = ann_id
            ann_id += 1
            ann["image_id"] = img_id_map[(f, old_img_id)]
            merged["annotations"].append(ann)

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged dataset saved to {output_file}")


if __name__ == "__main__":
    # Add all your COCO JSON annotation files here
    coco_files = [
        "labels_my-project-name_2025-10-01-07-08-50.json",
        "labels_my-project-name_2025-10-01-09-21-05.json",
        "labels_my-project-name_2025-10-01-09-47-09.json",
        "labels_my-project-name_2025-10-01-10-45-39.json"  # contains img063
    ]
    merge_coco(coco_files, "dataset/merged_labels.json")
