import xml.etree.ElementTree as ET  

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotation_data = {
        "folder": root.find("folder").text,
        "filename": root.find("filename").text,
        "size": {
            "width": int(float(root.find("size/width").text)),
            "height": int(float(root.find("size/height").text))
        },
        "objects": []
    }
    
    for obj in root.findall("object"):
        obj_data = {
            "name": obj.find("name").text,
            "bndbox": {
                "xmin": int(float(obj.find("bndbox/xmin").text)),
                "ymin": int(float(obj.find("bndbox/ymin").text)),
                "xmax": int(float(obj.find("bndbox/xmax").text)),
                "ymax": int(float(obj.find("bndbox/ymax").text))
            }
        }
        annotation_data["objects"].append(obj_data)
    
    return annotation_data