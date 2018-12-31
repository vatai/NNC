"""Figure out xml parsing."""

import xml.etree.ElementTree as ET


def get_bbox(obj):
    """
    Get the bounding box cordinates in an (xmin, xmax, ymin, ymax)
    touple from an object.
    """
    out_template = ['xmin', 'xmax', 'ymin', 'ymax']
    bbox = obj.find('bndbox')
    result = map(bbox.find, out_template)
    result = map(lambda t: int(t.text), result)
    return tuple(result)


def get_sysnet(obj):
    """
    Get the sysnet/wnid (string) from an object.
    """
    name = obj.find('name')
    return name.text


def get_fname_objs(file_path):
    """
    Return the filename and all the object in the file so it can be
    processed by get_bbox().
    """
    root = ET.parse(file_path)
    filename = root.find('filename').text
    objs = root.findall('object')
    return filename, objs


def proc_xml_file(file_path):
    """
    Process a file: return a list of 4 element tuples (see get_bbox),
    one for each object in the file.
    """
    filename, objs = get_fname_objs(file_path)
    bboxes = map(get_bbox, objs)
    sysnets = map(get_sysnet, objs)
    filenames = len(objs) * [filename]
    return zip(filenames, bboxes, sysnets)


FILE = "/home/vatai/tmp/ilsvrc/val/ILSVRC2012_val_00000002.xml"


print(proc_file(FILE))
print("DONE")
