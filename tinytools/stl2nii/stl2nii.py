import os


def is_right_file(file: str):
    return file.endswith(".stl")


def file_check(in_dir: str, out_dir: str):
    files = os.listdir(in_dir)
    labels = []
    for file in files:
        if is_right_file(file):
            labels.append(file[:-4])
        else:
            if not file.endswith(".nii.gz"):
                print(f"------`{file}` is not .stl file.")
    niis = []
    stls = []
    outs = []
    for each in labels:
        nii = os.path.join(in_dir, each+".nii.gz")
        stl = os.path.join(in_dir, each+".stl")
        out = os.path.join(out_dir, each+".nii.gz")
        if os.path.exists(nii):
            niis.append(nii)
            stls.append(stl)
            outs.append(out)
    return niis, stls, outs


def convert_stl_nii(in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    niis, stls, outs = file_check(in_dir, out_dir)
    for nii, stl, out in zip(niis, stls, outs):
        ref = slicer.util.loadVolume(nii)
        seg = slicer.util.loadSegmentation(stl)
        converted = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic(
        ).ExportVisibleSegmentsToLabelmapNode(seg, converted, ref)
        slicer.util.saveNode(converted, out)
        slicer.mrmlScene.Clear()
    n = len(outs)
    print(f"{n} files converted in {out_dir}.")
