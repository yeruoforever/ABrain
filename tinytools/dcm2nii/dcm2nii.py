import os
import SimpleITK as itk
from argparse import ArgumentParser
import pydicom
import logging
from typing import *


def any_none(*args) -> bool:
    for each in args:
        if each is None:
            return True
    return False


def is_head(part):
    return part.value in ["BRAIN", "HEAD"]


# def convert_dcm_nii(dcm: str, nii_dir: str):
#     series = os.listdir(dcm)
#     series = map(lambda x: os.path.join(dcm, x), series)
#     series = filter(lambda x: os.path.isdir(x), series)
#     reader = itk.ImageSeriesReader()
#     for s in series:
#         info = get_series_info(s)
#         if any_none(*info):
#             continue
#         part, s_desc, thickness_1, thickness_2 = info
#         if not is_head(part):
#             continue
#         if thickness_1.value != thickness_2.value:
#             logging.info(f"{s}")
#             logging.info(f"Slice Thickness != ")
#             logging.info(f"{thickness_1.value}!={thickness_2.value}")
#             continue
#         desc = s_desc.value
#         if desc not in ["1.25mm STND", "Head STAND 5mm"]:
#             continue
#         thickness = round(thickness_1.value, 2)
#         dicom_names = reader.GetGDCMSeriesFileNames(s)
#         reader.SetFileNames(dicom_names)
#         image = reader.Execute()
#         subject_id = dcm.split(os.sep)[-1]
#         output = os.path.join(nii_dir, f"{thickness}mm")
#         if not os.path.exists(output):
#             print(f"Directory({output}) not exists, creating...")
#             os.makedirs(output)
#         output = os.path.join(output, f"{subject_id}.nii.gz")
#         itk.WriteImage(image, output)


def get_series_info(dcm_dir: str) -> pydicom.FileDataset:
    files = os.listdir(dcm_dir)
    files = list(filter(lambda x: x.endswith(".dcm"), files))
    dcm = pydicom.dcmread(os.path.join(dcm_dir, files[0]))
    return dcm


def condition(dcm: pydicom.FileDataset) -> bool:
    series_desc = dcm.get((0x0008, 0x103E))
    if series_desc:
        desc: str = series_desc.value
        desc = desc.upper()
        if "SAG" in desc or "COR" in desc:
            return True
    return False


def name_map(dcm: pydicom.FileDataset, out_dir: str) -> str:
    patient_id = dcm.get((0x0010, 0x0020)).value
    series_desc = dcm.get((0x0008, 0x103E)).value
    file_name = os.path.join(out_dir, patient_id, f"{series_desc}.nii.gz")
    return file_name


def convert_dcm_nii(dcm: str, nii_dir: str, condition: Callable, name_map: Callable):
    series = os.listdir(dcm)
    series = map(lambda x: os.path.join(dcm, x), series)
    series = filter(lambda x: os.path.isdir(x), series)
    reader = itk.ImageSeriesReader()
    for s in series:
        info = get_series_info(s)
        if not condition(info):
            continue
        file_name = name_map(info, nii_dir)
        dicom_names = reader.GetGDCMSeriesFileNames(s)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        out_dir = os.path.dirname(file_name)
        if not os.path.exists(out_dir):
            print(f"Directory({out_dir}) not exists, creating...")
            os.makedirs(out_dir)
        itk.WriteImage(image, file_name)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("dcm_dirs", type=str)
    argparser.add_argument("output_dir", type=str)
    argparser.add_argument("--workers", default=8, type=int)
    args = argparser.parse_args()

    subdirs = os.listdir(args.dcm_dirs)
    subdirs = map(lambda x: os.path.join(args.dcm_dirs, x), subdirs)
    subdirs = list(filter(lambda x: os.path.isdir(x), subdirs))

    print(f"{len(subdirs)} subject images will be converted.")

    if args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(args.workers) as pool:
            for dirs in subdirs:
                pool.apply_async(
                    convert_dcm_nii, (dirs, args.output_dir, condition, name_map)
                )
            pool.close()
            pool.join()
    else:
        for dirs in subdirs:
            convert_dcm_nii(dirs, args.output_dir, condition, name_map)
