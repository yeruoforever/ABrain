import os
import SimpleITK as itk
from argparse import ArgumentParser
import pydicom
import logging


def any_none(*args) -> bool:
    for each in args:
        if each is None:
            return True
    return False


def is_head(part):
    return part.value in ["BRAIN", "HEAD"]


def convert_dcm_nii(dcm: str, nii_dir: str):
    series = os.listdir(dcm)
    series = map(lambda x: os.path.join(dcm, x), series)
    series = filter(lambda x: os.path.isdir(x), series)
    reader = itk.ImageSeriesReader()
    for s in series:
        info = get_series_info(s)
        if any_none(*info):
            continue
        part, s_desc, thickness_1, thickness_2 = info
        if not is_head(part):
            continue
        if thickness_1.value != thickness_2.value:
            logging.info(f"{s}")
            logging.info(f"Slice Thickness != ")
            logging.info(f"{thickness_1.value}!={thickness_2.value}")
            continue
        desc = s_desc.value
        if desc not in ["1.25mm STND", "Head STAND 5mm"]:
            continue
        thickness = round(thickness_1.value, 2)
        dicom_names = reader.GetGDCMSeriesFileNames(s)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        subject_id = dcm.split(os.sep)[-1]
        output = os.path.join(nii_dir, f"{thickness}mm")
        if not os.path.exists(output):
            print(f"Directory({output}) not exists, creating...")
            os.makedirs(output)
        output = os.path.join(output, f"{subject_id}.nii.gz")
        itk.WriteImage(image, output)


def get_series_info(dcm_dir: str) -> bool:
    files = os.listdir(dcm_dir)
    files = list(filter(lambda x: x.endswith(".dcm"), files))
    dcm = pydicom.read_file(os.path.join(dcm_dir, files[0]))

    part = dcm.get((0x0018, 0x0015))        # Body Part Examined
    desc = dcm.get((0x0008, 0x103e))        # Series Description
    # spacing = dcm.get(0x00280030)           # Pixel Spacing
    thickness_1 = dcm.get(0x00180050)       # Slice Thickness
    thickness_2 = dcm.get(0x00180088)       # Spacing Between Slices

    return part, desc, thickness_1, thickness_2


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
                pool.apply_async(convert_dcm_nii, (dirs, args.output_dir))
            pool.close()
            pool.join()
    else:
        for dirs in subdirs:
            convert_dcm_nii(dirs, args.output_dir)
