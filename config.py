
import argparse
from pathlib import Path

# Source: https://stackoverflow.com/questions/12151306/argparse-way-to-include-default-values-in-help
parser = argparse.ArgumentParser(description='NERV')

'''
###################
# data generation #
###################
'''
parser.add_argument('--dcm_path',
                    type=Path,
                    default=Path('/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data'),
                    help='Location of the folder containing dicom files')

parser.add_argument('--csv_path',
                    type=Path,
                    default=Path('data.csv'),
                    help='Location of the data.csv')

parser.add_argument('--new_spacing',
                    type=float,
                    default=1.,
                    help='Coarse resolution (mm)')

parser.add_argument('--coarse_img_save_name',
                    type=str,
                    default='1mm.npy',
                    help='Save name of the coarse 3-D image file')

parser.add_argument('--coarse_mask_save_name',
                    type=str,
                    default='1mm_mask.npy',
                    help='Save name of the coarse 3-D mask file')

args = parser.parse_args()