import config
from utils import GetDataEDA2CSV
from utils_coarse_data_generation import GenerateCoarseImage, GenerateCoarseMask

print('\n\n+++++ Running 1_data_generation.py +++++')
GetDataEDA2CSV(data_path=config.args.dcm_path,
               output_csv_path=config.args.csv_path)  # generate the csv file

GenerateCoarseImage(csv_path=config.args.csv_path,
                    new_spacing=config.args.new_spacing,
                    save_name=config.args.coarse_img_save_name)

_, _ = GenerateCoarseMask(csv_path=config.args.csv_path,
                          new_spacing=config.args.new_spacing,
                          save_name=config.args.coarse_mask_save_name)

print('+++++ Finished running 1_data_generation.py +++++\n\n')
