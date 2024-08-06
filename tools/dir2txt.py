import glob
import os


def save_image_paths_to_txt(dir_root, subset):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    image_paths = []
    root_imgs = os.path.join(dir_root, 'images')
    image_dir = os.path.join(root_imgs, subset)
    output_txt_file = os.path.join(dir_root, subset + '.txt')
    for ext in image_extensions:
        for image_path in glob.glob(os.path.join(image_dir, '**', ext), recursive=True):
            # relative_path = os.path.relpath(image_path, dir_root)
            image_paths.append(image_path)
    with open(output_txt_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')


dir_root = '/home/Huangzhe/Test/fire'

save_image_paths_to_txt(dir_root, subset='test')
