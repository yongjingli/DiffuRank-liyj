import json
import pickle


def debug_items():
    splatter_lvis_path = "/home/pxn-lyj/Egolee/data/cap3d_data/lvis-annotations-filtered.json"
    cap3d_lvis_path = "/home/pxn-lyj/Egolee/data/cap3d_data_lvis/compressed_files_info.pkl"
    cap3d_all_path = "/home/pxn-lyj/Egolee/data/cap3d_data/compressed_files_info.pkl"
    # COMPRESS_IDS = list(range(68))
    COMPRESS_IDS = [1]

    with open(splatter_lvis_path) as f:
        splatter_lvis_paths = json.load(f)
    print("splatter_lvis_paths:", len(splatter_lvis_paths))  # 44798

    with open(cap3d_all_path, 'rb') as f:
        cap3d_all_data = pickle.load(f)
        compress_names = ["compressed_imgs_perobj_{}".format(str(i).zfill(2)) for i in COMPRESS_IDS]
        cap3d_all_paths = []
        for compress_name in compress_names:
            cap3d_all_paths += cap3d_all_data[compress_name]
    print("cap3d_all_paths:", len(cap3d_all_paths))    # 15000

    with open(cap3d_lvis_path, 'rb') as f:
        cap3d_lvis_data = pickle.load(f)
    cap3d_lvis_paths = cap3d_lvis_data['compressed_diffurank_lvis_00']
    print("cap3d_lvis_data:", len(cap3d_lvis_paths))   # 5952   单个包的lvis为5952,总数约为 5952 * 68 = 404736

    intersection = list(set(splatter_lvis_paths) & set(cap3d_all_paths))
    print("intersection:", len(intersection))   # 857  # 在15000的数量里有857，看来质量过滤的还挺多的

    intersection2 = list(set(splatter_lvis_paths) & set(cap3d_lvis_paths))
    print("intersection2:", len(intersection2))  # 5808   # 在5952中有5808， zero123与cap3d的lvis重叠还是挺大的

    # pip install objaverse
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import objaverse.xl as oxl
    import objaverse
    objaverse_lvis_annotations = objaverse.load_lvis_annotations()
    abjaverse_lvis_all_paths = []
    for name in objaverse_lvis_annotations:
        abjaverse_lvis_all_paths += objaverse_lvis_annotations[name]
    print("abjaverse_lvis_all_paths:", len(abjaverse_lvis_all_paths))  # 46207
    intersection3 = list(set(splatter_lvis_paths) & set(abjaverse_lvis_all_paths))   # 包含splatter_lvis_paths, 而objaverse-x会多2k左右lvis
    # intersection3 = list(set(cap3d_all_paths) & set(lvis_all_paths))
    print("abjaverse_lvis_all_paths:", len(intersection3))   # 44798


if __name__ == "__main__":
    print("Start")
    debug_items()
    print("End")
