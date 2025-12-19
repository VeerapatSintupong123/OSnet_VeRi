from torchreid import data
import os

class VeRiDataset(data.ImageDataset):
    # All you need to do here is to generate three lists,
    # which are train, query and gallery.
    # Each list contains tuples of (img_path, pid, camid),
    # where
    # - img_path (str): absolute path to an image.
    # - pid (int): person ID, e.g. 0, 1.
    # - camid (int): camera ID, e.g. 0, 1.
    # Note that
    # - pid and camid should be 0-based.
    # - query and gallery should share the same pid scope (e.g. pid=0 in query refers to the same person as pid=0 in gallery).  --> Very important!
    # - train, query and gallery share the same camid scope (e.g. camid=0 in train refers to the same camera as camid=0 in query/gallery).

    def __init__(self, **kwargs):
        PATH_TO_DATASET = "VeRi/"
        FOLDER_PATH, DATASET_TYPES, IMAGES = next(os.walk(f"{PATH_TO_DATASET}")) 
        training_array = []
        testing_array = []
        query_array = []

        for dataset_type in DATASET_TYPES:
            _, _, IMAGES = next(os.walk(f"{FOLDER_PATH}/{dataset_type}/"))

            if dataset_type == "image_train" or dataset_type == "image_gallery":

                labelSet = []
                for idx, image in enumerate(IMAGES):
                    pid = int(image.split('.jpg')[0].split('_')[-1])
                    labelSet.append(pid)
                label_dict = {label: index for index, label in enumerate(set(labelSet))}
                
                for image in IMAGES:
                    cam_id = int(image.split('.jpg')[0].split('_')[-2])
                    car_id = label_dict[int(image.split('.jpg')[0].split('_')[-1])]

                    if dataset_type == "image_gallery":
                        testing_array.append((FOLDER_PATH +  dataset_type + "/" + image, car_id, cam_id))
                    else:
                        training_array.append((FOLDER_PATH +  dataset_type + "/" + image, car_id, cam_id))
                        
            elif dataset_type == "image_query":
                for image in IMAGES:
                    cam_id = int(image.split('.jpg')[0].split('_')[-2])
                    car_id = label_dict[int(image.split('.jpg')[0].split('_')[-1])]
                    query_array.append((FOLDER_PATH +  dataset_type + "/" + image, car_id, cam_id))

            
        train = training_array
        query = query_array
        gallery = testing_array

        super(VeRiDataset, self).__init__(train, query, gallery, **kwargs)