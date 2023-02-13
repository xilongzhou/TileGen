"""

Your data must be stored in the following way. annotations is the folder for semantic maps. 
annotations_instance is the folder for instance maps.
 

PATH_TO_DATASET:

    full_data

        images
            training
                1.jpg
                2.jpg
                ...
            validation
                1.jpg
                2.jpg
                ...
            
        annotations
            training
                1.png
                2.png
                ...
            validation
                1.png
                2.png
                ...

        annotations_instance
            training
                1.png
                2.png
                ...
            validation
                1.png
                2.png
                ... 



"""


class SceneConfig():
    # TRAIN_DATASETS_TILE = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/BricksDataset_pat2_train_color']
    # TEST_DATASETS_TILE = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/BricksDataset_pat2_val_color']

    TRAIN_DATASETS_GROUND = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/GroundDataset_pat_train']
    TEST_DATASETS_GROUND = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/GroundDataset_pat_val']

    TRAIN_DATASETS_LEATHER = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/LeatherDataset2_2']
    TEST_DATASETS_LEATHER = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/LeatherDataset2_2']

    # TRAIN_DATASETS_STONE = ['/home/grads/z/zhouxilong199213/Projects/Adobe_project_2021S/Dataset/StoneDataset_1']
    # TEST_DATASETS_STONE = ['/home/grads/z/zhouxilong199213/Projects/Adobe_project_2021S/Dataset/StoneDataset_1']

    TRAIN_DATASETS_STONE = ['/home/grads/z/zhouxilong199213/Projects/Adobe_project_21S/Dataset/StoneDataset_1']
    TEST_DATASETS_STONE = ['/home/grads/z/zhouxilong199213/Projects/Adobe_project_21S/Dataset/StoneDataset_1']



    TRAIN_DATASETS_TILE = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/Tiles_Dataset']
    TEST_DATASETS_TILE = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/Tiles_Dataset']

    TRAIN_DATASETS_METAL = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/Metal_Dataset']
    TEST_DATASETS_METAL = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/Metal_Dataset']

    # TRAIN_DATASETS = ['/home/code-base/scratch_space/Server/data/ade_bedroom']
    # TEST_DATASETS = ['/home/code-base/scratch_space/Server/data/ade_bedroom'] 




