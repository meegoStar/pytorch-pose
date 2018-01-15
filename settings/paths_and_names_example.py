from templates import PathsAndNames

paths_names_instance = PathsAndNames()

paths_names_instance.STACKED_HG_WEIGHTS_PATH = '/home/ubuntu/cvlab/meego/pytorch-pose/pretrained_weights/hg_s8_b1/model_best.pth.tar'

paths_names_instance.JHMDB_RGB_ROOT = '/home/ubuntu/data/JHMDB/Rename_Images'
paths_names_instance.JHMDB_CROPPED_RGB_ROOT = '/home/ubuntu/data/Sub_JHMDB/crop_frames'
paths_names_instance.JHMDB_TRAIN_DICT_PATH = '/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/get_train_test_split/dic_train.pickle'
paths_names_instance.JHMDB_TEST_DICT_PATH = '/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/get_train_test_split/dic_test.pickle'

paths_names_instance.PENNACTION_RGB_ROOT = '/home/ubuntu/data/PennAction/Penn_Action/frames'
paths_names_instance.PENNACTION_CROPPED_RGB_ROOT = '/home/ubuntu/data/PennAction/Penn_Action/one_person_img'
paths_names_instance.PENNACTION_DICT_PATH = '/home/ubuntu/data/PennAction/Penn_Action/train_test_split'

paths_names_instance.CHECKPOINT_NAME = 'records/pose_stream/checkpoint.pth.tar'
paths_names_instance.BEST_MODEL_NAME = 'records/pose_stream/model_best.pth.tar'

paths_names_instance.TRAIN_CSV_NAME = 'records/pose_stream/training.csv'
paths_names_instance.TEST_CSV_NAME = 'records/pose_stream/testing.csv'
paths_names_instance.CSV_LOG_NAME = 'records/pose_stream/log.csv'
