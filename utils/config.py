# ----------------------running settings-------------------------- #
cp_data     = True       # using vqa-cp or not
version     = 'v2'      # 'v1' or 'v2'
train_set   = 'train'   # 'train' or 'train+val'
loss_type   = 'ce_margin'  # 'ce' or 'bce' or 'ce_margin'
in_memory   = False     # load all the image feature in memory

# ----------------------running settings-------------------------- #
entropy = 4.5
scale = 16
alpha = 0.5
temp = 0.2
use_cos = True
sc_epoch = 30
bias_inject = True
learnable_margins = True
randomization = True
supcon = True
dataset = 'slake'

image_dataset       = 'mscoco'
task                = 'OpenEnded' if not cp_data else 'vqacp'
test_split          = 'test2015'    # 'test-dev2015' or 'test2015'
min_occurence       = 9             # answer frequency less than min will be omitted

# ----------------------preprocess image config------------------ #
num_fixed_boxes         = 36        # max number of object proposals per image
output_features         = 2048      # number of features in each object proposal

main_path = None
qa_path = None
bottom_up_path = None
glove_path = None
ids_path = None
image_path = None
rcnn_path = None
cache_root = None
dict_path = None
glove_embed_path = None
min_occurence = 0
max_question_len = 21
trainval_num_images = 0
test_num_images = 0

def update_paths(dataset):
    global main_path, qa_path, bottom_up_path, glove_path, trainval_num_images, test_num_images, min_occurence
    global ids_path, image_path, rcnn_path, cache_root, dict_path, glove_embed_path, max_question_len

    main_path = f'./data/{dataset}'
    qa_path = main_path
    bottom_up_path = f'./data/{dataset}/detection_features/'
    glove_path = f'./data/glove.6B.300d.txt'

    ids_path = f'./data/{dataset}'
    image_path = f'./data/{dataset}/image'

    rcnn_path = f'./data/{dataset}/rcnn/'
    cache_root = f'./data/{dataset}'
    dict_path = f'{qa_path}/dictionary.json'
    glove_embed_path = f'{main_path}/glove6b_init.npy'

    if dataset.startswith('slake'):
        max_question_len = 21
    elif dataset.startswith('vqacp-v2'):
        max_question_len = 23

    if dataset == 'slake':
        trainval_num_images     = 546    # number of images for train and val
        test_num_images         = 96     # number of images for testing
    elif dataset == 'slake-cp':
        trainval_num_images     = 544
        test_num_images         = 511
    elif dataset == 'vqacp-v2':
        trainval_num_images     = 120932
        test_num_images         = 98226
    elif dataset == 'gqaood':
        trainval_num_images     = 72140
        test_num_images         = 388