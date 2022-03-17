__author__ = "Yarden Frenkel"
import pandas as pd
import tensorflow
import h5py
import cv2
import numpy as np

IMAGE_SIZE = (32, 32)
NUM_OF_CATEGORY = 7

FONT_TO_NUMBER_MAP = {
    'Alex Brush': 0,
    'Michroma': 1,
    'Raleway': 2,
    'Russo One': 3,
    'Open Sans': 4,
    'Ubuntu Mono': 5,
    'Roboto': 6
}
NUMBER_TO_FONT_MAP = {
    0: 'Alex Brush',
    1: 'Michroma',
    2: 'Raleway',
    3: 'Russo One',
    4: 'Open Sans',
    5: 'Ubuntu Mono',
    6: 'Roboto'
}


def get_crop_wraped_image(image: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    try:
        cropped_image = np.array(image, copy=True)
        pt_a, pt_b, pt_c, pt_d = boundaries
        width_ad = np.sqrt(((pt_a[0] - pt_d[0]) ** 2) + ((pt_a[1] - pt_d[1]) ** 2))
        width_bc = np.sqrt(((pt_b[0] - pt_c[0]) ** 2) + ((pt_b[1] - pt_c[1]) ** 2))
        max_width = max(int(width_ad), int(width_bc))

        height_ab = np.sqrt(((pt_a[0] - pt_b[0]) ** 2) + ((pt_a[1] - pt_b[1]) ** 2))
        height_cd = np.sqrt(((pt_c[0] - pt_d[0]) ** 2) + ((pt_c[1] - pt_d[1]) ** 2))
        max_height = max(int(height_ab), int(height_cd))

        input_pts = np.float32([pt_a, pt_b, pt_c, pt_d])
        output_pts = np.float32([[0, 0],
                                 [max_width, 0],
                                 [max_width, max_height],
                                 [0, max_height]
                                 ])
        m = cv2.getPerspectiveTransform(input_pts, output_pts)
        return cv2.warpPerspective(cropped_image, m, (max_width, max_height), flags=cv2.INTER_LINEAR)
    except Exception as e:
        raise type(e)(f'failed to get_crop_wraped_image due to {e}')


def build_test_letters_data_set(data_base) -> list:
    try:
        data_set = []
        images_names = list(data_base.keys())
        for image_name in images_names:  # for each image name
            index = 0
            current_image = data_base[image_name][:]  # the current image
            current_image_text = data_base[image_name].attrs['txt'].astype('U')  # list of the image words
            current_image_characters_bounding_box = data_base[image_name].attrs['charBB'].transpose()
            for i, word in enumerate(current_image_text):  # for each word
                for j, letter in enumerate(word):  # for each letter do:
                    char_bb = current_image_characters_bounding_box[index]
                    cropped_image = get_crop_wraped_image(current_image, char_bb)
                    data_set.append(
                        [image_name, j, letter, i, word, cropped_image.shape[0], cropped_image.shape[1], cropped_image])
                    index += 1
        return data_set
    except Exception as e:
        raise type(e)(f'failed to build_letters_data_set due to {e}')


def resize_image_to_shape(image: np.ndarray, shape: tuple, method: str = 'bilinear') -> np.array:
    try:
        return tensorflow.image.resize(image, [shape[0], shape[1]], antialias=True, method=method).numpy().astype(
            'int32')
    except Exception as e:
        raise type(e)(f'failed to resize_image_to_shape due to {e}')


def convert_image_to_grayscale(image: np.ndarray) -> np.ndarray:
    try:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_tensor = tensorflow.constant(img)
        img_expanded = tensorflow.expand_dims(img, 2).numpy()
        return img_expanded
    except Exception as e:
        raise type(e)(f'failed to convert_image_to_grayscale due to {e}')


def prepare_data_frame_to_test(test_df: pd.DataFrame) -> pd.DataFrame:
    try:
        test_df.columns = ['image_title', 'letter_index', 'letter', 'word_index', 'word', 'image_height', 'image_width',
                           'image']
        test_df['image'] = test_df['image'].apply(lambda img: convert_image_to_grayscale(img))
        test_df['image'] = test_df['image'].apply(lambda img: resize_image_to_shape(img, IMAGE_SIZE))
        return test_df
    except Exception as e:
        raise type(e)(f'failed to prepare_data_frame_to_test, due to: {e}')


def get_group_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    try:
        grouped = test_df.groupby(['image_title', 'word_index'])
        aggregate = list((k, v["predictions"].sum()) for k, v in grouped)
        agg_res = pd.DataFrame(aggregate, columns=["word", "predictions"])
        agg_res['pred_val'] = agg_res['predictions'].apply(lambda arr: arr.argmax())
        agg_res['image_title'] = agg_res['word'].apply(lambda x: x[0])
        agg_res['word_index'] = agg_res['word'].apply(lambda x: x[1])
        agg_res.drop(['word', 'predictions'], axis=1, inplace=True)
        return agg_res
    except Exception as e:
        raise type(e)(f'failed to get_group_predictions, due to: {e}')


if __name__ == '__main__':
    print("\nOpen The Dataset...")
    db = h5py.File('./SynthText_test.h5', 'r')
    print("\nLoading The Dataset...")
    dataset = build_test_letters_data_set(data_base=db['data'])
    df = pd.DataFrame(dataset)
    print("\nPreparing The Dataset...")
    test = prepare_data_frame_to_test(df)
    test_images = test['image'].values
    test_images = np.stack(test_images)
    test_images = test_images / 255
    print("\nLoading The Model...")
    model = tensorflow.keras.models.load_model('./model.h5')
    print("\nPredicting...")
    predictions = model.predict(test_images)
    test['predictions'] = list(predictions)
    pred_agg = get_group_predictions(test)
    result = pd.merge(test, pred_agg, on=['image_title', 'word_index'])
    print("\nPreparing 'Results.csv' File...")
    result = result[['image_title', 'letter', 'pred_val']]
    result['font_name'] = result['pred_val'].apply(lambda x: NUMBER_TO_FONT_MAP[x])
    dummies = pd.get_dummies(result['font_name'])
    result = pd.concat([result.drop('pred_val', axis=1), dummies], axis=1)
    result = result[
        ['image_title', 'letter', 'font_name', 'Raleway', 'Open Sans', 'Roboto', 'Ubuntu Mono', 'Michroma', 'Alex Brush'
            , 'Russo One']]
    result.columns = ['image', 'char', 'font_name', 'Raleway', 'Open Sans', 'Roboto', 'Ubuntu Mono', 'Michroma',
                      'Alex Brush', 'Russo One']
    result.to_csv('results.csv')
    print("\nDone!")
    print("\nYou Can Watch The 'results.csv' In The Containing Folder")
