from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pandas as pd
import numpy as np
import warnings


def fxn():
    warnings.warn("Future", FutureWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Создаем модель с архитектурой VGG19 и загружаем веса, обученные
# на наборе данных ImageNet
model = VGG19(weights='vgg19.h5')

# Загружаем изображение для распознавания, преобразовываем его в массив
# numpy и выполняем предварительную обработку
img_path = '4.jpg' # здесь указываем путь к изображению
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

cats = {'tabby', 'tabby_cat', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Siamese', 'Egyptian_cat', 'cougar', 'puma',
        'catamount',
        'mountain_lion', 'painter', 'panther', 'Felis concolor', 'lynx', 'catamount', 'leopard', 'Panthera pardus',
        'snow leopard', 'ounce',
        'Panthera_uncia', 'jaguar', 'panther', 'Panthera_onca', 'Felis_onca', 'lion', 'king_of_beasts', 'Panthera_leo',
        'tiger', 'Panthera_tigris',
        'cheetah', 'chetah', 'Acinonyx_jubatus'}

dogs = {'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Maltese_terrier', 'Maltese', 'Pekinese', 'Pekingese', 'Peke',
        'Shih-Tzu',
        'Blenheim_spaniel', 'papillon', 'toy terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'Afghan', 'basset',
        'basset_hound', 'beagle',
        'bloodhound', 'sleuthhound', 'bluetick', 'black-and-tan coonhound', 'Walker_hound', 'Walker_foxhound',
        'English_foxhound',
        'redbone', 'borzoi', 'Russian_wolfhound', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound',
        'Ibizan_Podenco',
        'Norwegian_elkhound', 'elkhound', 'otterhound', 'otter_hound', 'Saluki', 'gazelle_hound', 'Scottish_deerhound',
        'deerhound',
        'Weimaraner', 'Staffordshire_bullterrier', 'Staffordshire_bull_terrier', 'American_Staffordshire_terrier',
        'Staffordshire_terrier',
        'American_pit_bull_terrier', 'pit_bull_terrier', 'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier',
        'Irish_terrier',
        'Norfolk_terrier', 'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier',
        'Sealyham_terrier',
        'Sealyham', 'Airedale', 'Airedale_terrier', 'cairn', 'cairn_terrier', 'Australian_terrier', 'Dandie_Dinmont',
        'Dandie_Dinmont_terrier', 'Boston_bull', 'Boston_terrier', 'miniature_schnauzer', 'giant_schnauzer',
        'standard_schnauzer'
        'Scotch_terrier', 'Scottish_terrier', 'Scottie', 'Tibetan_terrier', 'chrysanthemum dog', 'silky_terrier',
        'Sydney_silky',
        'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa', 'Lhasa apso', 'flat-coated retriever',
        'curly-coated retriever',
        'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay retriever', 'German_short-haired pointer', 'vizsla',
        'Hungarian_pointer',
        'English_setter', 'Irish_setter', 'red_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber',
        'clumber_spaniel', 'English_springer', 'English_springer_spaniel'
                                               'Welsh_springer_spaniel', 'cocker_spaniel', 'English_cocker_spaniel',
        'cocker', 'Sussex_spaniel', 'Irish_water_spaniel',
        'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog',
        'bobtail', 'Shetland_sheepdog', 'Shetland_sheep dog', 'Shetland'
                                                              'collie', 'Border_collie', 'Bouvier_des_Flandres',
        'Bouviers_des_Flandres', 'Rottweiler', 'German_shepherd', 'German_shepherd_dog', 'German_police_dog', 'alsatian'
                                                                                                              'Doberman',
        'Doberman_pinscher', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain dog', 'Appenzeller',
        'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard',
        'St Bernard', 'Eskimo_dog', 'husky',
        'malamute', 'malemute', 'Alaskan_malamute', 'Siberian_husky', 'dalmatian', 'coach_dog', 'carriage_dog',
        'affenpinscher', 'monkey_pinscher', 'monkey_dog',
        'basenji', 'pug', 'pug-dog', 'Leonberg', 'Newfoundland', 'Newfoundland_dog', 'Great Pyrenees', 'Samoyed',
        'Samoyede', 'Pomeranian',
        'chow', 'chow_chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Pembroke_Welsh_corgi', 'Cardigan',
        'Cardigan_Welsh_corgi',
        'toy_poodle', 'miniature_poodle', 'standard_poodle'}


def pol_zn(o, i):
    # Запускаем распознавание объекта на изображении
    preds = model.predict(o)
    m = decode_predictions(preds, top=3)[0]
    # Печатаем три класса объекта с самой высокой вероятностью
    # print('Результаты распознавания:', decode_predictions(preds, top=3)[0])
    m = tuple(m)
    m = list(m[i])
    m = m[1]
    return m


q = model.predict(x)

for i in [0, 1, 2]:
    if pol_zn(x, i) in cats:
        print('0')
    elif pol_zn(x, i) in dogs:
        print('1')
    elif pol_zn(x, i) not in cats and pol_zn(x, i) not in dogs:
        print('2')

