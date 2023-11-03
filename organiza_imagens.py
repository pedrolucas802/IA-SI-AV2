# import cv2
from PIL import Image
import numpy as np
import os


# folderRoot = 'faces\\faces\\'
def organize_images():
    folderRoot = 'faces/'
    individual = ['an2i', 'at33', 'boland', 'bpm', 'ch4f', 'cheyer', 'choon', 'danieln', 'glickman', 'karyadi',
                  'kawamura', 'kk49', 'megak', 'mitchell', 'night', 'phoebe', 'saavik', 'steffi', 'sz24', 'tammo']

    expressions = ['_left_angry_open', '_left_angry_sunglasses', '_left_happy_open', '_left_happy_sunglasses',
                   '_left_neutral_open', '_left_neutral_sunglasses', '_left_sad_open', '_left_sad_sunglasses',
                   '_right_angry_open', '_right_angry_sunglasses', '_right_happy_open', '_right_happy_sunglasses',
                   '_right_neutral_open', '_right_neutral_sunglasses', '_right_sad_open', '_right_sad_sunglasses',
                   '_straight_angry_open', '_straight_angry_sunglasses', '_straight_happy_open',
                   '_straight_happy_sunglasses',
                   '_straight_neutral_open', '_straight_neutral_sunglasses', '_straight_sad_open',
                   '_straight_sad_sunglasses',
                   '_up_angry_open', '_up_angry_sunglasses', '_up_happy_open', '_up_happy_sunglasses',
                   '_up_neutral_open', '_up_neutral_sunglasses', '_up_sad_open', '_up_sad_sunglasses']

    Red = 60
    X = np.empty((Red * Red, 0))
    Y = np.empty((len(individual), 0))

    for i in range(len(individual)):
        for j in range(len(expressions)):
            path = os.path.join(folderRoot, individual[i], individual[i] + expressions[j] + '.pgm')

            if os.path.exists(path):
                with Image.open(path) as PgmImg:
                    PgmImg = PgmImg.convert("L")
                    ResizedImg = PgmImg.resize((Red, Red))

                VectorNormalized = np.array(ResizedImg).flatten('F')
                ROT = -np.ones((len(individual), 1))
                ROT[i, 0] = 1

                # ResizedImg.show()

                VectorNormalized.shape = (len(VectorNormalized), 1)
                X = np.append(X, VectorNormalized, axis=1)
                Y = np.append(Y, ROT, axis=1)
            else:
                print(f"File not found: {path}")

    print(f'Quantidade de amostras do conjunto de dados: {X.shape[1]}')
    print('A quantidade de preditores está relacionada ao redimensionamento!')
    print(f'Para esta rodada escolheu-se um redimensionamento de {Red}')
    print(f'Portanto, a quantidade de preditores desse conjunto de dados: {X.shape[0]}')
    print(f'Este conjunto de dados possui {Y.shape[0]} classes')
    print('****************************************************************')
    print('****************************************************************')
    print('***********************RESUMO***********************************')
    print('****************************************************************')
    print('****************************************************************')
    print(f'X tem ordem {X.shape[0]}x{X.shape[1]}')
    print(f'Y tem ordem {Y.shape[0]}x{Y.shape[1]}')

    # print(X)
    # print("------")
    # print(Y)

    return X, Y


organize_images()