# -*- coding: utf-8 -*-

from skimage import io, color, img_as_ubyte
from sklearn.metrics.cluster import entropy
import os, sys
import numpy as np
from PIL import Image
import statistics
import pandas as pd
import datetime as dt
import image_slicer

# Argumento que passa o nome da imagem
#argv_img = sys.argv[1]
argv_img = 'test.jpg'

# Variaveis de features
dir_64_imgs = './64_slices'
bw, bh = 8, 8 # tamanho bloco (altura e largura)
stego = "yes"

def identificar_os():
    # Variaveis
    windows_unidade_padrao = 'c'
    diretorio_base = ''

    # Checa se é windows (nt) ou linux/osx
    if os.name == 'nt':
        diretorio_base = windows_unidade_padrao + ':'
    else:
        diretorio_base = os.getenv("HOME")
    return diretorio_base

def cria_diretorio():
    if os.path.isdir(dir_64_imgs):
        pass
    else:
       os.mkdir(dir_64_imgs)
    return dir_64_imgs

def img_full_to_gray(path_img):
    # Por duvida sobre qual metodo usar para abrir a imagem, deixarei os dois aqui
    # ==========================================================================
    # image_gray = imread(path_img, as_gray=True)
    # .shape confere as dimensoes das matrizes
    # Abre a img com o seguinte padrao:
    # [[0.41142667  0.41142667  0.41142667 ... 0.41142667 0.41142667 0.41142667]
    # [0.41142667  0.41142667  0.41142667 ... 0.41142667 0.41142667 0.41142667]
    # [0.41142667  0.41142667  0.41142667 ... 0.41142667 0.41142667 0.41142667]]
    # =========================================================================
    # Utilizando o rgb2gray, o padrao da matrix é diferente:
    # [[234 234 234 ... 234 234 234]
    #  [234 234 234 ... 234 234 234]
    #  [234 234 234 ... 234 234 234]]
    # =========================================================================
    rgbImg = io.imread(path_img)
    r_gray_img = img_as_ubyte(color.rgb2gray(rgbImg))
    return r_gray_img

def slice_img(path_img):
    # Variaveis
    qtd_slices = 64 #8x8

    #Criar o diretorio caso nao exista
    dir = cria_diretorio()

    # Dividir a imagem em 64 partes
    tiles = image_slicer.slice(path_img, qtd_slices, save=False)
    image_slicer.save_tiles(tiles, directory=dir,format='JPEG')

def open_64_imgs():
    from os import listdir
    from os.path import isfile, join

    # Gerar uma lista com todas as imagens do dir 64_slices
    all_imgs = [f for f in listdir(dir_64_imgs) if isfile(join(dir_64_imgs, f))]
    all_imgs2 = all_imgs
    # Tive q criar uma copia pq a funcao extrai_36_slices estava sobreescrevendo all_imgs para 36

    # Ordem Alfabetica
    all_imgs = sorted(all_imgs)
    all_imgs2 = sorted(all_imgs2)

    # Remover as "bordas" da Imagem para a ultima feature
    list_36_slices = extrai_36_slices(all_imgs)


    #transformar as imagens em grayscale
    r_list_36img_gray = img_to_gray(list_36_slices)
    r_list_64img_gray = img_to_gray(all_imgs2)

    return r_list_36img_gray, r_list_64img_gray

def img_to_gray(path_img):
    from skimage.io import imread
    # Variaveis
    list_gray_img = []

    for i in range(len(path_img)):
        gray_img = imread(dir_64_imgs + os.sep + path_img[i], as_gray=True)
        # rgbImg = io.imread(dir_64_imgs + os.sep + path_img[i])
        # gray_img = img_as_ubyte(color.rgb2gray(rgbImg))
        list_gray_img.append(gray_img)

    return list_gray_img

def extrai_36_slices(list_all_64):
    # A imagem composta de 8x8, retiraremos as seguintes posicoes.
    #
    #    0	1	2	3	4	5	6	7
    #    8	*	*	*	*	*	*	15
    #   16	*	*	*	*	*	*	23
    #   24	*	*	*	*	*	*	31
    #   32	*	*	*	*	*	*	39
    #   40	*	*	*	*	*	*	47
    #   48	*	*	*	*	*	*	55
    #   56	57	58	59	60	61	62	63

    # Variaveis
    # Possui as posições em index que devem ser removidas
    imgs_index = [0,1,2,3,4,5,6,7,8,15,16,23,24,31,32,39,40,47,48,55,56,57,58,59,60,61,62,63]
    for i in reversed(imgs_index):
        del list_all_64[i]

    return list_all_64

def deleta_dir_64_slices(dir):
    import shutil
    # Remove diretorios com arquivos
    shutil.rmtree(dir)

def extract_entropy_full(path_img):
    ### FEATURE 1 ###
    # Extraindo a entropia da imagem inteira
    # Image.open(argv_img).convert('LA') converte em grayscale
    rgbImg = io.imread(path_img)
    grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
    return_entropia = entropy(grayImg)
    return return_entropia

def entropia_por_bloco(list_blocos_img):
    # String de Retorno
    return_lista_entropia_img = []

    for i in range(len(list_blocos_img)):
            return_lista_entropia_img.append(entropy(list_blocos_img[i]))
    return return_lista_entropia_img

def media_entropia_blocos(entropia_blocos):
    ### FEATURE 2 ###
    return_media_entropia = statistics.mean(entropia_blocos)
    return return_media_entropia

def stdev_entropia_blocos(entropia_blocos):
    ### FEATURE 3 ###
    return_stdev_entropia = statistics.stdev(entropia_blocos)
    return return_stdev_entropia

def regex(list_64):
    import re

    # Variaveis
    r_lista_regex = []

    # Remover todos os caracteres que nao forem 0 e 1
    for i in range(len(list_64)):
        # Regex sub substitui em "" oq nao for 0 ou 1 em casas unicas
        regex = re.sub('0*([1-9]\d+|[2-9])','',str(list_64[i]))
        # Regex findall extrai apenas os numeros
        regex2 = re.findall('\d+', str(regex))
        r_lista_regex.append(regex2)

    return r_lista_regex

def markov_chains(lista_img):
    # Variaveis
    chain_list = []
    matrix_2x2_minima = ['0','0','1','1']
    return_markov_chain = []

    # Transformar em lista unidimesional
    for i in range(len(lista_img)):
        chain_list.append(np.concatenate(lista_img[i]).ravel())

    # Regex
    lista_regex = regex(chain_list)

    # Adicionar 2 vezes 0 e 1 para formar a tabela 2x2
    for i in range(len(lista_regex)):
        for j in range(len(matrix_2x2_minima)):
            lista_regex[i].append(matrix_2x2_minima[j])

        # Aplicar a markov chains para cada posição
        mk = lista_regex[i]
        # A probabilidade (forward) do Estado Atual (0) para se Tornar um proximo
        #   Estado (1) pode ser encontrado na coluna '1' e linha '0' (0.5).
        #
        # Caso queira as probabilidades anteriores (backward), apenas sete o
        #       normalize = 1.
        #
        # === Exemplo do DATAFRAME para referencia da explicao acima ===
        #        Proximo    0    1
        #        Atual
        #        0        0.7  0.5
        #        1        0.0  1.0
        x = pd.crosstab(pd.Series(mk[:-1], name='Atual'),pd.Series(mk[1:],name='Proximo'),normalize=0)
        return_markov_chain.append(x)

    # Retornar markov_chains ou em dataframe (em array pode-se utilizar .values ao final)
    return return_markov_chain

def isolar_variaveis_matrix_markov_chains(lista_dataframe_mk):
    # Variaveis para append
    return_mk_00 = []
    return_mk_01 = []
    return_mk_10 = []
    return_mk_11 = []

    for i in range(len(lista_dataframe_mk)):
        x = 0
        # A cada linha eu faço o append para a lista de variaveis.
        for index, row in lista_dataframe_mk[i].iterrows():
            # Como é uma matrix fixa de 2x2, preferi usar um contador simples
            #       para a escolha de qual lista será usada para o append.
            if x == 0:
                    return_mk_00.append(row['0'])
                    return_mk_01.append(row['1'])
            else:
                    return_mk_10.append(row['0'])
                    return_mk_11.append(row['1'])
            x = x+1

    return return_mk_00, return_mk_01, return_mk_10, return_mk_11

def media_matrix_markov(input_mk_00,input_mk_01,input_mk_10,input_mk_11):
    ### FEATURE 4-7 ###
    return_media_00 = statistics.mean(input_mk_00)
    return_media_01 = statistics.mean(input_mk_01)
    return_media_10 = statistics.mean(input_mk_10)
    return_media_11 = statistics.mean(input_mk_11)

    return return_media_00, return_media_01, return_media_10, return_media_11

def stdev_matrix_markov(input_mk_00,input_mk_01,input_mk_10,input_mk_11):
    ### FEATURE 8-11 ###
    return_stdev_00 = statistics.stdev(input_mk_00)
    return_stdev_01 = statistics.stdev(input_mk_01)
    return_stdev_10 = statistics.stdev(input_mk_10)
    return_stdev_11 = statistics.stdev(input_mk_11)

    return return_stdev_00, return_stdev_01, return_stdev_10, return_stdev_11

def media_full_img(path_img):
    # Variaveis
    markov_chain_full = []
    matrix_2x2_minima = ['0','0','1','1']

    # Transformar a img em gray e aplicar o regex
    gray_img = img_full_to_gray(path_img)
    lista_regex = regex(gray_img)

    # Adicionar 2 vezes 0 e 1 para formar a tabela 2x2
    for i in range(len(lista_regex)):
        for j in range(len(matrix_2x2_minima)):
            lista_regex[i].append(matrix_2x2_minima[j])

        # Aplicar a markov chains para cada posição
        mk = lista_regex[i]
        # A probabilidade (forward) do Estado Atual (0) para se Tornar um proximo
        #   Estado (1) pode ser encontrado na coluna '1' e linha '0' (0.5).
        #
        # Caso queira as probabilidades anteriores (backward), apenas sete o
        #       normalize = 1.
        #
        # === Exemplo do DATAFRAME para referencia da explicao acima ===
        #        Proximo    0    1
        #        Atual
        #        0        0.7  0.5
        #        1        0.0  1.0
        x = pd.crosstab(pd.Series(mk[:-1], name='Atual'),pd.Series(mk[1:],name='Proximo'),normalize=0)
        markov_chain_full.append(x)

    # Reutilizar as funcoes já definidas
    list_00, list_01, list_10, list_11 = isolar_variaveis_matrix_markov_chains(markov_chain_full)
    r_full_00, r_full_01, r_full_10, r_full_11 = media_matrix_markov(list_00, list_01, list_10, list_11)

    return r_full_00, r_full_01, r_full_10, r_full_11

def conditional_entropy_in_dct(list_36_gray_img, list_64_gray_img):
    from scipy.fftpack import fft, dct
    from pyitlib import discrete_random_variable as drv

    # Variaveis
    list_36_dct = []
    list_36_conditional_entropy = []
    chain_list_36_dct = []
    chain_list_36_origin = []

    # Converter cada bloco em DCT-II
    for i in range(len(list_36_gray_img)):
            list_36_dct.append(dct(list_36_gray_img[i]))

    for i in range(len(list_36_dct)):
        chain_list_36_dct.append(np.concatenate(list_36_dct[i]).ravel())

    for i in range(len(list_36_gray_img)):
        chain_list_36_origin.append(np.concatenate(list_36_gray_img[i]).ravel())

    # Aplicar a Conditional Entropy para cada bloco/slice
    for i in range(len(chain_list_36_dct)):
        list_36_conditional_entropy.append(drv.entropy_conditional(chain_list_36_dct[i],chain_list_36_origin[i], base=np.exp(2)))

    return list_36_conditional_entropy

def criar_csv(path):
     # Inicia com o caminho absoluto
    caminho_absoluto = path + os.sep + 'csv' + os.sep

    # Confere se já existe a pasta para receber o CSV
    if not os.path.exists(caminho_absoluto):
        os.makedirs(caminho_absoluto)
        print("Criado Diretório:", caminho_absoluto)

    # Cria o arquivo com o padrão db_stego.csv ou db_origin.csv
    caminho_ab_csv = caminho_absoluto + 'db_stego.csv'
    print("Caminho do Arquivo CSV: ", caminho_ab_csv)

    arquivo = open(caminho_ab_csv, 'a+', encoding="utf-8")
    arquivo.writelines("""Stego?;Entropy_img_full;Mean_entropy_slices;Stdev_entropy_slices;Mean_MarkovChains_00;Mean_MarkovChains_01;Mean_MarkovChains_10;Mean_MarkovChains_11;Stdev_MarkovChains_00;Stdev_MarkovChains_01;Stdev_MarkovChains_10;Stdev_MarkovChains_11;Mean_imgfull_MarkovChains_00;Mean_imgfull_MarkovChains_01;Mean_imgfull_MarkovChains_10;Mean_imgfull_MarkovChains_11;Conditional_Entropy_Slice_01;Conditional_Entropy_Slice_02;Conditional_Entropy_Slice_03;Conditional_Entropy_Slice_04;Conditional_Entropy_Slice_05;Conditional_Entropy_Slice_06;Conditional_Entropy_Slice_07;Conditional_Entropy_Slice_08;Conditional_Entropy_Slice_09;Conditional_Entropy_Slice_10;Conditional_Entropy_Slice_11;Conditional_Entropy_Slice_12;Conditional_Entropy_Slice_13;Conditional_Entropy_Slice_14;Conditional_Entropy_Slice_15;Conditional_Entropy_Slice_16;Conditional_Entropy_Slice_17;Conditional_Entropy_Slice_18;Conditional_Entropy_Slice_19;Conditional_Entropy_Slice_20;Conditional_Entropy_Slice_21;Conditional_Entropy_Slice_22;Conditional_Entropy_Slice_23;Conditional_Entropy_Slice_24;Conditional_Entropy_Slice_25;Conditional_Entropy_Slice_26;Conditional_Entropy_Slice_27;Conditional_Entropy_Slice_28;Conditional_Entropy_Slice_29;Conditional_Entropy_Slice_30;Conditional_Entropy_Slice_31;Conditional_Entropy_Slice_32;Conditional_Entropy_Slice_33;Conditional_Entropy_Slice_34;Conditional_Entropy_Slice_35;Conditional_Entropy_Slice_36;""")
    arquivo.write("\n")
    arquivo.close()
    return caminho_ab_csv

def feature_to_list(var_00, var_01, var_10, var_11):
    # Variaveis
    lista_append = []

    # Apend as variaveis
    lista_append.append(var_00)
    lista_append.append(var_01)
    lista_append.append(var_10)
    lista_append.append(var_11)

    return lista_append

def salvar_csv(input_csv,id_stego,ent_full,mean_ent,stdev_ent,l_mean_mk,l_stdev_mk,l_mean_full,l_cond_ent):
    # As variaveis com l_ no inicio são listas.

    arquivo = open(input_csv, 'a+', encoding="utf-8")

    # Escrevendo as features simples
    arquivo.write(str(id_stego))
    arquivo.write(";")
    arquivo.write(str(ent_full))
    arquivo.write(";")
    arquivo.write(str(mean_ent))
    arquivo.write(";")
    arquivo.write(str(stdev_ent))
    arquivo.write(";")

    # Escrevendo as features em lista com loop
    for i in l_mean_mk:
        arquivo.write(str(i))
        arquivo.write(";")

    for j in l_stdev_mk:
        arquivo.write(str(j))
        arquivo.write(";")

    for k in l_mean_full:
        arquivo.write(str(k))
        arquivo.write(";")

    for x in l_cond_ent:
        arquivo.write(str(x))
        arquivo.write(";")

    arquivo.write("\n")
    arquivo.close()
    print('\nSalvo')

def main():
    # === PRE FEATURES ===
    # ====================

    # Calcular tempo inicial do script
    tempo_inicial = dt.datetime.now().strftime("%H:%M:%S")

    # Dividir a img em 64 partes
    slice_img(argv_img)

    # Abrir cada img do dir 64_imgs
    list_gray_36, list_gray_64 = open_64_imgs()

    # Extrair a entropia para cada bloco(slices).
    lista_entropia_blocos = entropia_por_bloco(list_gray_64)

    # Transformar cada bloco em Lista de Markov Chains
    matrix_markov_chains = markov_chains(list_gray_64)

    # # Isolar variaveis do Dataframe para calculo de Média e Desvio Padrao
    mk_00, mk_01, mk_10, mk_11 = isolar_variaveis_matrix_markov_chains(matrix_markov_chains)

    # === FEATURES ===
    # ================

    # FEATURE 1 - Entropia da imagem inteira
    entropia_img_full = extract_entropy_full(argv_img)

    # FEATURE 2 - Media da entropia de cada bloco da imagem
    media_entropia = media_entropia_blocos(lista_entropia_blocos)

    # FEATURE 3 - Desvio Padrao da entropia de cada bloco da imagem
    stdev_entropia = stdev_entropia_blocos(lista_entropia_blocos)

    # FEATURE 4-7 - Média de markov chains por cada bloco (0/0, 0/1, 1/0, 1/1).
    media_mk_00, media_mk_01, media_mk_10, media_mk_11 = media_matrix_markov(mk_00, mk_01, mk_10, mk_11)

    # FEATURE 8-11 - Desvio Padrao de markov chains por cada bloco (0/0, 0/1, 1/0, 1/1).
    stdev_mk_00, stdev_mk_01, stdev_mk_10, stdev_mk_11 = stdev_matrix_markov(mk_00, mk_01, mk_10, mk_11)

    # FEATURE 12-15 - "The average probability" por toda a imagem para cada uma das
    #                   transições (0/0, 0/1, 1/0, 1/1).
    media_full_00, media_full_01, media_full_10, media_full_11 = media_full_img(argv_img)

    # FEATURE 16-51 - The conditional entropy for each non-boundary
    #                 position in the 8 x 8 DCT coefficient grid, calculated for
    #                 the entire image.
    list_36_features_conditional_entropy = conditional_entropy_in_dct(list_gray_36, list_gray_64)

    # === FINALIZACAO ===
    # ===================
    # Deleta o dir 64_slices
    deleta_dir_64_slices(dir_64_imgs)

    # Criar o CSV
    dir_base = identificar_os()
    caminho_csv = criar_csv(dir_base)

    # Transformar as features em listas para facil insercao
    list_media_mk = feature_to_list(media_mk_00, media_mk_01, media_mk_10, media_mk_11)
    list_stdev_mk = feature_to_list(stdev_mk_00, stdev_mk_01, stdev_mk_10, stdev_mk_11)
    list_media_mk_full = feature_to_list(media_full_00, media_full_01, media_full_10, media_full_11)

    # Salva em um CSV os dados recolhidos
    salvar_csv(caminho_csv,stego,entropia_img_full,media_entropia,stdev_entropia,list_media_mk,list_stdev_mk,list_media_mk_full,list_36_features_conditional_entropy)

    # Encerra o calculo do tempo do script
    print("--- Tempo Inicial de Execução:   "+ tempo_inicial +" ---")
    tempo_final = dt.datetime.now().strftime("%H:%M:%S")
    print("--- Tempo Final de Execução:     "+ tempo_final  + " ---")


if __name__ == "__main__":
        main()
