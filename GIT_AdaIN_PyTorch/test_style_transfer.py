import argparse
import os
import numpy as np
from AdaIN import StyleTransferNet
from PIL import Image
import torch
import torchvision.transforms as transforms 
from torchvision.utils import save_image


## Idees d'ajout au code

# - Tester d'autres valeurs d'alpha, qui impacteront le niveau de transfert de style
# - Ajouter une option pour pouvoir donner en entree des images en noir est blanc (normalement les convertir en 3 channels devrait suffire)
class AlphaRange(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
	def __eq__(self, other):
		return self.start <= other <= self.end
	def __str__(self):
		return 'Alpha Range'

# Definition des arguments requis lors de l'execution du code
parser = argparse.ArgumentParser()
parser.add_argument('--input_image', type=str, help='test image')# Path de l'image d'entree (de contenu)
parser.add_argument('--style_image', type=str, help='style image')# Path de l'image de style
parser.add_argument('--weight', type=str, help='decoder weight file')# Fichier contenant les données des poids
parser.add_argument('--alpha', type=float, default=1.0, choices=[AlphaRange(0.0, 1.0)], help='Level of style transfer, value between 0 and 1')
# Niveau de transfert de style
parser.add_argument('--cuda', action='store_true', help='Using GPU to train')# Uses GPU to train


if __name__ == '__main__':
    # Recupere et stocke les arguments donnés
	opt =parser.parse_args()
	input_image = Image.open(opt.input_image)
	style_image = Image.open(opt.style_image)
	output_format = opt.input_image[opt.input_image.find('.'):]# Le format de l'image de sortie et le même que celui de l'image de contenu
    # Cree le repertoire de sortie
	out_dir = './results/'
	os.makedirs(out_dir, exist_ok=True)
	with torch.no_grad():
		vgg_model = torch.load('vgg_normalized.pth')# Load le modele

		net = StyleTransferNet(vgg_model)
		net.decoder.load_state_dict(torch.load(opt.weight))# Load les poids

		net.eval()
        # Reshape les deux images d'entree à la même taille
		input_image = transforms.Resize(512)(input_image)
		style_image = transforms.Resize(512)(style_image)
		# Transform les images en tenseurs
		input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)
		style_tensor = transforms.ToTensor()(style_image).unsqueeze(0)
	

		if torch.cuda.is_available() and opt.cuda:
			net.cuda()# ?
			input_tensor = input_tensor.cuda()# ?
			style_tensor = style_tensor.cuda()# ?
		out_tensor = net([input_tensor, style_tensor], alpha = opt.alpha)# Applique le modele aux images choisies


		save_image(out_tensor, out_dir + opt.input_image[opt.input_image.rfind('/')+1: opt.input_image.find('.')]
								+"_style_"+ opt.style_image[opt.style_image.rfind('/')+1: opt.style_image.find('.')]
								+ output_format)# Sauvegarde de l'image de sortie