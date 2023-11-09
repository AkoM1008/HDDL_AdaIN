import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



mse_criterion = nn.MSELoss()
def calc_mean_std(feat, eps=1e-5): #feat = feature maps = sortie d'une couche de CNN
	size = feat.size()
	assert(len(size)==4) #assert checks if condition true, if not throws an error 
	N, C = size[:2] #on recup les 2 premieres dimensions de feat: batch size et nb of channels; les deux derniers sont W et H, width et height de feat
	feat_var = feat.view(N, C, -1).var(dim=2) +eps #.view combine les deux dernieres dimensions W et H en 1, i.e. flatten feat, var calcule la variance de cette nouvelle dim (donc variance de feat pour chaque sample et channel), +eps pour eviter de /0 pdt la normalization 
	feat_std = feat_var.sqrt().view(N, C, 1, 1) #on calcule la standard deviation et reshape en 4 dimensions originales de feat (car on a calcule var et std pour chaque sample et channel donc W=H=1)
	feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1) #idem pour la moyenne
	return feat_mean, feat_std

#L_c se calcule comme dist entre f(g(t)) et t donc on essaie de minimiser l'erreur commise par encoder et decoder (?); parce que si f et g parfait, alors L_c=0
def calc_content_loss(input_im, target): #input_im et target sont f(g(t)) et t du papier; f=encoder, g=decoder, t=sortie de la couche AdaIN
	assert (input_im.size() == target.size()) 
	assert (target.requires_grad is False) #on verifie que target n'implemente pas autograd (tracking des operations pour calculer le gradient)/pas trop compris prq 
	return mse_criterion(input_im, target) #L_c = content loss (6.2. du papier)

#ds le papier on fait la somme sur toutes les couches de VGG-19, cette fct fait calcule pour une couche a la fois (?)
def calc_style_loss(input_im, target):
	assert (input_im.size() == target.size())
	assert (target.requires_grad is False)
	input_mean, input_std = calc_mean_std(input_im)
	target_mean, target_std = calc_mean_std(target)
	return mse_criterion(input_mean, target_mean) + \
			mse_criterion(input_std, target_std) #L_s = style loss


vggnet = nn.Sequential(
			# encode 1-1
			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)), #1er 3 = nb de channels, 2e 3 = nb de filtres de cette couche
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'), #idem mais avec padding reflect = au lieu de remplir les bords avec 0, on les remplit avec des valeurs proches des bords pour eviter le edge effect
			nn.ReLU(inplace=True), # relu 1-1, ReLu = max (0,x)
			# encode 2-1
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), #inplace = modif l'input directement sans allouer la memoire supp 
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #no padding, no dilation (=ajouter des espaces entre les elements des filtres pour couvrir plus, pas trop trop compris), ceil_mode=False donc on arrondit par truncation 

			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 2-1
			# encoder 3-1
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 3-1
			# encoder 4-1
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 4-1
			# rest of vgg not used (on utilise pas ces couches ?? prq les coder ??)
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True)
			)			


class AdaIN(nn.Module): #module = couche ou pls couches d'un reseau 
	def __init__(self):
		super().__init__()

	def forward(self, x, y): #forward pass de la couche, x et y les entrees de la couche = eux-memes des tensors (ici sorties des 4 couches vgg), x et y issus de content et style images 
		eps = 1e-5	
		mean_x = torch.mean(x, dim=[2,3]) #calcule la moyenne de tensor x le long des dim 2 et 3 (dim 2 et 3 = W et H donc on calcule la moy le long des valeurs spatiales de x), mean_x est tensor de dim 2 (N et C)
		mean_y = torch.mean(y, dim=[2,3]) #cela correspond bien a IN car on va normaliser par chaque sample de batch

		std_x = torch.std(x, dim=[2,3]) 
		std_y = torch.std(y, dim=[2,3])

		mean_x = mean_x.unsqueeze(-1).unsqueeze(-1) #on ajoute 2 dims supp a la fin pour obtenir un tensor de dim 4 (devient (N,C)-> (N,C,1,1)), necessaire car x de dim 4
		mean_y = mean_y.unsqueeze(-1).unsqueeze(-1) #unsqueeze similaire au view, les deux sont utilises pour reshape les entrees

		std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps #on ajoute eps pour eviter de /0 
		std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

		out = (x - mean_x)/ std_x * std_y + mean_y #on normalize d'abord x par sa moy et std et ensuite on fait l'inverse avec std et moy de y pour obtenir out qui va avoir la moy et std de y


		return out #on obtient le tensor out qui a le contenu de x mais les features de y 




class StyleTransferNet(nn.Module):
	def __init__(self, vgg_model):
		super().__init__()

		vggnet.load_state_dict(vgg_model) #on load les poids deja obtenus ds vgg_model ds vggnet

		self.encoder = nn.Sequential( #on code encoder, 1er eleement de notre reseau, cf section 5
			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)),
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True)
			)

		self.encoder.load_state_dict(vggnet[:21].state_dict()) #on recup les poids des premieres 21 couches de vggnet (tt jusqu'a relu 4_1 exclu)
		for parameter in self.encoder.parameters():
			parameter.requires_grad = False
			
		self.decoder = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'), #upsample augmente les dim de tensor par scale_factor, nearest mode = les nouvelles cases remplies par la valeur de pixel le + proche 
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'), #upsample utilise au lieu de maxpool pour eviter checkerboard effects
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),

			)
		self.adaIN = AdaIN()
		self.mse_criterion = nn.MSELoss()




	def forward(self, x, alpha=1.0):

		content_img = x[0]
		style_img = x[1]

		encode_content = self.encoder(content_img) #f(c)
		encode_style = self.encoder(style_img) #f(s)

		encode_out = self.adaIN(encode_content, encode_style) #AdaIN(f(c),f(s))=t
		

		if self.training:
			gen_img = self.decoder(encode_out) #g(t)
			encode_gen = self.encoder(gen_img) #f(g(t)) -> on va l'utiliser pour la loss_c

			#on divise encoder par chaque couche pour appliquer la somme de L_s sur les sorties des couches de vgg, les fm sont des phi de la formule, prq cette division ? L=4 (3 ici et encode_gen/style etant deja calcules)
			fm11_style = self.encoder[:3](style_img)
			fm11_gen = self.encoder[:3](gen_img)

			fm21_style = self.encoder[3:8](fm11_style)
			fm21_gen = self.encoder[3:8](fm11_gen)

			fm31_style = self.encoder[8:13](fm21_style)
			fm31_gen = self.encoder[8:13](fm21_gen)
			
			loss_content = self.mse_criterion(encode_gen, encode_out) #formule de 6.2

			loss_style = self.mse_criterion(torch.mean(fm11_gen, dim=[2,3]), torch.mean(fm11_style, dim=[2,3])) +	\
						self.mse_criterion(torch.mean(fm21_gen, dim=[2,3]), torch.mean(fm21_style, dim=[2,3])) +	\
						self.mse_criterion(torch.mean(fm31_gen, dim=[2,3]), torch.mean(fm31_style, dim=[2,3])) +	\
						self.mse_criterion(torch.mean(encode_gen, dim=[2,3]), torch.mean(encode_style, dim=[2,3])) +	\
						self.mse_criterion(torch.std(fm11_gen, dim=[2,3]), torch.std(fm11_style, dim=[2,3])) +	\
						self.mse_criterion(torch.std(fm21_gen, dim=[2,3]), torch.std(fm21_style, dim=[2,3])) +	\
						self.mse_criterion(torch.std(fm31_gen, dim=[2,3]), torch.std(fm31_style, dim=[2,3])) +	\
						self.mse_criterion(torch.std(encode_gen, dim=[2,3]), torch.std(encode_style, dim=[2,3])) 

			return loss_content, loss_style
		encode_out = alpha * encode_out + (1-alpha)* encode_content 
		gen_img = self.decoder(encode_out) #=g(alpha*t+(1-alpha)*f(c))
		return gen_img























