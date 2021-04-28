import torch
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  
import torchvision.models as models
from torchvision.utils import save_image
import torchvision.transforms as transforms

print("Letter        :", sys.argv[1])
print("Style         :", sys.argv[2])
print("Style Letter  :", sys.argv[3])
print("Epochs        :", sys.argv[4])
print("Image Size    :", sys.argv[5])
print("Learning Rate :", sys.argv[6])
print("Alpha         :", sys.argv[7])
print("Beta          :", sys.argv[8])


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # The first number x in convx_y gets added by 1 after it has gone
        # through a maxpool, and the second y if we have several conv layers
        # in between a max pool. These strings (0, 5, 10, ..) then correspond
        # to conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 mentioned in NST paper
        self.chosen_features = ["0", "5", "10", "19", "28"]

        # We don't need to run anything further than conv5_1 (the 28th module in vgg)
        # Since remember, we dont actually care about the output of VGG: the only thing
        # that is modified is the generated image (i.e, the input).
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        # Store relevant features
        features = []

        # Go through each layer in model, if the layer is in the chosen_features,
        # store it in features. At the end we'll just return all the activations
        # for the specific layers we have in chosen_features
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


def load_image(image_name):
    im = Image.open(image_name)
    color=(255, 255, 255)
    im.load()  # needed for split()
    background = Image.new('RGB', im.size, color)
    background.paste(im, mask=im.split()[3])
    image = loader(background).unsqueeze(0)
    return image.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = int(sys.argv[4])

# Here we may want to use the Normalization constants used in the original
# VGG network (to get similar values net was originally trained on), but
# I found it didn't matter too much so I didn't end of using it. If you
# use it make sure to normalize back so the images don't look weird.
loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ]
)
target_name = sys.argv[1]
style_imgs = []
style_letter = sys.argv[3]
style_name = sys.argv[2]

result_file_input = "Generated/"+style_name+"/"+target_name[:-4]+"_I.png"

result_file = "Generated/"+style_name+"/"+target_name[:-4]+"_O.png"
target_name = "Kannada_Fonts/Belur/" + target_name
original_img = load_image(target_name)
save_image(original_img, result_file_input)
print(style_name, "-", target_name)
# style_img_names = os.listdir('English_Fonts/'+ style_name)
# style_img_names.sort()
# style_img_names = style_img_names[36:]
# style_img_names = ["008.png", "020.png", "028.png", "041.png", "042.png", "059.png"]
style_img_names = [style_letter]
for style_img_name in style_img_names:
    style_img = load_image('English_Fonts/'+ style_name + "/" + style_img_name)
    style_imgs.append(style_img)

# initialized generated as white noise or clone of original image.
# Clone seemed to work better for me.

# generated = torch.randn(original_img.data.shape, device=device, requires_grad=True)
# generated = load_image("kha.png").requires_grad_(True)
generated = original_img.clone().requires_grad_(True)
model = VGG().to(device).eval()

# Hyperparameters
total_steps = int(sys.argv[4])
learning_rate = float(sys.argv[6])
alpha = float(sys.argv[7])
beta = float(sys.argv[8])
optimizer = optim.Adam([generated], lr=learning_rate)
loss = []

best_loss = 0

for step in tqdm(range(1,total_steps+1)):
    # print("{0:6d}".format(step),"/","{0:6d}".format(total_steps), end=" : ")
    # Obtain the convolution features in specifically chosen layers
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features_all = []
    for style_img in style_imgs:
        style_features = model(style_img)
        style_features_all.append(style_features)

    # Loss is 0 initially
    style_loss = original_loss = 0

    # iterate through all the features for the chosen layers
    for gen_feature, orig_feature in zip(
        generated_features, original_img_features):
        # batch_size will just be 1
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)
    
    for style_features in style_features_all:
        for gen_feature, style_feature in zip(
            generated_features, style_features):
            # batch_size will just be 1
            batch_size, channel, height, width = gen_feature.shape
            # Compute Gram Matrix of generated
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )
            # Compute Gram Matrix of Style
            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )
            style_loss += torch.mean((G - A) ** 2)
            
    style_loss/=len(style_features_all)
    total_loss = alpha * original_loss + beta * style_loss
    loss.append(total_loss.item())
    if step == 1:
        best_loss = total_loss
    else:
        best_loss = total_loss
    # print("{0:6.2f}".format(alpha * original_loss), ":", "{0:12.2f}".format(beta * style_loss), ":", "{0:12.2f}".format(best_loss))
        
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        save_image(generated, result_file)
    # if step % 1000 == 0:
    #     save_image(generated, "generated_"+str(step)+".png")
        

# print(loss)
# x = np.arange(len(loss))
# plt.title("Line graph")  
# plt.xlabel("X axis")  
# plt.ylabel("Y axis")  
# plt.plot(x, loss, color ="green")  
# plt.show()

# img_array = np.array(Image.open('generated.png'))
# plt.imshow(img_array)