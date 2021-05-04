The code below generates the style transferred glyph of one kannada letter.

INPUTS
Before we run the code lets understand the inputs :

letter          : The image name of the input Kannada letter that we want to generate
                : Image name of any letter within the Kannada font directory

Style           : The letter style you want to copy
                : Can be any English font from the English_Fonts folder

style_letter    : the letter of english font that we wannt to use as style image
                : Image name of any letter within the English font directory

epochs          : The total number of steps to run the model
                : Ideally between 3,000 to 10,000

image_size      : The dimension of the image generated
                : Ideally between 200 to 400

learning_rate   : Hyper-parameter to get better accuracy
                : Ideally between 0.001 to 0.01

alpha           : Hyper-parameter to get better accuracy
                : Ideally between 1 to 100

beta            : Hyper-parameter to get better accuracy
                : Ideally between 0.001 to 10

OUTPUTS
The output image can be found in the Generated folder
Inside the style chosen 
With the name letter_O.png where letter is the name of the Kannada input letter

In the example below the generated image would be : Generated/Locomo/013_O.png

letter="013.png"
Style="Locomo" 
style_letter="042.png"
epochs=4000
image_size=200
learning_rate=0.01
alpha=10
beta=0.01
python3 nst.py $letter $Style $style_letter $epochs $image_size $learning_rate $alpha $beta