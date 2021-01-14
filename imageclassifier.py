import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from PIL import Image
from os import listdir
from resizeimage import resizeimage
import matplotlib as mpl
import matplotlib.pyplot as plt

# =================================== Control Panel =======================================
training = "Off"                  # It's either "On" or "Off"
reduce_data = "Off"               # It's either "On" or "Off"
n_training = 20000                # How many training images to keep
n_validating = 4000               # How many validating images to keep
training_times = 20               # How many epochs will be used
orientation = 'Portrait'          # It's either 'Landscape' or 'Portrait' for ML-Metrics
# =========================================================================================

# =================================== Graph Settings ======================================
color_train = 'royalblue'
validation_color = 'orange'
transparency = 1.0
line_size = 2.0
# -------------------------Settings Tex and Math Fonts and Size of Axes--------------------
plt.rcParams['text.usetex'] = True
plt.rc('font', family='Latin Modern', serif='Latin Modern Math')
plt.rcParams.update({'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Latin Modern Math',
                     'mathtext.cal': 'Latin Modern Math'})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=12)
plt.rc('axes', linewidth=4)
fontsize_global = 20
# =========================================================================================

# ============================== Size and Shape of the figure =============================
widescreen = 1.414285714             # A4 scale
scale = 29.7                         # width of A4 in cm

if orientation == 'Landscape':
    width = scale
    height = width / widescreen
elif orientation == 'Portrait':
    height = scale
    width = height / widescreen
else:
    print('Please, insert a valid input either "Landscape" or "Portrait"')
# =========================================================================================

# ========================== Convert centimeteres into inches =============================
inches_is_one_centimetre = 0.393701
width_in_inches = width * inches_is_one_centimetre
height_in_inches = height * inches_is_one_centimetre
# =========================================================================================

# ===================================== Plots Setup =======================================
# Plot of Accuracy
x1_min = 0.0
x1_max = training_times - 1
y1_min = 0
y1_max = 1.0
# Plot of Loss
x2_min = x1_min
x2_max = x1_max
y2_min = 0.0
y2_max = 10.0
# =========================================================================================

# ================================ Positioning of Plots ===================================
fig, (ax1, ax2) = plt.subplots(2)
fig.subplots_adjust(left=.2, bottom=.215, right=.89, top=.89)
grid = plt.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[1, 1], wspace=0.1, hspace=0.1)
ax1 = plt.subplot(grid[0, 0:])
ax2 = plt.subplot(grid[1, 0:])

# Plot of Accuracy
ax1.set_ylabel(r'Accuracy', fontsize=fontsize_global, labelpad=15)
for tick in ax1.get_xticklabels():
    tick.set_fontname("Latin Modern Math")
ax1.set_xlim(x1_min, x1_max)

for tick in ax1.get_yticklabels():
    tick.set_fontname("Latin Modern Math")
ax1.set_ylim(y1_min, y1_max)

ax1.tick_params(axis='x', direction='out', length=7, width=1, color='black', pad=10,
                grid_color='black', grid_alpha=0.3, grid_linewidth=0.5)
ax1.set_xticklabels([])

# Plot of Loss
ax2.set_xlabel(r'Epochs', fontsize=fontsize_global, labelpad=15)
ax2.set_ylabel(r'$log$(Loss)', fontsize=fontsize_global, labelpad=15)
for tick in ax2.get_xticklabels():
    tick.set_fontname("Latin Modern Math")
ax2.set_xlim(x2_min, x2_max)

for tick in ax2.get_yticklabels():
    tick.set_fontname("Latin Modern Math")
ax2.set_ylim(y2_min, y2_max)

ax2.tick_params(axis='both', direction='out', length=7, width=1, color='black', pad=10,
                grid_color='black', grid_alpha=0.3, grid_linewidth=0.5)
# =========================================================================================

# ======================================= Set Title =======================================
ax1.set_title(r"Machine Learning Metrics", fontname="Arial", fontsize=22, 
                fontweight='normal', loc='center', pad=20)
# =========================================================================================

# ============================== Load "Cifar10" Database via Keras ========================
(training_images, training_labels), (validating_images, validating_labels) = datasets.cifar10.load_data()
# =========================================================================================

# ================================= Normalize RGB of Images ===============================
training_images, validating_images = training_images / 255, validating_images / 255
# =========================================================================================

# =================================== Define class names ==================================
class_names = ['Plane', 'Car', 'Bird', 'Cat',
               'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# =========================================================================================

# ======================= Define how many training and validating images ==================
if (reduce_data == "On"):
    training_images = training_images[:n_training]
    training_labels = training_labels[:n_training]
    validating_images = validating_images[:n_validating]
    validating_labels = validating_labels[:n_validating]

elif (reduce_data == "Off"):
    exit
else:
    print("Please select a valid training input:: Either On or Off")
# =========================================================================================

if (training == "On"):

    # ====================== Outup Format of Machine Learning Metrics =====================
    mpl.use('pdf')
    # =====================================================================================

    # ======================== Build our Neural Network Demo Model ========================
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    # =====================================================================================

    # =============================== Compile the Model ===================================
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(training_images, training_labels, epochs=training_times, 
                        validation_data=(validating_images, validating_labels))
    loss, accuracy = model.evaluate(validating_images, validating_labels)
    ax1.plot(history.history['accuracy'], c=color_train, alpha=transparency,
             linewidth=line_size, linestyle='-', zorder=1.0, label='train')
    ax1.plot(history.history['val_accuracy'], c=validation_color, alpha=transparency,
             linewidth=line_size, linestyle='-', zorder=1.0, label='validation')
    ax2.plot(history.history['loss'], c=color_train, alpha=transparency,
             linewidth=line_size, linestyle='-', zorder=1.0, label='train')
    ax2.plot(history.history['val_loss'], c=validation_color, alpha=transparency,
             linewidth=line_size, linestyle='-', zorder=1.0, label='validation')
    # =====================================================================================

    # ==================================== Legends ========================================
    ax1.legend()
    ax2.legend()
    # =====================================================================================

    # =================================== Save pdf ========================================
    fig.set_size_inches(width_in_inches, height_in_inches)
    fig.savefig('ml-metrics.pdf')
    fig.clear()
    # =====================================================================================

    # ================================== Save model =======================================
    model.save('Image_Classifier.model')
    # =====================================================================================

elif (training == "Off"):

    # =============================== Clear Figure Data ===================================
    plt.clf()
    # =====================================================================================

    # =============================== Load Trained Model ==================================
    model = models.load_model('Image_Classifier.model')
    # =====================================================================================

    # ========================= Load Images for Classification ============================
    loaded_images = list()
    for filename in listdir('Images_for_Classification'):
        with open('Images_for_Classification/' + filename, 'r+b') as f:
            with Image.open(f) as image:
                hd_img = cv.imread('Images_for_Classification/' + filename)

                # ======== Change Images codification from BGR to RGB =====================
                hd_img = cv.cvtColor(hd_img, cv.COLOR_BGR2RGB)
                # =========================================================================

                # ========== Resize and Save Low Definition Images ========================
                cover = resizeimage.resize_cover(image, [32, 32])
                cover.save('ld_pic.jpeg', image.format)
                # =========================================================================

        # ======================= Load Low Definition Images ==============================
        img = cv.imread('ld_pic.jpeg')
        # =================================================================================

        # ============= Change Images codification from BGR to RGB ========================
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # =================================================================================

        # ============================= Prediction ========================================
        prediction = model.predict(np.array([img]) / 255)
        index = np.argmax(prediction)
        print(f'Prediction is {class_names[index]}')
        # =================================================================================

        # ==== Showing in High Definition which Image is being Predicted by our Model =====
        plt.imshow(hd_img, cmap=plt.cm.binary)
        plt.show()
        # =================================================================================
else:
    print("Please select a valid training input:: Either On or Off")