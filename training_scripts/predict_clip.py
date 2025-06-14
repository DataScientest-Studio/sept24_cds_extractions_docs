import torch
import clip
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import ImageEnhance
from datetime import datetime  # Pour suivre l'heure et le temps
from PIL import ImageOps, ImageFilter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score

# Fonction pour lire le fichier train.txt et retourner un DataFrame
def read_labels_from_txt(file_path):
    """
    Lit un fichier .txt et cree un DataFrame avec les colonnes 'file_name' et 'label'.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip().split(' '))
    df = pd.DataFrame(data, columns=['file_name', 'label'])
    df['label'] = df['label'].astype(int)  # Convertir les labels en entiers
    return df

# Charger le modele CLIP et le preprocesseur
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Lire le fichier train.txt
#train_file = "train.txt"  
train_file ="train.txt"
data_dir = "./data/images/"  

df = read_labels_from_txt(train_file)

print(f"Nombre total d'images dans le fichier : {len(df)}")

# Fichier pour sauvegarder les resultats
result_file = "result_clip.txt"
with open(result_file, 'w') as f_out:
    f_out.write("Image_Path, True_Class, Predicted_Class, Probability\n")

# Variables pour les graphiques
true_classes = []
predicted_classes = []


# Fonction pour ajuster le contraste de l'image
def adjust_contrast(image_path, factor=1.5):
    """
    Augmente le contraste de l'image.
    :param image_path: Chemin de l'image.
    :param factor: Facteur d'ajustement du contraste (>1 pour augmenter, <1 pour réduire).
    """
    try:
        image = Image.open(image_path)
        enhancer = ImageEnhance.Contrast(image)
        image_contrasted = enhancer.enhance(factor)
        return image_contrasted
    except Exception as e:
        print(f"Erreur lors de l'ajustement du contraste : {e}")
        return None


# Fonction pour extraire les contours d'une image
def extract_edges(image):
    """
    Extrait les contours de l'image en utilisant l'algorithme FIND_EDGES.
    :param image: Image PIL ouverte.
    :return: Image avec contours extraits.
    """
    try:
        # Convertir en niveaux de gris pour simplifier le processus de contours
        image_gray = image.convert("L")
        image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
        return image_edges
    except Exception as e:
        print(f"Erreur lors de l'extraction des contours : {e}")
        return None

    
# Fonction pour predire la classe d'une image
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Erreur : Le fichier '{image_path}' n'existe pas.")
        return None, None

    try:
        
         # Appliquer l'ajustement du contraste
        image = adjust_contrast(image_path, factor=2)

        if image is None:
            return None, None  # Si l'ajustement du contraste a echoue
       
       # Extraire les contours de l'image
        image = extract_edges(image)
        if image is None:
            return None, None  # Si l'extraction des contours échoue
       
        
        # redimensionner l'image
        #image = image.resize((1000, 760))
        image = image.resize((256, 256))
        
        # Pretraiter l'image
        image = preprocess(image).unsqueeze(0).to(device)
        
         # Tokenisation des classes et texte combiné
        text_inputs = clip.tokenize(classes).to(device)
        
        
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        predicted_class_idx = probs.argmax()
        return predicted_class_idx, probs
    except Exception as e:
        print(f"Erreur avec l'image {image_path}: {e}")
        return None, None

# Definir les classes (0 a 15 pour 16 classes)
classes = [f"Classe {i}" for i in range(16)]
text_inputs = clip.tokenize(classes).to(device)



# ---- Début du traitement ----
start_time = datetime.now()
print(f"Démarrage du traitement : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Predictions pour le jeu de donnees
for index, row in df.iterrows():
    path = os.path.join(data_dir, row['file_name'])
    label = row['label']

    try:
        predicted_class_idx, probs = predict_image(path)
        if predicted_class_idx is not None:
            true_classes.append(label)
            predicted_classes.append(predicted_class_idx)

            # Sauvegarder les resultats dans le fichier
            with open(result_file, 'a') as f_out:
                f_out.write(f"{path}, {label}, {predicted_class_idx}, {probs[0][predicted_class_idx]:.4f}\n")
        else:
            print(f"Prediction echouee pour l'image : {path}")
    except Exception as e:
        print(f"Erreur inattendue pour l'image : {path}. Detail : {e}")

    # Afficher la progression toutes les 500 images
    if (index + 1) % 500 == 0:
        print(f"Progression : {index + 1}/{len(df)} images traitees")

print(f"Nombre d'images traitees avec succes : {len(true_classes)}")


# ---- Fin du traitement ----
end_time = datetime.now()
print(f"Fin du traitement : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Temps total pris
elapsed_time = end_time - start_time
print(f"Temps total de traitement : {elapsed_time}")

print(f"Taille des true_classes : {len(true_classes)}")
print(f"Taille des predicted_classes : {len(predicted_classes)}")
print(f"Classes dans true_classes : {np.unique(true_classes, return_counts=True)}")

# Matrice de confusion

cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap="viridis", ax=ax)
#disp.plot(cmap="viridis", ax=plt.gca())
plt.title("Matrice de Confusion contraste facteur : 1.5, Taille : 256,256, Contour  ")
# Fixer les positions et les labels des ticks
ax.set_xticks(range(len(classes)))  # Positions des ticks (0 à 15)
ax.set_xticklabels(classes, rotation=45, ha="right")  # Inclinaison des labels

plt.show()

# Rapport de classification
print("\nRapport de classification :")
report = classification_report(true_classes, predicted_classes, target_names=classes)
print(report)

# Précision, rappel et F1-Score individuels
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print(f"Précision globale : {precision:.2f}")
print(f"Rappel global : {recall:.2f}")
print(f"F1-Score global : {f1:.2f}")


# Generation des graphiques
plt.figure(figsize=(10, 8))
counts, bins, _ = plt.hist(true_classes, bins=np.arange(0, 17), alpha=0.7, label='Classes réelles', color='blue', density=False)
counts_pred, bins_pred, _ = plt.hist(predicted_classes, bins=np.arange(0, 17), alpha=0.7, label='Classes prédites', color='orange', density=False)

# Annoter les valeurs sur les barres
for i in range(len(counts)):
    plt.text(bins[i] + 0.5, counts[i], str(int(counts[i])), ha='center', va='bottom', fontsize=10)


plt.title("contraste facteur : 1.5, Taille : 256,256, Contour  ")
plt.xlabel("Classes")
plt.ylabel("Nombre d'Images")
plt.legend()
plt.show()

# Calcul et affichage de la precision
correct_predictions = [1 if true == pred else 0 for true, pred in zip(true_classes, predicted_classes)]
accuracy = sum(correct_predictions) / len(correct_predictions)

plt.figure(figsize=(6, 4))
plt.bar(["Precision"], [accuracy], color='green')
plt.ylim(0, 1)
plt.title("Precision des Predictions")
plt.ylabel("Precision")
plt.show()
