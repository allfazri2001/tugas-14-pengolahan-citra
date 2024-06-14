# tugas-14-pengolahan-citra
# Nama : Fajri Al Jauhari
# NIM :  312210476
# Kelas : TI.22.A.5

import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assign and open image from URL
url = 'https://i.pinimg.com/564x/3f/d9/65/3fd9654453ca38395a9024513fc8a189.jpg'
response = requests.get(url, stream=True)

# Saving the image
with open('image.png', 'wb') as f:
    f.write(response.content)

# Read the image using OpenCV
img = cv2.imread('image.png')

# Converting the image into gray scale for faster computation.
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculating the SVD
u, s, v = np.linalg.svd(gray_image, full_matrices=False)

# Inspect shapes of the matrices
print(f'u.shape:{u.shape}, s.shape:{s.shape}, v.shape:{v.shape}')

# Calculate variance explained by each singular value
var_explained = np.round(s*2 / np.sum(s*2), decimals=6)

# Variance explained top Singular vectors
print(f'Variance Explained by Top 20 singular values:\n{var_explained[0:20]}')

# Plotting the variance explained by the top 20 singular values
sns.barplot(x=list(range(1, 21)), y=var_explained[0:20], color="dodgerblue")

plt.title('Variance Explained Graph')
plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.show()

# Plot images with different number of components
comps = [3648, 1, 5, 10, 15, 20]
plt.figure(figsize=(12, 6))

for i in range(len(comps)):
    low_rank = u[:, :comps[i]] @ np.diag(s[:comps[i]]) @ v[:comps[i], :]

    plt.subplot(2, 3, i + 1)
    plt.imshow(low_rank, cmap='gray')
    if i == 0:
        plt.title(f'Actual Image with n_components = {comps[i]}')
    else:
        plt.title(f'n_components = {comps[i]}')

plt.tight_layout()
plt.show()

![Screenshot 2024-06-14 131650](https://github.com/allfazri2001/tugas-14-pengolahan-citra/assets/167978131/32e190fd-3d58-4716-a126-bb7d2b58315c)
