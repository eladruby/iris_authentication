#ייבוא ספריות
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

#הגדרת ממדי תמונת הקלט
def load_data(images_path, masks_path, IMG_WIDTH, IMG_HEIGHT):
    #הגדרת משתנים ריקים שיכילו את התמונות ואת המסיכות
    X, Y = [], []

    #לולאה שעוברת על כל תמונה בתיקיית העיניים
    for name in os.listdir(images_path):
        #קריאת התמונה והפיכתה לווקטור
        img = imread(os.path.join(images_path, name))
        #הקטנתה/הגדלתה לגודל הקלט של המודל שלנו
        img = resize(img, (IMG_WIDTH, IMG_HEIGHT))
        #נירמול התמונה כך שהערכים יהיו בין 0 ל-1
        img = img / 255.0
        #הוספת התמונה למערך התמונות
        X.append(img)

    #לולאה שעוברת על כל מסיכה בתיקיית המסיכות
    for mask in os.listdir(masks_path):
        #קריאת המסיכה והפיכתה לווקטור
        m = imread(os.path.join(masks_path, mask))
        #הקטנתה/הגדלתה לגודל הקלט של המודל שלנו והוספת מימד מאחר ומודל הקנובולוציה עובד עם תמונות ב-3 מימדים
        m = np.expand_dims(resize(m, (IMG_WIDTH, IMG_HEIGHT)), axis=-1)
        #נירמול התמונה כך שהערכים יהיו בין 0 ל-1
        img = img / 255.0
        #הוספת המסיכה למערך המסיכות
        Y.append(m)

    #ידע לעבוד איתם באימון המודל kerasבכדי ש  Numpy החזרת המערכים כמערך
    return np.array(X), np.array(Y)
