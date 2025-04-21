#ייבוא הספריות
import os
from preprocessing import load_data # type: ignore
from unet_model import build_unet # type: ignore
import tensorflow as tf

# הגדרת מיקום נתיב הדאטה-סט ומיקום שמירת קובץ המודל המאומן
IMAGES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "UBIRIS_V2_Database", "iris")
MASKS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "UBIRIS_V2_Database", "labels")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "segmentation_model.keras")


#כמות האיפוקים
EPOCHS = 2
#כמות התמונות שמקובצות ביחד בכל איטרציה
BATCH_SIZE = 16
#יחס ההפרדה בין הדאטה של האימון לבין דאטה הולידציה הנועד רק לבדיקה
#דאטה ולידציה - 20%
#דאטה אימון - 80%
VAL_SPLIT = 0.2

#טעינת הדאטה-סט לשני מערכים, אחד לעיניים והשני למסיכות
#preprocessing.py יש פה שימוש בפונקציה מהקובץ
#128 אלה ממדי האורך והרוחב של התמונה
X_train, Y_train = load_data(IMAGES_PATH, MASKS_PATH, 128, 128)

#U-Netבניית מודל ה
#unet_model.py מתבצע גם כאן שימוש בפונקציה מהקובץ 
model = build_unet()
#קימפול המודל עם האופטימייזר אדם ופונקציית מחיר למודלים עם שני פלטים כמו אצלנו
# metrics=['accuracy'] - של המודל accuracy משתמש כדי להציג את המדד
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#הדפסת סיכום המודל
model.summary()

#אימון המודל עם ההיפר פרמטרים ושני מערכי הדאטה-סט
model.fit(X_train, Y_train, validation_split=VAL_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE)

#שמירת המודל המאומן בנתיב שנחבר
model.save(MODEL_PATH)
print("Model saved to:", MODEL_PATH)
