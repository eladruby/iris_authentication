#ייבוא ספריות
from tensorflow.keras import Model # type: ignore
from numpy.linalg import norm
import tensorflow as tf
import keras
import streamlit as st
from PIL import Image
import numpy as np
import os
import sys

#מוסיף את האופציה לראות את כל התיקיות שנמצאות שתי נתיבים אחורה כלומר שתי תיקיות החוצה בכדי שנוכל לייבא פונקציות
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#ייבוא פונקציית האיתור ונירמול הקשתית מקובץ נפרד
from src.segmentation.iris_img_normalization import IrisImgNormalization


#Triplet Lossיצירת מחלק במטרתה להכיל את הארכיטקטורה של מודל ה
class TripletLoss:
    #פונקציית הבנאי שקוראת בקריאה ליצירת אובייקט במחלקה
    def __init__(self):
        #הגדרת ממדי הקלט של המודל
        self.inp = tf.keras.Input(shape=(None, None, 3))

    #פונקציית בניית המודל
    def embedding(self):

        #VGG16 ייבוא מודל הקנובולוציה המוכר 
        #include_top=False כך שהפונקציה מגיעה ללא שכבת הסיווג מכיוון שאין לנו סיווג במקרה שלנו
        #weights=None כך שהמודל לא יטען משקולות שאומנו כבר מלפני כן בכדי שנוכל לאמן אותו על הדאטה-סט הספציפי שלי לתוצאות טובות יותר
        #input_tensor=self.inp כך שהקלט של המודל הוא הקלט שהגדרנו למעלה
        vgg16_fe = tf.keras.applications.VGG16(
            include_top=False, weights=None, input_tensor=self.inp
        )
        #מגדיר כך ששום שכבה של המודל לא תהיה ניתנת לאימון (רק לכרגע נשנה זאת עוד רגע)
        vgg16_fe.trainable = False
        #Flatten ואז Dense כבר לא נכון פה משום שלאחר שנעביר אותו בשכבת  Dense מכיוון שהקלט שלנו מגיע בצורה של מלבן ולא של ריבוע כלומר 400 על 75 אז השימוש בשכבת 
        #אז נאבד את כל המבנה המרחבי של התמונה מה שיקשה נורא על המודל ללמוד תכונות בעזרת מיקום וצורות שזה בדיוק איך שהמודל שלי עובד
        #לכן נשתמש בשכבת קונבולוציה עם פילטרים של 1 על 1 שלא תשטח לנו את תכונות המודל ועדיין תעבור על כל פיקסל ותלמד על חשיבותו מבלי להתייחס לממדי התמונה
        #stride=1 כדי שהשכבה תעבור על כל פיקסל בתמונה כמו שאמרנו
        #padding='same' כך שהשכבה לא תקטין את התמונה
        #activation='relu' פונקציית האקטיבציה של השכבה
        #name='out_conv' נותן שם לשכבה
        #שייבאנו ובעצם לקשר בין השניים ולבפוך אותם למודל אחד מחובר VGG16החלק האחרון בקוד נועד בשביל לחבר את השכבה הזו את אל השכבה האחרונה במודל ה 
        out_conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=1, strides=1, padding='same',
            activation='relu', name='out_conv')(vgg16_fe.layers[-1].output)
        #אחת הדרכים הנפוצות לשטח תמונה ולהשאיר רק את התכונות הרלוונטיות שלה MaxPooling עכשיו נחבר לשכבה הקודמת (האחרונה), שכבת 
        out = tf.keras.layers.GlobalMaxPooling2D()(out_conv)
        #החזרת כל המודל שבנינו
        #name='embedding' נותן שם למודל
        #באנגלית זה תכונה אז זה בעצם אומר שהמודל הזה לוקח תמונה ומחזיר את התכונות שלה embedding
        return Model(inputs=vgg16_fe.inputs, outputs=out, name='embedding')

#פונקציית נירמול הקלט
def preprocess(uploaded_file, segmentation, model):
    #פתיחת התמונה שהועלתה כתמונה צבעונית
    image = Image.open(uploaded_file).convert("RGB")
    #הרוחב הרצוי של התמונה
    w_target = 400
    #הרוחב והאורך של התמונה בפועל
    w_orig, h_orig = image.size

    #עכשיו נרצה להפוך את התמונה להיות ברוחב 400 ולהשאיר את היחס אורך רוחב שלה
    #חישוב האורך הרצוי של התמונה לפי היחס של התמונה המקורית
    h_target = int((w_target / w_orig) * h_orig)
    #הקטנת/הגדלת התמונה לאורך והרוחב הרצויים
    image_resized = image.resize((w_target, h_target))
    #העברת התמונה בפונקציית האיתור ונירמול הקשתית
    #אם סומן שיש צורך בסגמנטציה אז יצורף מודל הסגמנטציה לפונקציה כך שהיא תדע להשתמש בו
    if segmentation:
        img = IrisImgNormalization(image_resized, model)
    else:
        #אם לא אז הפונקציה תפעל כרגיל ללא סגמנטציה
        img = IrisImgNormalization(image_resized)
    #None אם לא נמצא קשתית בתמונה אז נחזיר 
    if img is None:
        return None
    
    #float32 הפיכת התמונה למערך של פיקסלים מסוג
    #מצפה לפורמט הזה VGG16מאחר ו
    #ואז מחלקים את כל הערכים ב255 כדי שהתמונה תהיה עם ערכים הנעים בין 0-1
    #לא יציג ערכים גבוהים מדי כמו 0-255 שיחזירו לנו שגיאה Streamlitעושים זאת בכדי ש
    #בגלל שהוא לא עובד עם ערכים גבוהים מדי
    img = np.array(img).astype("float32")

    #כי ככה הגדרנו את הקלט של המודל בבנייתו Batch בכדי שנוכל להכניס את התמונה למודל נצטרך להוסיף ממד
    img = np.expand_dims(img, axis=0)

    #לכן נוודא שתמיד יהיה לתמונה ממד צבע VGG16אם איכשהו הוחזרה תמונה בלי ממדי צבע אז היא לא תוכל להכינס למודל ה 
    if img.ndim < 4:
        img = np.stack([img] * 3, axis=-1)

    return img
    



#הגדרת משתנה למחלקה האחראית על מודל הטריפלט לוס שאיתו אימנו את המשקולות
model_copy = TripletLoss()
#יצירת המודל בעזרת הפונקציה שהגדרנו במחקלה של המודל
embedding = model_copy.embedding()

#בונים את הקלט של המודל בכדי שנוכל להלביש עליו את המשקולות המאומנות
embedding.build((75, 400, 3))

#הגדרת נתיב קוסץ המשקולות אותו נטען
weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "triplet_loss.weights.h5")
#טעינת המקט של המשקולות המאומנות על המודל
embedding.load_weights(weights_path)

#שזה בעצם כלי עזר ליצירת ממשקים בקוד פשוט Streamlitיצירת כותרת בממש ה
st.title("Iris Authentication System")

#ניצור שתי עמודות בכדי שנוכל להעלות את שתי התמונות
col1, col2 = st.columns(2)

#יצירת שתי כפתורים להעלת קובץ בשביל שנוכל לעלות את שתי התמונות שלהן אנחנו רוצים לחשב את המרחק
#type=["jpg"] - jpg כך שנוכל להעלות רק קבצי 
#מתחת לכל כפתור נמצא מתג שמחליט האם באיתור הקשתית של אותה התמונה יש צורך בשימוש סגמנטציה
#st.toggle - נותן ערך חיובי כשדלוק וערך שלילי כשכבוי
with col1:
    img_file1 = st.file_uploader("Upload Image 1 jpg file only*", type=["jpg"])
    seg1 = st.toggle("Use segmentation for Image 1")

with col2:
    img_file2 = st.file_uploader("Upload Image 2 jpg file only*", type=["jpg"])
    seg2 = st.toggle("Use segmentation for Image 2")


#בדיקה אם העלנו את שתי התמונות כך שרק אז נתחיל בחישוב המרחק
if img_file1 and img_file2:

    #אם לפחות באחת מהתמונות יש צורך בסגמנטציה אז הייבא את המודל סגמנטציה המאומן
    if seg1 or seg2:
        #יצירת הנתיב אל קובץ מודל הסגמנטציה
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "segmentation_model.keras")
        #מאחר ויצא עדכון בו אם לא מצהירים כי ברצוננו לטעון את קובץ המודל אז מקבלים שגיאה בכדי למנוע מוירוסים של קבצי מודלים באינטרנט להכנס למחשב
        keras.config.enable_unsafe_deserialization()
        #keras טעינת המודל בעזרת ספריית הטעינת מודלים של 
        model = keras.models.load_model(MODEL_PATH, compile=False)

    #העברת שתי התמונות בפונקציית הנירמול שיצרנו כעת
    img1 = preprocess(img_file1, seg1, model if seg1 else None)
    img2 = preprocess(img_file2, seg2, model if seg2 else None)

    #כלומר לא נמצאה הקשתית אז נחזיר שגיאה None אם לפחות אחת התמונות מחזירה
    if img1 is None or img2 is None:
        st.error("Could not find iris")
    else:
        #אם לא אז נציג את שתי התמונות לאחר הנירמול
        #st.image - מציג את התמונות
        #caption - נותן שם לכל תמונה
        #width - קובע את הרוחב של התמונה
        st.image([img1.squeeze() / 255, img2.squeeze() / 255],
                 caption=["Image 1", "Image 2"], width=150)

        #הכנסת התמונות למודל בכדי להוציא את ווקטור התכונות שלהן
        embedding1 = embedding.predict(img1)
        embedding2 = embedding.predict(img2)
        #חישוב המרחק בין הווקטורים של שתי התמונות
        #norm -  לוקח כל ערך במערך אחד ומחסר אותו בשני, את התוצאה הוא מעלה בריבוע, את סכום כל הריבועים של כל התוצאות הוא שם בתוך שורש וזה חישוב מרחק אוקלידי
        distance = norm(embedding1 - embedding2)

        #כתיבת המרחק בין שתי התמונות
        st.write("**Distance between embeddings:**", distance)
        #קביעת הערך המקסימלי למרחק של שתי תמונות שנחשבות זהות כלומר כל מה שמעל נחשב לעיניים שונות
        #קבעתי את המרחק לפי ניסוי וטעייה
        threshold = 1.5
        #בדיקה אם מרחק התמונות קטן מהמרחק המקסימלי
        if distance < threshold:
            #אם כן אז הן זהות
            st.success("Authentication Successful")
        else:
            #אם לא אז הן שונות
            st.warning("Authentication Failed")

