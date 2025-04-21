#ייבוא ספריות
import tensorflow as tf
from tensorflow.keras import Model # type: ignore

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

    #פונקציית פתיחת אופציית האימון של המודל
    #model = המודל שהתקבל כקלט
    def fine_tune(self, model):
        #הפיכת המודל המתקבל כקלט לאפשרי לאימון
        model.trainable = True
        #הקפאת 14 השכבות הראשונות
        freeze_until = 14
        #לולאה שעוברת על 14 השכבות הראשונות ומקפיאה את האימון שלהן
        for layer in model.layers[:freeze_until + 1]:
            layer.trainable = False
            
        #וגן בשביל להקל על עומס האימון overfitting אנחנו מקפיאים את השכבות הראשונות בכדי שהמודל קודם כל לא יגיע למצב בו הוא מתאמן נורא טוב רק על הדאטה שלנו ונקבל 
        #החזרת המודל בו יש רק 14 שכבות קפואות ולא כולן קפואות
        return Model(inputs=model.inputs, outputs=model.outputs)
