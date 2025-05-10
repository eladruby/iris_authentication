#ייבוא ספריות
from preprocessing import PreProcessing
from triplet_model import TripletLoss
import tensorflow as tf
import os

EPOCHS = 11
BATCH_SIZE = 2
#מספר השלישיות שנרצה ליצור (נראה זאת בהמשך)
TRIPLETS = 700
LEARNING_RATE = 0.0001
MARGIN = 0.7

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "IITT_Database")

#__init__  הפעלת הפונקציה שבניתי במחלקה לייבוא ונירמול הדאטה-סט
#אותה פונקציית בניה כוללת כמעט את כל הפונקציות שכתבתי
#כך שהיא הולכת לתיקיית הדאטה-סט ומתחילה לייבא את התמונות ולנרמל אותן
dataset = PreProcessing(data_src=DATA_PATH, TRIPLETS=TRIPLETS)
#כאובייקט במחלקה שיצרנו האחראית על המודל triplet_model הגדרת 
triplet_model = TripletLoss()
#שלנו Triplet Loss הפעלת פונקציית הבניה של המחלקה היוצרת את מודל 
embedding_model = triplet_model.embedding()

#את גודל הווקטור שנרצה לטעון עליו את המשקולות tensorflowבכדי שנוכל לטעון את המשקולות לפני שנכניס אליהן את ווקטור התמונה נצטרך להגדיר ל
#ידע איפה לרסס את השמן tensorflowכלומר השכבה הזו היא כמו תבנית כדי ש 
embedding_model(tf.random.normal([BATCH_SIZE, 75, 400, 3]))
# על מנת שנוכל לגרום למודל להתאמן על כמה תמונות בכל איטרציה BATCH_SIZE הכנת הדאטה סט לקבוצות בגודל 
#Triplet Loss נעשה פה שימוש בפונצייה ההופכת את הדאטה לשלישיות מאחר ואנחנו מתעסקים עם מודל
dataset = dataset.get_triplets_batch().batch(BATCH_SIZE)

#הגדרת אופטימייזר אדם עם קצב הלמידה שהגדרנו בהתחלה
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#הלולאה שכוללת את מהלכו של כל איפוק באימון
for epoch in range(EPOCHS):
    #לולאה הלוקחת כל תמונה מהשלישיה ומעבירה אותו בתוך תוכן הלולאה
    for anchor, positive, negative in dataset:
        #tape נותנים לנו ,מגדירים אותו כמשתנה Tensorflowחישוב הגראדיאנט האוטומטי ש
        with tf.GradientTape() as tape:
            #הכנסת התמונה אל המודל שמחזיר את הווקטור של התכונות העיקריות שלה
            #עושה את התהליך הזה לכל אחת משלישיית התמונות באותו המודל
            anchor_out = embedding_model(anchor)
            positive_out = embedding_model(positive)
            negative_out = embedding_model(negative)

            #המרחק בין התמונה החיובית לתמונה הראשית
            #המרחק מחושב לפי נוסחת המרחק האוקלידי המחשב את ההפרש בין כל ערך מתאים בשני התמונות שעכשיו הם ווקטורים ומעלה את התוצאה בריבוע
            #עושה ממוצע לכל הערכים בציר 1- כלומר הציר האחרון שהוא ציר של הווקטור  reduce_mean
            #(batch_size,    228)
            #     ^           ^
            #הציר האחרון הציר הראשון   
            pos_dist = tf.reduce_sum(tf.square(anchor_out - positive_out), axis=1)
            #אותו חישוב מרחק בין התמונה הראשית (תמונת העוגן) אל התמונה השלילית
            neg_dist = tf.reduce_sum(tf.square(anchor_out - negative_out), axis=1)
            #לפי הנוסחה שלה Triplet Loss מחשב את פונקציית המחיר
            loss = tf.maximum(pos_dist - neg_dist + MARGIN, 0.0)
            #מחשב את הממוצע של כל הערכים בכדי שהתוצאה תהיה מספר יחיד מאחר וזה ההפסד והוא אמור להיות מספר שמעיד על חוזק תיקון הטעות של המודל
            loss = tf.reduce_mean(loss)

        #מחשב את הגראדיאנט 
        gradients = tape.gradient(loss, embedding_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, embedding_model.trainable_variables))

    #Lossמדפיס את מספר האיפוק ביחד עם מדד ה
    print(f"Epoch {epoch+1} loss: {loss.numpy():.4f}")

#שומר את משקולות הפונקציה לנתיב שבחרנו תחילה
#os.path.dirname(__file__) - נתיב הקובץ הנוכחי
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "triplet_loss.weights.h5")
#פעולת שמירת המשקולות של המודל שאימנו
embedding_model.save_weights(SAVE_PATH)
print("Weights saved to:", SAVE_PATH)
