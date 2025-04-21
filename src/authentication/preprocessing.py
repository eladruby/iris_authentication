#ייבוא ספריות
import os
import numpy as np
import tensorflow as tf
import sys

#מוסיף את האופציה לראות את כל התיקיות שנמצאות שתי נתיבים אחורה כלומר שתי תיקיות החוצה בכדי שנוכל לייבא פונקציות
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#ייבוא פונקציית האיתור ונירמול הקשתית
from src.segmentation.iris_url_normalization import IrisURLNormalization

#יצירת מחלקה לטיפול בייבוא ונירמול הדאטה-סט
class PreProcessing:
    #numpy של array יצירת מערך השומר את כל התמונות כ 
    image_train = np.array([])
    #numpy של array יצירת מערך השומר את כל התוויות כ 
    #כל תמונה תקבל תווית בהתאם למסר האדם אליו היא שייכת כלומר תמונות זהות יקבלו את אותה התווית
    label_train = np.array([])
    #מערך של כל התוויות רק מבלי לחזור על כל תווית, כלומר המערך כולל ספירה מעלה מ0 ועד למספר האנשים הנמצאים בדאטה-סט
    unique_train_label = np.array([])
    #ספרייה בה כל תמונה תקבל אינדקס מיוחד משלה מ0 ומעלה כלומר שהאינדקס הגדול ביותר יהיה פשוט מספר התמונות בדאטה-סט
    #ייראה בערך ככה
    # 0: [0, 1, 2],
    # 1: [3, 4, 5], ....
    # 225: [675, 676, 677]
    map_train_label_indices = dict()

    #פעולה שמתבצעת באופן אוטומטי כשיוצרים את המחלקה
    #מקבלת את המיקום של הדאטה-סט
    def __init__(self, data_src, TRIPLETS):
        
        #הגדרת משתנה שמקבל את כמות השלישיות שנרצה ליצור
        self.TRIPLETS = TRIPLETS
        #הגדרת נתיב הדאטה-סט של המחלקה לנתיב שהוכנס כקלט
        self.data_src = data_src

        print("Loading the Dataset...")
        #הפעלת הפונקציה והכנסתה לשני המשתנים שיצרנו ממקודם: מערך התמונות ומערך התוויות של התמונות
        self.image_train, self.label_train = self.preprocessing()
        #הלוקחת מערך של מספרים ומוציאה כפילויות unique הנקראת numpy הכנסת התוויות המופיעות במשתנה התוויות רק פעם אחת בשימוש בפונקציה של
        self.unique_train_label = np.unique(self.label_train)
        #יצירת מילון שבו לכל תווית (כלומר לכל אדם) יש רשימה של אינדקסים שבהם מופיעות התמונות שלו
        self.map_train_label_indices = {
            label: np.flatnonzero(self.label_train == label)
            for label in self.unique_train_label
        }

        #הדפסות של סיכום ההרצה בכדי לראות שהכל תקין
        print("Preprocessing Done. Summary:")
        #מספר התמונות שאספנו
        print("Images trained: ", self.image_train.shape)
        #מספר התוויות שאספנו
        print("Labels trained: ", self.label_train.shape)
        #מספר האנשים שאספנו
        print("Unique labels: ", self.unique_train_label)

    #פונקציה הקוראת את כל התמונות מהדאטה-סט ומחזירה אותן בתור רשימה של תמונות ורשימה של תוויות
    def read_dataset(self):
        #משתנה שמטרתו לספור כמה תמונות יש בכל הדאטה-סט כולו
        count = 0
        #הבאת רשימת כל תיקיות האנשים מהתיקייה הראשית של הדאטה-סט
        directories = os.listdir(self.data_src)
        #לולאה שבודקת כמה תמונות יש בסך הכל בדאטה-סט
        #הלולאה מתחילה בלרוץ כל פעם על תיקייה אחרת
        for directory in directories:
            #שזה תמונה הנמצא בתיקייה יש בכדי לספור אותן file בתוך כל תיקייה היא נכנסת לראות כמה
            #os.path.join(self.data_src, directory) מחבר את הנתיבים שנמצאים בתוך הסוגריים לנתיב אחד
            count += len([file for file in os.listdir(os.path.join(self.data_src, directory))])

        #יצירת שתי רשימות ריקות בגודל המתאים מראש, אחת לתמונות ואחת לתוויות
        #שתופסת מקום רב מהזיכרון append עושים זאת בכדי למנוע שימוש בפוקנציית
        x = [None] * count
        y = [None] * count
        #משתנה אינדקס שמתקדם כל פעם שנכנסת תמונה לרשימה
        index = 0

        #לולאה שרצה שוב על כל תיקייה בדאטה-סט
        for directory in directories:
            try:
                #מדפיס באיזו תיקייה אנחנו מטפלים כרגע
                print("Reading directory: ", directory)

                #לולאה על כל תמונה בתיקייה הנוכחית
                for pic in os.listdir(os.path.join(self.data_src, directory)):
                    #מפעיל את פונקציית הנירמול שיצרנו הלוקח תמונה של עין ומחזיר תמונה מנורמלת של הקשתית בלבד
                    img = IrisURLNormalization(os.path.join(self.data_src, directory, pic))
                    #keras המרה של התמונה למערך שמתאים למודל של
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    #הכנסת התמונה למערך התמונות
                    x[index] = img
                    #הכנסת מספר הבן אדם או התיקייה כתווית מקבילה לתמונה
                    y[index] = directory
                    #התקדמות של האינדקס לתמונה הבאה
                    index += 1

            #אם קורה משהו לא תקין למשל תיקייה ריקה או תמונה פגומה אז מדפיסים שגיאה וממשיכים האלה בכדי שהקוד לא ייקרוס
            except Exception as e:
                print("Error reading directory: ", directory)
                print(e)

        print("Finished reading dataset")

        #החזרת רשימת התמונות ורשימת התוויות
        return x, y

    #keras פונקציה שמשתמשת בפונקציה הקודמת לקריאת כל התמונות בדאטה-סט. מערבבת אותם ומחזירה אותם בצורה הנכונה לעבודה עם ספריית 
    def preprocessing(self):

        #הרצת פונקציית הקריאה שתחזיר לנו את כל התמונות והתוויות מהדאטה-סט
        x, y = self.read_dataset()

        #ממירה את רשימת התוויות לרשימה של תוויות ייחודיות (ללא כפילויות) כדי שנוכל למספר אותן
        labels = list(set(y))

        #ממיינת את התוויות הייחודיות לפי סדר מספרי כדי שייראה כך ['0','1','2','3']
        labels.sort(key=int)

        #מביא לכל שם ברשימה מספר כך שייראה ככה
        #[('3', 3), ('2', 2), ('1', 1), ('0', 0)]
        #ואז עוטף אותו בספרייה כך שייראה כך
        #{'3':3, '2':2, '1':1, '0':0}}
        label_dict = dict(zip(labels, range(len(labels))))

        #numpy בלבד והכנסתם למערך של int הפיכת כל קבוצה בספרייה למספר 
        #([0, 1, 2, 3])
        y = np.asarray([label_dict[label] for label in y])

        #פונקציה שמייצרת מערך של מספרים עולים מ0 ועד לאורך התוויות כלומר מערך בגודל של מספר התמונות שלנו בו מספרים מבולגנים 
        #ייראה בערך ככה
        # [5, 2, 6, 4, 0, 1, 3]
        shuffle_indices = np.random.permutation(len(y))

        #יצירת שתי רשימות ריקות שיקבלו את התמונות והתוויות אחרי הערבוב
        x_shuffled = []
        y_shuffled = []

        #לולאה שרצה על כל אינדקס בסדר החדש וממלאת את הרשימות כך שעכישו היא מתערבבת
        for index in shuffle_indices:
            x_shuffled.append(x[index])
            y_shuffled.append(y[index])

        #keras בכדי שנוכל לעבוד איתן בתוך מודלים של numpy החזרת הרשימות כרשימות של
        return np.asarray(x_shuffled), np.asarray(y_shuffled)

    
    #פונקציה שיוצרת שלישיות של שתי תמונות כמעט זהות של אותו אדם ותמונה של אדם שונה
    #תמונה ראשית  (anchor) חיובית ודומה (positive) שלילית ושונה (negative)
    def get_triplets(self):

        #פונקציה שבוחרת שני אנשים שונים בעזרת תיקיית התוויות המיוחדות שלא מופעיות כפילויות
        #label_l - חיובי
        #label_r - שלילי
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)

        #label_l מתוך התווית הראשונה שבחרנו שולפים שני אינדקסים שונים שמייצגים שתי תמונות עיניים של אותו אדם
        #הראושנה תהווה את התמונה הראשית ושניה תהווה את התמונה החיובית
        a, p = np.random.choice(self.map_train_label_indices[label_l], 2, replace=False)

        #label_r בוחרים אינדקס רנדומלי מתמונות עיניים של אדם אחר 
        n = np.random.choice(self.map_train_label_indices[label_r])

        #יש להבהיר שהאינדקס שנבחר מכל תמונה הוא האינדקס הראשי כלומר לא מ0 ועד3 אלה מ0 ועד מספר התמונות הקיימות
        #לדוגמה
        #a = 22, p = 23, n = 555

        #החזרת שלושת האינדקסים
        return a, p, n

    
    #פונקציה שמכינה אובייקט דאטה סט של שלישיות מוכנות
    def get_triplets_batch(self):

        #יצירת שלוש רשימות ריקות שיאחסנו את האינדקסים של התמונות שנבחרו עבור כל שלישייה
        indexs_a, indexs_p, indexs_n = [], [], []

        #קובעים כמה שלישיות נרצה לייצר במקרה הזה 500 שלישיות
        n = self.TRIPLETS

        #לולאה שתרוץ 500 פעמים ותייצר בכל פעם שלישייה חדשה של אינדקסים
        for _ in range(n):
            #כל פעם מכינה שלישייה רנדומלית חדשה
            a, p, n = self.get_triplets()

            #מוסיפה כל אינדקס לרשימה המתאימה
            indexs_a.append(a)
            indexs_p.append(p)
            indexs_n.append(n)

        #Dataset נרצה להפוך את השלישיות לאובייקט Tensorflow בכדי לעבוד ביעילות עם אימון של 
        #לאובייקט anchor נכניס את כל התמונות 
        anchor_dataset = tf.data.Dataset.from_tensor_slices(self.image_train[indexs_a, :])

        #לאובייקט positive נכניס את כל התמונות 
        positive_dataset = tf.data.Dataset.from_tensor_slices(self.image_train[indexs_p, :])

        #לאובייקט negative נכניס את כל התמונות 
        negative_dataset = tf.data.Dataset.from_tensor_slices(self.image_train[indexs_n, :])

        #עכשיו מחברים במקביל את שלושת האובייקטים הנפרידם לאובייקט אחד משותף בו בכל אינדקס יש שלישייה של תמונות
        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))

        #מחזיר את האובייקט הסופי  שהוא בעצם 500 שלישיות של מוכנות לאימון
        return dataset
