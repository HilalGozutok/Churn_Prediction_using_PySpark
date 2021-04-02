##################################################
# Churn Prediction using PySpark
##################################################
"""
Surname : Soy isim
CreditScore : Kredi skoru
Geography : Ülke (Germany/France/Spain)
Gender : Cinsiyet (Female/Male)
Age : Yaş
Tenure : Kaç yıllık müşteri
Balance : Bakiye
NumOfProducts : Kullanılan banka ürünü
HasCrCard : Kredi kartı durumu (0=No,1=Yes)
IsActiveMember : Aktif üyelik durumu (0=No,1=Yes)
EstimatedSalary : Tahmini maaş
Exited : Terk mi değil mi? (0=No,1=Yes)
"""

import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init("c:\spark\spark")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext
# http://localhost:4040/jobs/

##################################################
# VERİ SETİ
##################################################

spark_df = spark.read.csv(r"C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_12\churn2.csv", header=True, inferSchema=True)
spark_df
spark_df.show()

##################################################
# EDA
##################################################

# Gözlem ve değişken sayısı
print("Shape: ", (spark_df.count(), len(spark_df.columns)))

# Değişken tipleri
spark_df.printSchema()
spark_df.dtypes

# Değişken seçme
spark_df.Age

# Head
spark_df.head(5) # tuple içinde liste tarzında
spark_df.show(5) # sparktaki asıl head()
spark_df.take(5) # liste tarzında

# Değişken isimlerinin küçültülmesi
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)

# özet istatistikler
spark_df.describe().show()

# sadece belirli değişkenler için özet istatistikler
spark_df.describe(["age", "balance"]).show()

# Kategorik değişken sınıf istatistikleri
spark_df.groupby("exited").count().show()

# Eşsiz sınıflar
spark_df.select("exited").distinct().show()

# select(): Değişken seçimi
spark_df.select("age", "surname").show(5)

# filter(): Gözlem seçimi / filtreleme
spark_df.filter(spark_df.age > 40).show()
spark_df.filter(spark_df.age > 40).count()

# groupby işlemleri
spark_df.groupby("exited").count().show()
spark_df.groupby("exited").agg({"age": "mean"}).show()

# Tüm numerik değişkenlerin seçimi ve özet istatistikleri
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().toPandas().transpose()

# Tüm kategorik değişkenlerin seçimi ve özeti
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()

# Exited'a göre sayısal değişkenlerin özet istatistikleri
for col in num_cols:
    spark_df.groupby("exited").agg({col: "mean"}).show()

##################################################
# DATA PREPROCESSING & FEATURE ENGINEERING
##################################################

##################################################
# Missing Values
##################################################

from pyspark.sql.functions import when, count, col
# spark_df col da gez bir kolon null barındırdığında bunun count ını al ve yazdır.
spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T
spark_df.dropna().show()
##################################################
# Feature Interaction
##################################################

spark_df.show(5)
spark_df = spark_df.withColumn('NEW_age/est', spark_df.age / spark_df.estimatedsalary)


##################################################
# Bucketization / Bining / Num to Cat
##################################################

############################
# Bucketizer ile Değişken Türetmek/Dönüştürmek
############################

from pyspark.ml.feature import Bucketizer

spark_df.select('age').describe().toPandas().transpose()

bucketizer = Bucketizer(splits=[0, 35, 45, 95], inputCol="age", outputCol="age_cat")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df.show(20)

spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1)
spark_df.groupby("age_cat").count().show()
spark_df.groupby("age_cat").agg({'exited': "mean"}).show()

spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer"))
spark_df.show()
############################
# when ile Değişken Türetmek (segment)
############################

spark_df = spark_df.withColumn('segment', when(spark_df['tenure'] < 5, "segment_b").otherwise("segment_a"))
spark_df.show()
############################
# when ile Değişken Türetmek (age_cat_2)
############################

spark_df = spark_df.withColumn('age_cat_2',
                    when(spark_df['age'] < 36, "young").
                    when((35 < spark_df['age']) & (spark_df['age'] < 46), "mature").
                    otherwise("senior"))
"""
spark_df = spark_df.withColumn('NEW_Tenure_Status',
                               when(spark_df['tenure'] < 2, "New").
                               when((2 < spark_df['tenure']) & (spark_df['tenure'] < 6), "Accustomed").
                               when((5 < spark_df['tenure']) & (spark_df['tenure'] < 14), "Loyal").
                               otherwise("Constant"))

"""
spark_df = spark_df.withColumn('Geography_cat',
                               when(spark_df['geography'] == 'Germany', 1).
                               when(spark_df['geography'] == 'France', 2).
                               when(spark_df['geography'] == 'Spain', 3).otherwise(0))
spark_df = spark_df.drop("geography")
spark_df.show()
##################################################
# Label Encoding
##################################################

# değişkenleri StringIndexer metodunu kullanarak dönüştürüyoruz.
spark_df.show()
indexer_cols = ["segment", "gender","age_cat_2","NEW_Tenure_Status"]

for col in indexer_cols:
    name = col + "_label"
    indexer = StringIndexer(inputCol=col, outputCol=name)
    temp_sdf = indexer.fit(spark_df).transform(spark_df)
    spark_df = temp_sdf.withColumn(name, temp_sdf[name].cast("integer"))
    spark_df = spark_df.drop(col)

spark_df.show()


##################################################
# One Hot Encoding
##################################################
ohe_cols = ["age_cat"]
for col in ohe_cols:
    name = col + "_ohe"
    encoder = OneHotEncoder(inputCol=col, outputCol=name)
    spark_df = encoder.fit(spark_df).transform(spark_df)
    spark_df = spark_df.drop(col)

spark_df.show(5)

##################################################
# TARGET'ın Tanımlanması
##################################################

# TARGET'ın tanımlanması
stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))

spark_df.show()
spark_df = spark_df.drop('age','age_cat','surname','exited',"age_cat_2")
spark_df.show()
##################################################
# Feature'ların Tanımlanması
##################################################
cols = ['tenure','hascrcard','numofproducts',
        'isactivemember', 'NEW_age/est',
        'Geography_cat','segment_label',
        'age_cat_2_label','age_cat_ohe']


# Vectorization
va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()
final_df = va_df.select("features", "label")
final_df.show(5)

train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show()
test_df.show(10)

# final df with set of features, label variables
final_df = va_df.select('features', 'label')
train_df, test_df = final_df.randomSplit([0.7, 0.3])
print('Training Dataset Count: ' + str(train_df.count()))
# Training Dataset Count: 6996
print('Test Dataset Count: ' + str(test_df.count()))
# Test Dataset Count: 3004

##################################################
# MODELING
##################################################

##################################################
# Logistic Regression
##################################################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.select("label", "prediction").show()
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count() # 0.82

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print(" accuracy: %f,\n precision: %f,\n recall: %f,\n f1: %f,\n roc_auc: %f" % (acc, precision, recall, f1, roc_auc))

# accuracy: 0.822903,
# precision: 0.835923,
# recall: 0.967378,
# f1: 0.790188,
# roc_auc: 0.613379


##################################################
# GBM
##################################################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)

y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

##################################################
# Model Tuning
##################################################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
