# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 19:07:59 2018

@author: david
"""
import csv
import xlrd
import shutil
import os

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import types

from pyspark.sql import functions
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression



def csv_from_excel():   # Convert xlsx to csv file
    if os.path.exists('./PQ_result'):   # Create directory, if existing then remove it first
        shutil.rmtree('./PQ_result')
    os.mkdir('./PQ_result')        
    xls = xlrd.open_workbook('G:/PQ_data_regression/PQ Assessment/sem2/results/test6 all BESS/results.xlsx')   # Select xlsx file and corresponding sheet
    sheet = xls.sheet_by_name('THD')
    
    with open('./PQ_result/THD.csv', 'w') as new_csv:   # Write mode
        wr = csv.writer(new_csv, lineterminator='\n')
        for row in range(sheet.nrows):   # Write by rows
            wr.writerow(sheet.row_values(row))

    new_csv.close()


def convertColumn(df, names, newtype):   # Determine the data type for each column
  for name in names: 
     df = df.withColumn(name, df[name].cast(newtype))
  return df 


             # ----------------------------- Main function -------------------------------------  

             
if __name__ == "__main__":   
    sc = SparkContext()   # Set up pyspark environment
    sqlContext = SQLContext(sc)
    
#   csv_from_excel()  # No need to rin it each time!
    
    rdd=sc.textFile("./PQ_result/THD.csv") 
    rdd = rdd.map(lambda x: x.split(","))   # Split each row, lambda是个临时函数
    rdd = rdd.zipWithIndex().filter(lambda (row,index): index > 0).keys()   # Remove 1st line of rdd which is irrelavant
    
    df = rdd.map(lambda line: Row(Hours=line[0],    
                                  THD18th=line[1], 
                                  THD42th=line[2])).toDF()   # Convert the RDD to a dataframe
    columns = ['Hours', 'THD18th', 'THD42th']
    df = convertColumn(df, columns, types.FloatType())

    
    df.describe().show()   # Show overall statistical features
 
# Data pre-processing  
    
    df = df.withColumn("THD18th", functions.col("THD18th") * 100) \
        .withColumn("THD42th", functions.col("THD42th") * 100)   # Convert to percentage values, 注意换行符后面不能有空格，否则报错！
   
    df = df.select("Hours", "THD18th")   # Select 'Hours' & 'THD18th' for analysis
    df.show(24)
    
    input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))   # Create new dataframe with lables
    labeled_df = sqlContext.createDataFrame(input_data, ["label", "features"])
    labeled_df.show(24)
    
    standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")   # Re-scaling
    scaler = standardScaler.fit(labeled_df)
    scaled_df = scaler.transform(labeled_df)
    scaled_df.show(24)
    
    train_data, test_data = scaled_df.randomSplit([0.7,0.3])   # Randomly choose 30% as test data
    test_data.show(24)
    
    
    #lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)   # Train models
    lr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)
    linearModel = lr.fit(train_data)
    
    predicted = linearModel.transform(test_data)   # Prediction
    predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
    labels = predicted.select("label").rdd.map(lambda x: x[0])

    predictionAndLabel = predictions.zip(labels).collect()   # Zip predicted values with lables
    
    print(linearModel.coefficients)   # Evaluate the model
    print(linearModel.intercept)
    #print(linearModel.summary.rootMeanSquaredError)
    
    sc.stop()
