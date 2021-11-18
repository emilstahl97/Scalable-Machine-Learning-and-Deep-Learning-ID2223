// Databricks notebook source
val housing = spark.read.options(Map("inferSchema"->"true","delimiter"->",","header"->"true"))
  .csv("dbfs:/FileStore/shared_uploads/emilstah@kth.se/housing-1.csv")



// COMMAND ----------

housing.printSchema()
housing.count()

// COMMAND ----------

// Look at the data
housing.show(5)
housing.where("population > 10000").show()


// COMMAND ----------

/* 2.3. Statistical summary

Print a summary of the table statistics for the attributes housing_median_age, total_rooms, median_house_value, and population. You can use the describe command.
*/

housing.describe("housing_median_age", "total_rooms", "median_house_value", "population").show()

import org.apache.spark.sql.functions._

housing.select(max("housing_median_age"), min("total_rooms"), avg("median_house_value")).show()

// COMMAND ----------

/*
2.4. Breakdown the data by categorical data

Print the number of houses in different areas (ocean_proximity), and sort them in descending order.
*/

housing.groupBy("ocean_proximity").count().sort(col("ocean_proximity").desc).show()
housing.groupBy("ocean_proximity").avg("median_house_value").withColumnRenamed("avg(median_house_value)", "avg_value").show()
housing.createOrReplaceTempView("df")
spark.sql("SELECT ocean_proximity, AVG(median_house_value) AS avg_value FROM df GROUP BY ocean_proximity").show()


// COMMAND ----------

/*
2.5. Correlation among attributes

Print the correlation among the attributes housing_median_age, total_rooms, median_house_value, and population. To do so, first you need to put these attributes into one vector. Then, compute the standard correlation coefficient (Pearson) between every pair of attributes in this new vector. To make a vector of these attributes, you can use the VectorAssembler Transformer.
*/

import org.apache.spark.ml.feature.VectorAssembler

val va = new VectorAssembler().setInputCols(Array("housing_median_age", "total_rooms", "median_house_value", "population")).setOutputCol("features")

val housingAttrs = va.transform(housing)

housingAttrs.show(5)

// COMMAND ----------

/*
2.6. Combine and make new attributes

Now, let's try out various attribute combinations. In the given dataset, the total number of rooms in a block is not very useful, if we don't know how many households there are. What we really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very useful, and we want to compare it to the number of rooms. And the population per household seems like also an interesting attribute combination to look at. To do so, add the three new columns to the dataset as below. We will call the new dataset the housingExtra.

rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households
*/

val housingCol1 = housing.withColumn("rooms_per_household", col("total_rooms") / col("households"))
val housingCol2 = housingCol1.withColumn("bedrooms_per_room", col("total_bedrooms") / col("total_rooms"))
val housingExtra = housingCol2.withColumn("population_per_household", col("population") / col("households"))

housingExtra.select("rooms_per_household", "bedrooms_per_room", "population_per_household").show(5)


// COMMAND ----------

/*

3. Prepare the data for Machine Learning algorithms

Before going through the Machine Learning steps, let's first rename the label column from median_house_value to label.
*/

val renamedHousing = housingExtra.withColumnRenamed("median_house_value", "label")

renamedHousing.show(5)

// COMMAND ----------

// label columns
val colLabel = "label"

// categorical columns
val colCat = "ocean_proximity"

// numerical columns
val colNum = renamedHousing.columns.filter(_ != colLabel).filter(_ != colCat)

// COMMAND ----------

/*
3.1. Prepare continues attributes

Data cleaning

Most Machine Learning algorithms cannot work with missing features, so we should take care of them. As a first step, let's find the columns with missing values in the numerical attributes. To do so, we can print the number of missing values of each continues attributes, listed in colNum.
*/

import org.apache.spark.ml.feature.Imputer

val imputer = new Imputer().setStrategy("median").setInputCols(Array("total_bedrooms", "bedrooms_per_room")).setOutputCols(Array("total_bedrooms", "bedrooms_per_room"))
                       
val imputedHousing = imputer.fit(renamedHousing).transform(renamedHousing)

imputedHousing.select("total_bedrooms", "bedrooms_per_room").show(5)

// COMMAND ----------

// 3.1 Scaling

import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

val va = new VectorAssembler().setInputCols(colNum).setOutputCol("vector_features")
val featuredHousing = va.transform(imputedHousing)

val scaler = new StandardScaler().setInputCol("vector_features").setOutputCol("scaler_features")
val scaledHousing = scaler.fit(featuredHousing).transform(featuredHousing)

scaledHousing.show(5)

// COMMAND ----------

// 3.2 Prepare categorical attributes

renamedHousing.select("ocean_proximity").distinct().show()


// COMMAND ----------

// 3.2 String indexer

import org.apache.spark.ml.feature.StringIndexer

val indexer = new StringIndexer().setInputCol(colCat).setOutputCol("ocean_proximity_numbers")
val idxHousing = indexer.fit(renamedHousing).transform(renamedHousing)

idxHousing.show(5)

// COMMAND ----------

indexer.fit(renamedHousing).labels

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code
// 3.2 One-hot encoding

import org.apache.spark.ml.feature.OneHotEncoder

val encoder = new OneHotEncoder().setInputCol("ocean_proximity_numbers").setOutputCol("ocean_proximity_vectors")

val ohHousing = encoder.fit(idxHousing).transform(idxHousing)

ohHousing.show(5)

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code
// 4. Pipeline

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import scala.collection.mutable

var numStage = new mutable.ArrayBuffer[PipelineStage]()
numStage += imputer
numStage += va
numStage += scaler

var catStage = new mutable.ArrayBuffer[PipelineStage]()
catStage += indexer
catStage += encoder

val numPipeline = new Pipeline().setStages(numStage.toArray)
val catPipeline = new Pipeline().setStages(catStage.toArray)

val pipeline = new Pipeline().setStages(Array(numPipeline, catPipeline))
val newHousing = pipeline.fit(renamedHousing).transform(renamedHousing)

newHousing.show(5)

// COMMAND ----------

val va2 = new VectorAssembler().setInputCols(Array("scaler_features", "ocean_proximity_vectors"))
.setOutputCol("features")
val dataset = va2.transform(newHousing).select("features", "label")

dataset.show(5, false)


// COMMAND ----------

// 5. Make a model

val Array(trainSet, testSet) = dataset.randomSplit(Array(0.8, 0.2))

print("Training set:\n")
trainSet.show(5)
print("Test set:\n")
testSet.show(5)


// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code
// 5.1. Linear regression model

import org.apache.spark.ml.regression.LinearRegression

// train the model
val lr = new LinearRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)
val lrModel = lr.fit(trainSet)
val trainingSummary = lrModel.summary

// someone fix printout 
println(s"Coefficients: $lrModel.coefficients. Intercept: $lrModel.intercept")
println(s"RMSE: $trainingSummary.rootMeanSquaredError")

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

// make predictions on the test data
val predictions = lrModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("label")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

/*
5.2. Decision tree regression

Repeat what you have done on Regression Model to build a Decision Tree model. Use the DecisionTreeRegressor to make a model and then measure its RMSE on the test dataset.
*/

import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val dt = new DecisionTreeRegressor()

// train the model
val dtModel = dt.fit(trainSet)

// make predictions on the test data
val predictions = dtModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("label")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

/*
5.3. Random forest regression

Let's try the test error on a Random Forest Model. You can use the RandomForestRegressor to make a Random Forest model.
*/

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val rf = new RandomForestRegressor()

// train the model
val rfModel = rf.fit(trainSet)

// make predictions on the test data
val predictions = rfModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("label")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

/*
5.4. Gradient-boosted tree regression

Finally, we want to build a Gradient-boosted Tree Regression model and test the RMSE of the test data. Use the GBTRegressor to build the model.
*/

import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val gb = new GBTRegressor()

// train the model
val gbModel = gb.fit(trainSet)

// make predictions on the test data
val predictions = gbModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("label")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

