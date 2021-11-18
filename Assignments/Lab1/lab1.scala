// Databricks notebook source
val housing = spark.read.options(Map("inferSchema"->"true","delimiter"->",","header"->"true"))
  .csv("dbfs:/FileStore/shared_uploads/emilstah@kth.se/housing-1.csv")



// COMMAND ----------

housing.printSchema()
housing.count()

// COMMAND ----------

housing.show(5)
housing.where("population > 10000").show()


// COMMAND ----------

housing.describe("housing_median_age", "total_rooms", "median_house_value", "population").show()

import org.apache.spark.sql.functions._

housing.select(max("housing_median_age"), min("total_rooms"), avg("median_house_value")).show()

// COMMAND ----------

housing.groupBy("ocean_proximity").count().sort(col("ocean_proximity").desc).show()
housing.groupBy("ocean_proximity").avg("median_house_value").withColumnRenamed("avg(median_house_value)", "avg_value").show()
housing.createOrReplaceTempView("df")
spark.sql("SELECT ocean_proximity, AVG(median_house_value) AS avg_value FROM df GROUP BY ocean_proximity").show()


// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val va = new VectorAssembler().setInputCols(Array("housing_median_age", "total_rooms", "median_house_value", "population")).setOutputCol("features")

val housingAttrs = va.transform(housing)

housingAttrs.show(5)

// COMMAND ----------

val housingCol1 = housing.withColumn("rooms_per_household", col("total_rooms") / col("households"))
val housingCol2 = housingCol1.withColumn("bedrooms_per_room", col("total_bedrooms") / col("total_rooms"))
val housingExtra = housingCol2.withColumn("population_per_household", col("population") / col("households"))

housingExtra.select("rooms_per_household", "bedrooms_per_room", "population_per_household").show(5)

// COMMAND ----------

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

import org.apache.spark.ml.feature.Imputer

val imputer = new Imputer().setStrategy("median").setInputCols(Array("total_bedrooms", "bedrooms_per_room")).setOutputCols(Array("total_bedrooms", "bedrooms_per_room"))
                       
val imputedHousing = imputer.fit(renamedHousing).transform(renamedHousing)

imputedHousing.select("total_bedrooms", "bedrooms_per_room").show(5)

// COMMAND ----------

import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

val va = new VectorAssembler().setInputCols(colNum).setOutputCol("vector_features")
val featuredHousing = va.transform(imputedHousing)

val scaler = new StandardScaler().setInputCol("vector_features").setOutputCol("scaler_features")
val scaledHousing = scaler.fit(featuredHousing).transform(featuredHousing)

scaledHousing.show(5)

// COMMAND ----------

renamedHousing.select("ocean_proximity").distinct().show()


// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer

val indexer = new StringIndexer().setInputCol(colCat).setOutputCol("ocean_proximity_numbers")
val idxHousing = indexer.fit(renamedHousing).transform(renamedHousing)

idxHousing.show(5)

// COMMAND ----------

indexer.fit(renamedHousing).labels

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.feature.OneHotEncoder

val encoder = new OneHotEncoder().setInputCol("ocean_proximity_numbers").setOutputCol("ocean_proximity_vectors")

val ohHousing = encoder.fit(idxHousing).transform(idxHousing)

ohHousing.show(5)

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}

val numPipeline = Pipeline.setStages(Array(imputer, va, scaler))
val catPipeline = Pipeline.setStages(Array(indexer, encoder))
val numPipeline = PipelineStage.setStages(Array(imputer, va, scaler))


val pipeline = new Pipeline().setStages(Array(numPipeline, catPipeline))
val newHousing = pipeline.fit(renamedHousing).transform(renamedHousing)

newHousing.show(5)

// COMMAND ----------


