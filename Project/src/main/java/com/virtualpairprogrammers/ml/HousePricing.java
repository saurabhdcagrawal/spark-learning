package com.virtualpairprogrammers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePricing {
    public static void main(String args[]) {
        System.setProperty("hadoop.home.dir", "c:/users/saura/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);
// apply VM options in run configuration
        //--add-exports java.base/sun.nio.ch=ALL-UNNAMED
        SparkSession spark = SparkSession.builder().appName("testingSQL").master("local[*]").config("spark.sql.warehouse.dir", "file:///c:/tmp/").getOrCreate();
        Dataset<Row> dataset = spark.read().option("header", true).option("inferSchema", true)
                .csv("project/src/main/resources/ml/kc_house_data.csv");
       // dataset.show();
        // dataset.printSchema();
        dataset.describe().show();
        //VectorAssembler vectorAssembler= new VectorAssembler().setInputCols(new String[]{"bedrooms","bathrooms","sqft_living"}).setOutputCol("features");
        //keep playing with features for better rmse// can it be automated?
        VectorAssembler vectorAssembler= new VectorAssembler().setInputCols(new String[]{"bedrooms","bathrooms","sqft_living","sqft_lot","floors","grade"}).setOutputCol("features");

        Dataset<Row> datasetWithFeatures= vectorAssembler.transform(dataset);
        //datasetWithFeatures.show();
        Dataset<Row> modelInputData=datasetWithFeatures.select("price","features").withColumnRenamed("price","label");
        modelInputData.show();

        Dataset<Row>[] trainingAndTestData=modelInputData.randomSplit(new double[]{0.8,0.2});
        Dataset<Row> trainingData=trainingAndTestData[0];
        Dataset<Row> testData=trainingAndTestData[1];
        LinearRegression linearRegression= new LinearRegression();
        LinearRegressionModel model =linearRegression.fit(trainingData);
        System.out.println("Model has intercept "+ model.intercept()+ " and coefficients "+ model.coefficients());
        //model.transform(testData).show();
        System.out.println("Training data r2"+ model.summary().r2()+" Training data rmse "+model.summary().rootMeanSquaredError()+" Test data r2"+
                model.evaluate(testData).r2()+" Test data rmse "+model.evaluate(testData).rootMeanSquaredError());


        //What does std dev mean//read normal distribution
        //1)from mean-std_dev value to mean+std_dev value, we have 60% houses
        //min and max ideally should be far..but if they fall in this range ..we have far more houses on right than on left

        //improve measures of accuracy
        //1. Eliminate dependent variables ..eg sales tax on housing price .. remove variables that are completely dependent on values we are trying to predict
        //2 . Does each variable have sufficiently wide range of values?

        //3)Are any variables good potential predictors?
        //4) Are there any duplicate variables?(variables correlated with each other)
        //features selection
        //exclude boolean values
        //grade is category(categorical data)


        //coorrelation between price and sqft living

        dataset=dataset.drop("id","date","waterfront","view","condition","grade","yr_renovated","zipcode","lat","long");
        for(String col: dataset.columns())
            System.out.println("The correlation between price and "+col+" is "+ dataset.stat().corr("price",col));
        //drop based on correlation
        dataset=dataset.drop("sqft_loop","sqft_loft15","yr_built","sqft_living15");
        //take data and put in a table and correlation table
        //exclude things close to 1 and close to -1
        for(String col1: dataset.columns()){
            for(String col2: dataset.columns()){
                System.out.println("The correlation between "+col1 +" and "+col2 +" is " + dataset.stat().corr(col1,col2));
            }
        }
        //data preparation
        //(1) Replace nulls with appropriate value (Spark doesnt allow any features to have null value)
        //(2) Data missing..remove that record (system failure or data loss)
        //(3) Records are Erroneous-- eg date of birth in future, price is negative --remove
        //(4) non numeric data to data... gender-> split data as m, f, Unknown and use boolean values(spark
        //can automate)
        spark.close();

        }

    }
