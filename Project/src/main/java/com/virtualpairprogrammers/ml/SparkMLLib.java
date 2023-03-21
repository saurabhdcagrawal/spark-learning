package com.virtualpairprogrammers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SparkMLLib {

    public static void main(String args[]){
        System.setProperty("hadoop.home.dir", "c:/users/saura/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);
// apply VM options in run configuration
        //--add-exports java.base/sun.nio.ch=ALL-UNNAMED
         SparkSession spark= SparkSession.builder().appName("testingSQL").master("local[*]").config("spark.sql.warehouse.dir","file:///c:/tmp/").getOrCreate();
        Dataset<Row> dataset=spark.read().option("header",true).option("inferSchema",true)
                .csv("project/src/main/resources/ml/GymCompetition.csv");
        //dataset.printSchema();
        //dataset.show();

        //data indexer--->non numeric model as continuos values
        StringIndexer genderIndexer= new StringIndexer();
        genderIndexer.setInputCol("Gender");
        genderIndexer.setOutputCol("GenderIndex");
        dataset=genderIndexer.fit(dataset).transform(dataset);
        dataset.show();

        //encoder
        OneHotEncoder genderEncoder= new OneHotEncoder();
        genderEncoder.setInputCols(new String[]{"GenderIndex"});
        genderEncoder.setOutputCols(new String[]{"GenderVector"});
        dataset=genderEncoder.fit(dataset).transform(dataset);
        dataset.show();


        //vector assembler will assemble features as vectors
        VectorAssembler vectorAssembler= new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"Age","Height","Weight"});
        vectorAssembler.setOutputCol("features");
        Dataset<Row> datasetWithFeatures= vectorAssembler.transform(dataset);
        datasetWithFeatures.show();
        Dataset<Row> modelInputData=datasetWithFeatures.select("NoOfReps","features").withColumnRenamed("NoOfReps","label");
        modelInputData.show();

        LinearRegression linearRegression= new LinearRegression();
        LinearRegressionModel model =linearRegression.fit(modelInputData);
        System.out.println("Model has intercept "+ model.intercept()+ " and coefficients "+ model.coefficients());
        model.transform(modelInputData).show();

        //comeback and finish pipelines, case study
        spark.close();
    }
}
