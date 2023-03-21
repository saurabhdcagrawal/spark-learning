package com.virtualpairprogrammers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
//Group data based on similar features
public class GymCompetitiorsClustering {
    public static void main(String args[]) {
        System.setProperty("hadoop.home.dir", "c:/users/saura/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);
// apply VM options in run configuration
        //--add-exports java.base/sun.nio.ch=ALL-UNNAMED
        SparkSession spark = SparkSession.builder().appName("testingSQL").master("local[*]").config("spark.sql.warehouse.dir", "file:///c:/tmp/").getOrCreate();
        Dataset<Row> dataset = spark.read().option("header", true).option("inferSchema", true)
                .csv("project/src/main/resources/ml/GymCompetition.csv");
        //dataset.printSchema();
        //dataset.show();

        //data indexer--->non numeric model as continuos values
        StringIndexer genderIndexer = new StringIndexer();
        genderIndexer.setInputCol("Gender");
        genderIndexer.setOutputCol("GenderIndex");
        dataset = genderIndexer.fit(dataset).transform(dataset);
        dataset.show();

        //encoder
        OneHotEncoder genderEncoder = new OneHotEncoder();
        genderEncoder.setInputCols(new String[]{"GenderIndex"});
        genderEncoder.setOutputCols(new String[]{"GenderVector"});
        dataset = genderEncoder.fit(dataset).transform(dataset);
        dataset.show();
        }
    }
