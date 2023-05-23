package com.virtualpairprogrammers.ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

public class VppChapterViews {
    public static void main(String args[]) {

        System.setProperty("hadoop.home.dir", "c:/users/saura/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.ERROR);
// apply VM options in run configuration
        //--add-exports java.base/sun.nio.ch=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED --add-opens java.base/java.lang.invoke=ALL-UNNAMED
        SparkSession spark = SparkSession.builder().appName("VPPChapterViews").master("local[*]").config("spark.sql.warehouse.dir", "file:///c:/tmp/").getOrCreate();
        Dataset<Row> dataset = spark.read().option("header", true).option("inferSchema", true)
                .csv("project/src/main/resources/ml/vppChapterViews/part-r-00000-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "project/src/main/resources/ml/vppChapterViews/part-r-00001-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "project/src/main/resources/ml/vppChapterViews/part-r-00002-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "project/src/main/resources/ml/vppChapterViews/part-r-00003-d55d9fed-7427-4d23-aa42-495275510f78.csv");


        //filter records where is cancelled true
        dataset=dataset.filter("is_cancelled=false").drop("observation_date","is_cancelled");
        dataset=dataset
                .withColumn("firstSub",when(col("firstSub").isNull(),0).otherwise(col("firstSub")))
                .withColumn("all_time_views",when(col("all_time_views").isNull(),0).otherwise(col("all_time_views")))
                .withColumn("last_month_views",when(col("last_month_views").isNull(),0).otherwise(col("last_month_views")))
                .withColumn("next_month_views",when(col("next_month_views").isNull(),0).otherwise(col("next_month_views")));
        dataset=dataset.withColumnRenamed("next_month_views","label");


        StringIndexer payMethodIndexer = new StringIndexer();
        dataset=payMethodIndexer.setInputCol("payment_method_type")
                        .setOutputCol("payIndex")
                        .fit(dataset)
                        .transform(dataset);
        StringIndexer countryIndexer = new StringIndexer();
        dataset=countryIndexer.setInputCol("country")
                .setOutputCol("countryIndex")
                .fit(dataset)
                .transform(dataset);
        StringIndexer periodIndexer = new StringIndexer();
        dataset=periodIndexer.setInputCol("rebill_period_in_months")
                .setOutputCol("periodIndex")
                .fit(dataset)
                .transform(dataset);


        OneHotEncoder encoder = new OneHotEncoder();
        dataset=encoder.setInputCols(new String[] {"payIndex","countryIndex","periodIndex"})
                .setOutputCols(new String[] {"payVector","countryVector","periodVector"})
                        .fit(dataset).transform(dataset);

        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(new String[] {"firstSub","age","all_time_views","last_month_views","payVector","countryVector","periodVector"})
                .setOutputCol("features");
        //transform and then do label features for data set
        //Use it for training and test
        Dataset<Row> inputData=vectorAssembler.transform(dataset).select("label","features");
        inputData.show();
        //use inputData
        Dataset<Row>[] dataSplits=inputData.randomSplit(new double[]{0.9,0.1});
        Dataset<Row> trainingAndTestData=dataSplits[0];
        Dataset<Row> holdOutData=dataSplits[1];

        LinearRegression linearRegression = new LinearRegression();

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

        ParamMap[] paramMap = paramGridBuilder.addGrid(linearRegression.regParam(), new double[] {0.01,0.1,0.3,0.5,0.7,1})
                .addGrid(linearRegression.elasticNetParam(), new double[] {0,0.5,1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.9);

        TrainValidationSplitModel model=trainValidationSplit.fit(trainingAndTestData);

        LinearRegressionModel lrModel=(LinearRegressionModel) model.bestModel();

        System.out.println("Model has intercept: "+ lrModel.intercept()+ " and coefficients: "+ lrModel.coefficients());
        //Get the values of regParam and netParam that were selected
        System.out.println("reg Param: "+lrModel.getRegParam()+" net Param: "+lrModel.getElasticNetParam());

        //model.transform(testData).show();
        System.out.println("Training data r2:"+ lrModel.summary().r2()+" Training data rmse: "+lrModel.summary().rootMeanSquaredError());
        System.out.println(" Test data r2:"+
                lrModel.evaluate(holdOutData).r2()+" Test data rmse:"+lrModel.evaluate(holdOutData).rootMeanSquaredError());




  /*      dataset=dataset.withColumn("sqft_above_percentage",(col("sqft_above").divide(col("sqft_living"))))
                .withColumnRenamed("price","label");
        dataset.show();
        dataset.printSchema();
        dataset.describe().show();

        Dataset<Row>[] dataSplits=dataset.randomSplit(new double[]{0.8,0.2});
        Dataset<Row> trainingAndTestData=dataSplits[0];
        Dataset<Row> holdOutData=dataSplits[1];

        StringIndexer conditionIndexer = new StringIndexer();
        conditionIndexer.setInputCol("condition");
        conditionIndexer.setOutputCol("conditionIndex");

        StringIndexer gradeIndexer = new StringIndexer();
        gradeIndexer.setInputCol("grade");
        gradeIndexer.setOutputCol("gradeIndex");

        StringIndexer zipcodeIndexer = new StringIndexer();
        zipcodeIndexer.setInputCol("zipcode");
        zipcodeIndexer.setOutputCol("zipcodeIndex");

        OneHotEncoder encoder = new OneHotEncoder();
        encoder.setInputCols(new String[] {"conditionIndex","gradeIndex","zipcodeIndex"});
        encoder.setOutputCols(new String[] {"conditionVector","gradeVector","zipcodeVector"});

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] {"bedrooms","bathrooms","sqft_living","sqft_above_percentage","floors","conditionVector","gradeVector","zipcodeVector","waterfront"})
                .setOutputCol("features");

        VectorAssembler vectorAssembler2 = new VectorAssembler()
                .setInputCols(new String[] {"bedrooms","sqft_living","sqft_above_percentage","floors","conditionVector","gradeVector","zipcodeVector","waterfront"})
                .setOutputCol("features");

        LinearRegression linearRegression = new LinearRegression();

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

        ParamMap[] paramMap = paramGridBuilder.addGrid(linearRegression.regParam(), new double[] {0.01,0.1,0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[] {0,0.5,1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.8);

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[] {conditionIndexer,gradeIndexer,zipcodeIndexer,encoder,vectorAssembler,trainValidationSplit});
        PipelineModel pipelineModel = pipeline.fit(trainingAndTestData);
        TrainValidationSplitModel model = (TrainValidationSplitModel)pipelineModel.stages()[5];
        LinearRegressionModel lrModel = (LinearRegressionModel)	model.bestModel();

        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutData);
        holdOutResults.show();
        holdOutResults = holdOutResults.drop("prediction");
        System.out.println("Model has intercept: "+ lrModel.intercept()+ " and coefficients: "+ lrModel.coefficients());
        //Get the values of regParam and netParam that were selected
        System.out.println("reg Param: "+lrModel.getRegParam()+" net Param: "+lrModel.getElasticNetParam());

        //model.transform(testData).show();
        System.out.println("Training data r2:"+ lrModel.summary().r2()+" Training data rmse: "+lrModel.summary().rootMeanSquaredError());
        System.out.println(" Test data r2:"+
                lrModel.evaluate(holdOutResults).r2()+" Test data rmse:"+lrModel.evaluate(holdOutResults).rootMeanSquaredError());
  */      spark.close();

    }

}
