package com.virtualpairprogrammers.ml;

import breeze.util.DenseIntIndex;
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
import static org.apache.spark.sql.functions.col;

public class HousePricing {
    public static void main(String args[]) {

        System.setProperty("hadoop.home.dir", "c:/users/saura/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.ERROR);
// apply VM options in run configuration
        //--add-exports java.base/sun.nio.ch=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED --add-opens java.base/java.lang.invoke=ALL-UNNAMED
        SparkSession spark = SparkSession.builder().appName("testingSQL").master("local[*]").config("spark.sql.warehouse.dir", "file:///c:/tmp/").getOrCreate();
        Dataset<Row> dataset = spark.read().option("header", true).option("inferSchema", true)
                .csv("project/src/main/resources/ml/kc_house_data.csv");


        dataset=dataset.withColumn("sqft_above_percentage",(col("sqft_above").divide(col("sqft_living"))))
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



        /* Old ways
        //Adding percentage of sqft_above reasoning comes later
        dataset.withColumn("sqft_above_percentage",col("sqft_above").divide(col("sqft_living")));

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

        /*Linear Regression Parameters
        Dataset<Row>[] dataSplits=modelInputData.randomSplit(new double[]{0.8,0.2});
        Dataset<Row> trainingandtestData=dataSplits[0];
        Dataset<Row> holdoutData=dataSplits[1];

        ParamGridBuilder paramGridBuilder= new ParamGridBuilder();
        ParamMap[]  paramMap=paramGridBuilder.addGrid(linearRegression.regParam(),new double[]{0.01,0.1,0.5})
                             .addGrid(linearRegression.elasticNetParam(),new double[]{0,0.5,1})
                             .build();
        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setTrainRatio(0.8);
        TrainValidationSplitModel model=trainValidationSplit.fit(trainingandtestData);
        //Automatically choose the best model from our grid
        LinearRegressionModel lrModel=(LinearRegressionModel) model.bestModel();
        //Get the values of regParam and netParam that were selected
        System.out.println("reg Param"+lrModel.getRegParam()+" net Param "+lrModel.getElasticNetParam());

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
        } */
        //data preparation
        //(1) Replace nulls with appropriate value (Spark doesnt allow any features to have null value)
        //(2) Data missing..remove that record (system failure or data loss)
        //(3) Records are Erroneous-- eg date of birth in future, price is negative --remove
        //(4) non numeric data to data... gender-> split data as m, f, Unknown and use boolean values(spark
        //can automate)
        spark.close();

    }

    }
