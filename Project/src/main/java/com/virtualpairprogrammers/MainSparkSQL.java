package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;
import scala.Function1;

public class MainSparkSQL {

    public static void main(String args[]){
        System.setProperty("hadoop.home.dir", "c:/users/saura/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);


        SparkSession spark= SparkSession.builder().appName("testingSQL").master("local[*]").config("spark.sql.warehouse.dir","file:///c:/tmp/").getOrCreate();
        //this line is not reading in data set //this is not happening in memory//under the hood an execution plan is built up
        //only when we come to an operation/action spark wll execute the execution plan //only then data is going to be read in
        //in a real cluster, only then worker nodes will start do their thing
        Dataset<Row> dataset=spark.read().option("header",true).csv("project/src/main/resources/exams/students.csv");
        dataset.show();
        System.out.println("Total number of rows "+dataset.count());

        Row firstRow= dataset.first();
        //firstrow.get is not string but object because if used data source other than csv, for eg jdbc.. than different set of data
        String subj=firstRow.get(2).toString();
        System.out.println("Subject using get "+subj);

// getAs
        subj=firstRow.getAs("subject").toString();
        System.out.println("Subject using getAs is "+subj);

        String year= firstRow.getAs("year").toString();
        System.out.println("Year using getAs is "+year);
//filters using expressions
        //immutable// therefore get new datasets every time
        Dataset<Row> modernArtResultsUsingFilterExp= dataset.filter("subject='Modern Art' AND year>=2007 ");
        modernArtResultsUsingFilterExp.show();
//filters using lambdas
        Dataset<Row> modernArtResultsUsingLambdas=dataset.filter((FilterFunction<Row>) row -> row.getAs("subject").equals("Modern Art") && Integer.parseInt(row.getAs("year"))>=2007);
        modernArtResultsUsingLambdas.show();

        //filter using columns
        Column subjectCol =dataset.col("subject");
        Column yearCol =dataset.col("year");

// good syntax for optionally add columns for filter
        Dataset<Row> modernArtResultsUsingColumns=dataset.filter(subjectCol.equalTo("Modern Art").and(yearCol.geq(2007)));
// this way is like DSL ..compiles as regular java but reads as SQL
        //static sql.spark syntax
        System.out.println("modernArtResultsUsingColumnsSparkStaticlib");
        Dataset<Row> modernArtResultsUsingColumnsSparkStaticlib=dataset.filter(col("subject").equalTo("Modern Art").and(col("year").geq(2007)).and(col("score").geq(80)));
        modernArtResultsUsingColumnsSparkStaticlib.show();
   //  Spark Temporary view..create a temporary view first
        System.out.println("Spark temp view");

        dataset.createOrReplaceTempView("my_students_table");
        Dataset<Row> frenchResultsUsingView=spark.sql("select score,year from my_students_table where subject='French'");
        frenchResultsUsingView.show();
        spark.close();

    }
}
