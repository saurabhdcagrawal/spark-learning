package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class MainSparkRDD {

    public static void main(String args[]){
//Streams take anonymous functions
        Logger.getLogger("org.apache").setLevel(Level.WARN);
        SparkConf conf= new SparkConf().setAppName("startingSpark").setMaster("local[*]");
        JavaSparkContext sc= new JavaSparkContext(conf);
        /*List<Double> inputDataOne= new ArrayList<Double>();
        inputDataOne.add(10.45);
        inputDataOne.add(1.45);
        inputDataOne.add(3.45);
        //immutable
        JavaRDD<Double> myRDD= sc.parallelize(inputDataOne);
        //Reduction
        Double resultOne=myRDD.reduce((a,b)->a+b);
        System.out.println(resultOne);*/
//Mapping

        List<Integer> inputDataTwo= new ArrayList<Integer>();

        inputDataTwo.add(10);
        inputDataTwo.add(25);
        inputDataTwo.add(16);
        JavaRDD<Integer> newRDD= sc.parallelize(inputDataTwo);

        JavaRDD<Double> resultTwo=newRDD.map((a->Math.sqrt(a)));
        System.out.println("Count "+resultTwo.count());
        //resultTwo.foreach(a->System.out.println(a));
        //resultTwo.foreach(System.out::println);
        //multiple cpu's ... serialization error //forEach for collection and foreach for RDD
        resultTwo.collect().forEach(System.out::println);

        JavaRDD<Long> countNewRDD=newRDD.map(a->1L);
        Long count= countNewRDD.reduce((a,b)->a+b);
        System.out.println("Count using map reduce "+count);
//Tuples //to output a collection of number and its sqrt.. RDD can be object oriented
        JavaRDD<Tuple2<Integer,Double>> tupleRDD=newRDD.map(a->new Tuple2<>(a,Math.sqrt(a)));

        sc.close();

        SparkSession spark= SparkSession.builder().appName("testingSQL").master("local[*]").config("spark.sql.warehouse.dir","file:///c:/users/saura/tmp/").getOrCreate();
        Dataset<Row> dataset=spark.read().option("header",true).csv("src/main/resources/exams/students.csv");
        dataset.show();

    }
}
