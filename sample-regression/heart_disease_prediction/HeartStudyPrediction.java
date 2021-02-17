package ai.certifai.training.regression.heart_disease_prediction;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.filter.InvalidNumColumns;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class HeartStudyPrediction {
    public static Logger log = LoggerFactory.getLogger(HeartStudyPrediction.class);

    public static final long seed = 123456;
    private static int label = 15;
    private static int output = 2;
    private static double lr = 0.0001;
    private static int epoch = 50;


    public static void main(String[] args) throws Exception, IOException {

        String inputFile = new ClassPathResource("framingham/framingham.csv").getFile().getAbsolutePath();
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(new File(inputFile)));

        //schema
        Schema schema = new Schema.Builder()
                .addColumnInteger("male")
                .addColumnInteger("age")
                .addColumnInteger("education")
                .addColumnInteger("currentSmoker")
                .addColumnInteger("cigsPerDay")
                .addColumnInteger("BPMeds")
                .addColumnInteger("prevalentStroke")
                .addColumnInteger("prevalentHyp")
                .addColumnInteger("diabetes")
                .addColumnInteger("totChol")
                .addColumnDouble("sysBP")
                .addColumnDouble("diaBP")
                .addColumnDouble("BMI") //todouble
                .addColumnDouble("heartRate") //todouble
                .addColumnInteger("glucose")
                .addColumnInteger("TenYearCHD")
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                //filter the NA value
                .filter(new FilterInvalidValues())

                //one-hot classification output
//                .integerToOneHot("TenYearCHD", 0, 1)
//                .categoricalToOneHot("TenYearCHD")
                .build();

        List<List<Writable>> data = new ArrayList<>();
        while (rr.hasNext()) {
            data.add(rr.next());
        }

        List<List<Writable>> transformed = LocalTransformExecutor.execute(data, tp);
        System.out.println(tp.getInitialSchema());
        System.out.println(tp.getFinalSchema());
        System.out.println(data.size());
        System.out.println(transformed.size());

        RecordReader crr = new CollectionRecordReader(transformed);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(crr, transformed.size(), 15, output);

        DataSet dataSet = iter.next();
        dataSet.shuffle();

        SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.7);
        DataSet train = splitTestAndTrain.getTrain();
        DataSet test = splitTestAndTrain.getTest();

        INDArray features = train.getFeatures();
        System.out.println("\nFeatures shape: " + features.shapeInfoToString());

        INDArray labels = train.getLabels();
        System.out.println("Labels shape: " + labels.shapeInfoToString() + "\n");

        // Assigning dataset iterator for training purpose
        ViewIterator trainIter = new ViewIterator(train, train.numExamples());
        ViewIterator testIter = new ViewIterator(test, test.numExamples());

        System.out.println("input: " + trainIter.inputColumns());
        System.out.println("output: " + trainIter.totalOutcomes());
        System.out.println(trainIter.next());


        // Data normalization
        DataNormalization scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        //train model
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Nesterovs(lr, 0.9))
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(trainIter.inputColumns())
                            .nOut(64)
                            .activation(Activation.IDENTITY)
                            .build())
                    .layer(1, new DenseLayer.Builder()
                            .activation(Activation.RELU)
                            .nOut(128)
                            .build())
                    .layer(2, new DenseLayer.Builder()
                            .activation(Activation.RELU)
                            .nOut(128)
                            .build())
                    .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                            .activation(Activation.SIGMOID)
                            .nOut(output)
                            .build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            System.out.println(model.summary());

            //performing early stopping criteria
//            EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
//                    .epochTerminationConditions(new MaxEpochsTerminationCondition(300))
//                    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
//                    .scoreCalculator(new DataSetLossCalculator(testIter, true))
//                    .evaluateEveryNEpochs(1)
//                    .build();
//
//
//            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,trainIter);
//            EarlyStoppingResult result = trainer.fit();
//
//            MultiLayerNetwork model = (MultiLayerNetwork) result.getBestModel();

            // Initialize UI server for visualization model performance
            log.info("****************************************** UI SERVER **********************************************");
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            model.setListeners(
                    new ScoreIterationListener(10),
                    new StatsListener(statsStorage));

            for(int i=0; i<epoch; i++) {
                trainIter.reset();
                model.fit(trainIter);
            }

            testIter.reset();

            //evaluation
//            System.out.println("\n===Train Evaluation===");
//            Evaluation evalTrain = model.evaluate(trainIter);
//            System.out.println(evalTrain.stats());

//            System.out.println("\n====Test Evaluation====");
//            Evaluation evalTest = model.evaluate(testIter);
//            System.out.println(evalTest.stats());
//
//            //prediction
//            INDArray targetLabels = test.getLabels();
//            System.out.println("\nTarget shape: " + targetLabels.shapeInfoToString());
//
//            INDArray predictions = model.output(testIter);
//            System.out.println("\nPredictions shape: " + predictions.shapeInfoToString() + "\n");
//
//            System.out.println("Target \t\t\t Predicted");
//
//            for (int i = 0; i < 20; i++) {
//                System.out.println(targetLabels.getRow(i) + "\t\t" + predictions.getRow(i));
//            }

        }
    }
