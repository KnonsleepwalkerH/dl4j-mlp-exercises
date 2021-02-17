package ai.certifai.training.regression.simple_linear_regression;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class SimpleLinearRegression {
    private static int epoch = 2000;
    private static double lr = 1e-4;
    private static int seed = 1234;

    public static void main(String[] args) throws Exception{

        //load file
        File inputFile = new ClassPathResource("SimpleLinearRegression/SimpleLinearRegression.csv").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);
        CSVRecordReader csvRR = new CSVRecordReader(1, ',');
        csvRR.initialize(fileSplit);

        //build schema
        Schema schema = new Schema.Builder()
                .addColumnInteger("SAT")
                .addColumnDouble("GPA")
                .build();

        //transform process
        TransformProcess tp = new TransformProcess.Builder(schema)
                .build();

        //parse file into writable container
        List<List<Writable>> data = new ArrayList<>();
        while(csvRR.hasNext()){
            data.add(csvRR.next());
        }

        List<List<Writable>> transformed = LocalTransformExecutor.execute(data, tp);

        //preparing to split data
        CollectionRecordReader crr = new CollectionRecordReader(transformed);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(crr, transformed.size(), 1, 1, true);

        //shuffle data
        DataSet dataSet = dataIter.next();
        dataSet.shuffle(seed);

        //split data
        SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.7);
        DataSet trainingSet = splitTestAndTrain.getTrain();
        DataSet testSet = splitTestAndTrain.getTest();

        //assigning train set and test set into iterator
        ViewIterator trainIter = new ViewIterator(trainingSet, trainingSet.numExamples());
        ViewIterator testIter = new ViewIterator(testSet, testSet.numExamples());

        System.out.println(trainIter.inputColumns());

        //normalize the data
        DataNormalization scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        //Configure NN
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Sgd(lr))
                .weightInit(WeightInit.XAVIER)
//                .l2(5* 1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(1)
                        .activation(Activation.IDENTITY)
                        .nOut(64)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(64)
                        .activation(Activation.IDENTITY)
                        .nOut(128)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(128)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary());

        //set listener
        model.setListeners(new ScoreIterationListener(100));

        model.fit(trainIter, epoch);

        //predict
        INDArray predict = model.output(testIter);

        System.out.println("Predicted" + "\t" + "Ground Truth");
        for (int i = 0; i < predict.length(); i++) {
            System.out.println(predict.getRow(i) + "\t" + testSet.getLabels().getRow(i));
        }

        //evaluate using regression evaluation
        RegressionEvaluation evalTrain = model.evaluateRegression(trainIter);
        System.out.println("eval train:" + evalTrain.stats());

        RegressionEvaluation evalTest = model.evaluateRegression(testIter);
        System.out.println("eval test:" + evalTest.stats());

    }
}
