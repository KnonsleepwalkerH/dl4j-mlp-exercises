package ai.certifai.training.regression.USA_Housing;

import ai.certifai.solution.regression.PlotUtil;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class USA_Housing {
    private static int epochs = 10;
    private static int seed = 1234;
    private static double lr = 1e-4;
    private static int hidden = 1000;

    public static void main(String[] args) throws IOException, InterruptedException {
        //load file
        File file = new ClassPathResource("USA_Housing/USA_Housing_noquote.txt").getFile();
        FileSplit fileSplit = new FileSplit(file);
        CSVRecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(fileSplit);

        //build schema
        Schema schema = new Schema.Builder()
                .addColumnsDouble("Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms",
                        "Avg. Area Number of Bedrooms", "Area Population")
                .addColumnDouble("Price")
                .addColumnString("Address")
                .build();

        //transform process
        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("Address")
                .build();

        System.out.println(tp.getInitialSchema());
        System.out.println(tp.getFinalSchema());

        //parse into writable container
        List<List<Writable>> data = new ArrayList<>();
        while (rr.hasNext()){
            data.add(rr.next());
        }

        List<List<Writable>> transformed = LocalTransformExecutor.execute(data, tp);

        System.out.println(data.size());
        System.out.println(transformed.size());

        //prepare to split data
        CollectionRecordReader crr = new CollectionRecordReader(transformed);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(crr, transformed.size(), 5, 5, true);

        DataSet dataSet = dataIter.next();
        dataSet.shuffle();

        //split the data
        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.8);
        DataSet trainSet = testAndTrain.getTrain();
        DataSet testSet = testAndTrain.getTest();

        //assigning into view iterator
        ViewIterator trainIter = new ViewIterator(trainSet, transformed.size());
        ViewIterator testIter = new ViewIterator(testSet, transformed.size());

//        //normalize the data
//        DataNormalization scaler = new NormalizerMinMaxScaler();
//        scaler.fit(trainIter);
//        trainIter.setPreProcessor(scaler);
//        testIter.setPreProcessor(scaler);

        //configure nn
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.00001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(5)
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

        //  Fitting the model for nEpochs
        for (int i=0; i<epochs; i++) {
            trainIter.reset();
            System.out.println("epoch: " + i);
            model.fit(trainIter);


            //  Evaluating the outcome of our trained model
            System.out.println("===validating===");
            RegressionEvaluation regEval = model.evaluateRegression(testIter);
            System.out.println(regEval.stats());
        }

        testIter.reset();

        //predict
        INDArray targetLabels = testSet.getLabels();
        INDArray predict = model.output(testIter);

        System.out.println("Target: " + "\t\t" + "Predicted:");
        for (int i = 0; i < targetLabels.rows(); i++) {
            System.out.println(targetLabels.getRow(i) + "\t" + predict.getRow(i));
        }
        // Plot the target values and predicted values
        PlotUtil.visualizeRegression(targetLabels, predict);
    }
}
