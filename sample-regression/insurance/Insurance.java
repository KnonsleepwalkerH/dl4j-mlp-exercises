package ai.certifai.training.regression.insurance;

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
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Insurance {
    private static int seed = 1234;
    private static int epochs = 30;
    private static double lr = 15e-4;
    private static int batchSize = 256;
    private static int hidden = 1000;

    public static void main(String[] args) throws Exception{
        //load file
        File inputFile = new ClassPathResource("insurance/insurance.csv").getFile();
        CSVRecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(inputFile));

        //build schema
        Schema schema = new Schema.Builder()
                .addColumnInteger("age")
                .addColumnCategorical("sex", Arrays.asList("female", "male"))
                .addColumnDouble("bmi")
                .addColumnInteger("children")
                .addColumnCategorical("smoker", Arrays.asList("yes", "no"))
                .addColumnCategorical("region", Arrays.asList("northeast", "southeast", "southwest", "northwest"))
                .addColumnDouble("charge")
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToInteger("sex", "smoker")
                .categoricalToOneHot("region")
                .build();

        System.out.println(tp.getInitialSchema());
        System.out.println(tp.getFinalSchema());

        List<List<Writable>> data = new ArrayList<>();
        while (rr.hasNext()){
            data.add(rr.next());
        }

        List<List<Writable>> transformed = LocalTransformExecutor.execute(data, tp);

        CollectionRecordReader crr = new CollectionRecordReader(transformed);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(crr, transformed.size(), 9, 9, true);

        DataSet dataSet = dataIter.next();
        dataSet.shuffle();

        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.7);
        DataSet trainSet = testAndTrain.getTrain();
        DataSet testSet = testAndTrain.getTest();

        ViewIterator trainIter = new ViewIterator(trainSet, batchSize);
        ViewIterator testIter = new ViewIterator(testSet, batchSize);

        //using normalizer in regression will affect the final reading which will separate the values far apart
        //not recommend in regression question

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
                        .nIn(trainIter.inputColumns())
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
