package ai.certifai.FT_day6;

import javafx.scene.effect.InnerShadow;
import jdk.nashorn.internal.ir.CallNode;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class alienpredator {

    private static int seed = 123;
    private static Random rng = new Random(seed);
    private static final String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
    private static double TrainingPercentage = 0.8;
    private static int height = 70;
    private static int width = 70;
    private static int channel = 3;
    private static int batchSize = 10;
    private static int numclass = 2;
    private static double lr = 1e-3;
    private static int epochs = 20;


    public static void main(String[] args) throws IOException {

        //load the data
        File datafile = new ClassPathResource("alienpredators").getFile();
        FileSplit fileSplit = new FileSplit(datafile, allowedFormats, rng);

        ParentPathLabelGenerator labels = new ParentPathLabelGenerator();
        BalancedPathFilter bpf = new BalancedPathFilter(rng, allowedFormats, labels);

        InputSplit[] rawData = fileSplit.sample(bpf, TrainingPercentage, 1-TrainingPercentage);
        InputSplit TrainData = rawData[0];
        InputSplit TestData = rawData[1];

        ImageRecordReader trainRecRead = new ImageRecordReader(height, width, channel, labels);
        ImageRecordReader testRecRead = new ImageRecordReader(height, width, channel, labels);

        //prepare the iterator
        trainRecRead.initialize(TrainData);
        testRecRead.initialize(TestData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecRead, batchSize, 1, numclass);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecRead, batchSize, 1, numclass);

        //Data normalization
        DataNormalization scalar = new ImagePreProcessingScaler();
        trainIter.setPreProcessor(scalar);
        testIter.setPreProcessor(scalar);

        //model configuration
        MultiLayerConfiguration apconfig = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(lr))
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .nIn(channel)
                        .nOut(24)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .activation(Activation.RELU)
                .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                .build())
                .layer(2,new ConvolutionLayer.Builder()
                        .nOut(12)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .activation(Activation.RELU)
                .build())
                .layer(3,new ConvolutionLayer.Builder()
                        .nOut(12)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .activation(Activation.RELU)
                .build())
                .layer(4,new SubsamplingLayer.Builder()
                    .kernelSize(2,2)
                    .stride(2,2)
                    .poolingType(SubsamplingLayer.PoolingType.MAX)
                .build())
                .layer(5,new DenseLayer.Builder()
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                .build())
                .layer(6,new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nOut(numclass)
                .build())
                .setInputType(InputType.convolutional(height,width,channel))
                .build();

        //model training
        MultiLayerNetwork apmodel = new MultiLayerNetwork(apconfig);
        apmodel.init();

        System.out.println(apmodel.summary());

        //model evaluation
        Evaluation evaluationTrain = apmodel.evaluate(trainIter);
        Evaluation evaluationTest = apmodel.evaluate(testIter);

        System.out.println("Train Eval"+evaluationTrain.stats());
        System.out.println("Test Eval"+evaluationTest.stats());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        apmodel.setListeners(
                new StatsListener(statsStorage),
                new ScoreIterationListener(1),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );

                apmodel.fit(trainIter,epochs);
    }

}
