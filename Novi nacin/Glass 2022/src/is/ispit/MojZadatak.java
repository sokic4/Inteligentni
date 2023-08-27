package is.ispit;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.exam.NeurophExam;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/*
    ARI
*/

public class MojZadatak implements NeurophExam, LearningEventListener {

    int inputCount = 9;
    int outputCount = 7;
    DataSet trainSet;
    DataSet testSet;
    double[] learRate = {0.2, 0.4, 0.6};
    double learningRate;
    int[] hiddenNeurons = {10, 20, 30};
    int hiddenNeuron;
    ArrayList<Training> trainings = new ArrayList<>();

    /**
     * U ovoj metodi pozivati sve metode koje cete implementirati iz NeurophExam
     * interfejsa
     */
    private void run() {
        DataSet ds = loadDataSet();
        ds = preprocessDataSet(ds);
        DataSet[] trainAndTest = trainTestSplit(ds);
        trainSet = trainAndTest[0];
        testSet = trainAndTest[1];

        for (double lr : learRate) {
            learningRate = lr;
            for (int hn : hiddenNeurons) {
                hiddenNeuron = hn;
                MultiLayerPerceptron neuralNet = createNeuralNetwork();
                trainNeuralNetwork(neuralNet, ds);
            }
        }

        saveBestNetwork();
    }

    @Override
    public DataSet loadDataSet() {
        DataSet dataSet = DataSet.createFromFile("glass.csv", inputCount, outputCount, ",");
        return dataSet;
    }

    @Override
    public DataSet preprocessDataSet(DataSet ds) {
        Normalizer norm = new MaxNormalizer(ds);
        norm.normalize(ds);
        ds.shuffle();
        return ds;
    }

    @Override
    public DataSet[] trainTestSplit(DataSet ds) {
        return ds.split(0.65, 0.35);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(inputCount, hiddenNeuron, outputCount);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {

        int numOfIterations = 0;
        int numOfTrainings = 0;

        MomentumBackpropagation learningRule = (MomentumBackpropagation) mlp.getLearningRule();
        learningRule.addListener(this);

        learningRule.setLearningRate(learningRate);
        learningRule.setMomentum(0.6);
        learningRule.setMaxIterations(1000);

        mlp.learn(trainSet);

        numOfTrainings++;
        numOfIterations += learningRule.getCurrentIteration();

        evaluate(mlp, testSet);

        System.out.println("Srednja vrednost broja iteracija je: " + (double) numOfIterations / numOfTrainings);

        return mlp;

    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {

        String[] classLabels = new String[]{"c1", "c2", "c3", "c4", "c5", "c6", "c7"};
        ConfusionMatrix cm = new ConfusionMatrix(classLabels);
        double accuracy = 0;

        for (DataSetRow row : ds) {
            mlp.setInput(row.getInput());
            mlp.calculate();

            int actual = getMaxIndex(row.getDesiredOutput());
            int predicted = getMaxIndex(mlp.getOutput());

            cm.incrementElement(actual, predicted);
        }

        for (int i = 0; i < outputCount; i++) {
            accuracy += (double) (cm.getTruePositive(i) + cm.getTrueNegative(i)) / cm.getTotal();
        }

        System.out.println("\n" + hiddenNeuron + " skrivenih slojeva "
                + "neurona i learning rate parametar " + learningRate + " :\n");
        System.out.println(cm.toString());

        accuracy = (double) accuracy / outputCount;

        System.out.println("Moj accuracy: " + accuracy);

        Training t = new Training(mlp, accuracy);
        trainings.add(t);
    }

    @Override
    public void saveBestNetwork() {
        Training maxTraining = trainings.get(0);
        for (Training training : trainings) {
            if (training.getAccuracy() > maxTraining.getAccuracy()) {
                maxTraining = training;
            }
        }
        maxTraining.getNeuralNet().save("nn.nnet");
    }

    public static void main(String[] args) {
        new MojZadatak().run();
    }

    @Override
    public void handleLearningEvent(LearningEvent le) {
        BackPropagation bp = (BackPropagation) le.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration()
                + " Total network error: " + bp.getTotalNetworkError());
    }

    private int getMaxIndex(double[] output) {
        int max = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[max] < output[i]) {
                max = i;
            }
        }
        return max;
    }
 
/*
    ARI
*/

}
